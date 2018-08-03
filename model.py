import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Jan Buys Code
def log_sum_exp(vec, dim):
   # sum over dim
   max_score, _ = torch.max(vec, dim, keepdim=True)
   max_score_broadcast = max_score.expand(vec.size())
   return max_score.squeeze(dim) + torch.log(torch.sum(
       torch.exp(vec - max_score_broadcast), dim))

"""
  Model
"""
class Net(nn.Module):
  def __init__(self, vocab_size, args, device):
    super(Net, self).__init__()
    self.hidden = args.hidden_dim
    self.embed_dim = 100
    self.vocab_size = vocab_size
    self.num_clusters = args.clusters
    self.type = args.type
    self.condition = args.condition
    self.device = device

    self.Kr = torch.from_numpy(np.array(range(self.num_clusters)))
    self.dummy = torch.ones(args.batch_size, 1).to(device)

    # vocab x 100
    self.embeddings = nn.Embedding(vocab_size, self.embed_dim)
    if args.glove_emb:
      self.embeddings.weight.data.copy_(
          torch.from_numpy(np.load("inference/infer_glove.npy")))
      self.embeddings.requires_grad = False

    if args.condition == 'lstm':
      self.cond = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)

    if 'hmm' not in args.type:

      if args.type == 'jordan':
        self.b_h = nn.Linear(1, self.embed_dim)
        self.start_1hot = torch.zeros(1, vocab_size).to(device)
        self.start_1hot[0, START] = 1
        self.start_1hot.requires_grad = False

      elif args.type == 'elman':
        self.trans = nn.Linear(self.embed_dim, self.embed_dim)

      elif args.type == 'rnn-1':
        self.trans = nn.Linear(self.embed_dim, self.embed_dim)

      elif args.type == 'rnn-2':
        self.trans = nn.Linear(self.embed_dim, self.embed_dim)

      elif args.type == 'dist':
        self.trans = nn.Linear(1, self.embed_dim**2, bias=False)
        self.trans.weight.data.uniform_(-1, 1)

      elif args.type == 'gru':
        self.trans = nn.GRUCell(self.embed_dim, self.embed_dim)

      elif args.type == 'lstm':
        self.trans = nn.LSTMCell(self.embed_dim, self.embed_dim)

      elif args.type == 'ran':
        print("RANs are not implemented")
        sys.exit()


      self.vocab = nn.Linear(self.embed_dim, vocab_size, bias=True)    # f(cluster, word)
      self.vocab.weight.data.uniform_(-1, 1)                           # Otherwise root(V) is huge

    else:
      # Cluster vectors
      self.Cs = []
      for i in range(self.num_clusters):
        self.Cs.append(torch.zeros(args.batch_size).long().to(self.device) + i) #TODO do without to(device)?

      # Embed clusters
      self.one_hot = False
      if self.one_hot:
        self.emb_cluster = nn.embedding(self.num_clusters, self.num_clusters)
        self.emb_cluster.weight.data = torch.eye(self.num_clusters)
        self.emb_cluster.requires_grad = False
      else:
        self.emb_cluster = nn.Embedding(self.num_clusters, self.num_clusters)

      # HMM Start Probabilities
      self.start = nn.Linear(1, self.num_clusters)

      self.vocab = nn.Linear(self.num_clusters, vocab_size, bias=False)   # f(cluster, word)
      self.vocab.weight.data.uniform_(-1, 1)                              # Otherwise root(V) is huge
      if args.condition == 'none':
        self.trans = nn.Linear(1, self.num_clusters**2, bias=False)         # f(cluster, cluster)
      else:
        self.trans = nn.Linear(self.embed_dim, self.num_clusters**2, bias=False)         # f(cluster, cluster)
      self.trans.weight.data.uniform_(-1, 1)                              # 0 to e (logspace)


  def forward(self, x):
    w = x.clone()
    x = self.embeddings(x)
    if self.condition == 'lstm':
      x, _ = self.cond(x)
    x = x.permute(0, 2, 1)
    N = x.size()[0] # args.batch_size
    T = w.size()[1]

    if 'hmm' not in self.type:
      cur_alpha = torch.zeros(N).to(self.device)
      zeros = torch.zeros(N, self.embed_dim).to(self.device)
      h_tm1 = torch.zeros(N, self.embed_dim).to(self.device)
      if self.type == 'lstm':
        c_tm1 = torch.zeros(N, self.embed_dim).to(self.device)

      if self.type == 'jordan':
        b_h = self.b_h(self.dummy)
        y_tm1 = self.start_1hot.expand(N, self.vocab_size)   # One hot start

      for t in range(1, T):
        if self.type == 'jordan':
          # h_t = act(W_h x_t + U_h y_t-1 + b_h)
          if self.condition == 'word':
            h_t = F.tanh(x[:,:,t-1] +  y_tm1 @ self.embeddings.weight + b_h)
          else:
            h_t = F.tanh(zeros +  y_tm1 @ self.embeddings.weight + b_h)

        elif self.type == 'elman':
          # h_t = act(W_h x_t + U_h h_t-1 + b_h)
          if self.condition == 'word':
            h_t = F.tanh(x[:,:,t-1] + self.trans(h_tm1))
          else:
            h_t = F.tanh(zeros + self.trans(h_tm1))

        elif self.type == 'rnn-1':
          if self.condition == 'word':
            h_t = F.softmax(x[:,:,t-1] + self.trans(h_tm1))
          else:
            h_t = F.softmax(zeros + self.trans(h_tm1))

        elif self.type == 'rnn-2':
          if self.condition == 'word':
            h_t = x[:,:,t-1] * F.softmax(self.trans(h_tm1))
          else:
            h_t = F.softmax(self.trans(h_tm1))

        elif self.type == 'dist':
          # if h_t-1 is a distribution and multiply by transition matrix
          K = self.embed_dim
          tran = F.log_softmax(self.trans(self.dummy).view(N, K, K), dim=-1)
          h_t = h_tm1.unsqueeze(1).expand(N, K, K) + tran
          h_t = log_sum_exp(h_t, 1)

        elif self.type == 'gru':
          if self.condition == 'word':
            h_t = self.trans(x[:,:,t-1], h_tm1)
          else:
            h_t = self.trans(zeros, h_tm1)

        elif self.type == 'lstm':
          if self.condition == 'word':
            h_t, c_t = self.trans(x[:,:,t-1], (h_tm1, c_tm1))
            c_tm1 = c_t.clone()
          else:
            h_t, c_t = self.trans(zeros, (h_tm1, c_tm1))
            c_tm1 = c_t.clone()

        # y_t = act(W_y h_t + b_y)
        y_t = F.log_softmax(self.vocab(h_t), -1)        # Emission

        word_idx = w[:, t].unsqueeze(1)
        cur_alpha += y_t.gather(1, word_idx).squeeze()           # Word Prob

        y_tm1 = y_t.clone()
        h_tm1 = h_t.clone()
      return -1 * torch.mean(cur_alpha)

    elif 'hmm' in self.type:
      K = self.num_clusters

      if self.condition == 'none':
        tran = F.log_softmax(self.trans(self.dummy).view(N, K, K), dim=-1)

      pre_alpha = torch.zeros(N, K).to(self.device)
      cur_alpha = torch.zeros(N, K).to(self.device)
      if self.type == 'hmm+1':
        pre_alpha = self.start(self.dummy).expand(N,K)
      else:
        pre_alpha = F.log_softmax(self.start(self.dummy).expand(N,K), dim=-1)

      Emissions = torch.stack([
          F.log_softmax(self.vocab(self.emb_cluster(self.Cs[i])), -1)
          for i in range(K)])
      Emissions = Emissions.transpose(0, 1)    # Move batch to the front

      for t in range(1, T):
        if self.condition != 'none':
          # Transition
          if self.type == 'hmm+1':
            tran = self.trans(x[:,:,t-1]).view(N, K, K)
            cur_alpha = pre_alpha.unsqueeze(-1).expand(N, K, K) + tran
            cur_alpha = log_sum_exp(cur_alpha, 1)
            cur_alpha = F.log_softmax(cur_alpha, dim=1)
            
            # Yonatan version:  
            #tran = self.trans(x[:,:,t-1]).view(N, K, K)
            #pre_alpha = pre_alpha.unsqueeze(2)
            #cur_alpha = F.log_softmax(tran @ pre_alpha, dim=-1)
            #cur_alpha = cur_alpha.clone().squeeze()
          else:
            tran = F.log_softmax(self.trans(x[:,:,t-1]).view(N, K, K), dim=-1)
            cur_alpha = pre_alpha.unsqueeze(-1).expand(N, K, K) + tran
            cur_alpha = log_sum_exp(cur_alpha, 1)


        # Emission
        word_idx = w[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
        emit_prob = Emissions[:, self.Kr].gather(2, word_idx).squeeze()
        cur_alpha[:, self.Kr] = cur_alpha[:, self.Kr] + emit_prob

        # Update
        pre_alpha = cur_alpha.clone()

      # TODO – Perplexity 2^-Sum(p * log2(p))
      return -1 * torch.mean(log_sum_exp(cur_alpha, dim=1))

  def print_emissions(self, fname, i2voc):
    o = open("Emissions.{}.txt".format(fname),'w')
    for i in range(self.num_clusters):
      V = F.log_softmax(self.vocab(self.emb_cluster(self.Cs[i])), dim=-1)
      listed = [(torch.exp(V[0][j]).data.item(), str(i2voc[j])) for j in
              range(self.vocab_size)]
      listed.sort()
      listed.reverse()
      o.write("\n%d\n" % i)
      for prob, word in listed[:50]:
        o.write("   {:10.8f}  {:10s}\n".format(100*prob, str(word)))
    o.close()




