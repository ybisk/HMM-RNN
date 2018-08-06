import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_utils as U

class HMM(nn.Module):
  def __init__(self, vocab_size, args, device):
    super(HMM, self).__init__()
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
          cur_alpha = U.log_sum_exp(cur_alpha, 1)
          cur_alpha = F.log_softmax(cur_alpha, dim=1).clone()
        else:
          tran = F.log_softmax(self.trans(x[:,:,t-1]).view(N, K, K), dim=-1)
          cur_alpha = pre_alpha.unsqueeze(-1).expand(N, K, K) + tran
          cur_alpha = U.log_sum_exp(cur_alpha, 1)


      # Emission
      word_idx = w[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
      emit_prob = Emissions[:, self.Kr].gather(2, word_idx).squeeze()
      cur_alpha[:, self.Kr] = cur_alpha[:, self.Kr] + emit_prob

      # Update
      pre_alpha = cur_alpha.clone()

    # TODO – Perplexity 2^-Sum(p * log2(p))
    return -1 * torch.mean(U.log_sum_exp(cur_alpha, dim=1))

  def emissions_list(self, index):
    V = F.log_softmax(self.vocab(self.emb_cluster(self.Cs[index])), dim=-1)
    e_list = [torch.exp(V[0][j]).data.item() for j in range(self.vocab_size)]
    return e_list

