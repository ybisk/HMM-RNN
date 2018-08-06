import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_utils as U
import cell

class HMM(nn.Module):
  def __init__(self, vocab_size, args, device):
    super(HMM, self).__init__()
    self.embed_dim = args.embed_dim
    self.vocab_size = vocab_size
    self.num_clusters = args.hidden_dim
    self.type = args.type
    self.feeding = args.feeding
    self.device = device

    self.Kr = torch.from_numpy(np.array(range(self.num_clusters)))
    self.dummy = torch.ones(args.batch_size, 1).to(device)

    # Input embedding parameters
    self.embed = nn.Embedding(vocab_size, self.embed_dim)
    if args.glove_emb:
      self.embed.weight.data.copy_(
          torch.from_numpy(np.load("inference/infer_glove.npy")))
      self.embed.requires_grad = False

    if args.feeding == 'encode-lstm':
      self.encode_context = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)

    # Transition parameters
  
    #TODO (Jan) not clear about this part
    # Cluster vectors
    self.Cs = []
    for i in range(self.num_clusters):
      self.Cs.append(torch.zeros(args.batch_size).long().to(self.device) + i) #TODO do without to(device)?

    # Embed clusters
    self.one_hot = False
    if self.one_hot:  # (Jan) ?
      self.emb_cluster = nn.embedding(self.num_clusters, self.num_clusters)
      self.emb_cluster.weight.data = torch.eye(self.num_clusters)
      self.emb_cluster.requires_grad = False
    else:
      self.emb_cluster = nn.Embedding(self.num_clusters, self.num_clusters)

    if self.type == 'hmm-new':
      self.trans = cell.HMMCell(self.embed_dim, self.num_clusters, self.feeding != 'none')
    else:
      if args.feeding == 'none':
        self.trans = nn.Linear(1, self.num_clusters**2, bias=False)         # f(cluster, cluster)
      else:
        self.trans = nn.Linear(self.embed_dim, self.num_clusters**2, bias=False)         # f(cluster, cluster)
      self.trans.weight.data.uniform_(-1, 1)                              # 0 to e (logspace)

    # HMM Start Probabilities
    self.start = nn.Linear(1, self.num_clusters)

    # Emission parameters
    self.emit = nn.Linear(self.num_clusters, vocab_size, bias=False)   # f(cluster, word)
    #TODO (Jan) implement emit bias?
    self.emit.weight.data.uniform_(-1, 1)                              # Otherwise root(V) is huge

  def embed_input(self, x):
    embed_x = self.embed(x)
    if 'encode' in self.feeding:
      embed_x, _ = self.encode_context(embed_x)
    return embed_x.permute(0, 2, 1) # batch_size x embed_dim x seq_length 

  def forward(self, x):
    w = x.clone()
    N = x.size()[0] # batch size
    T = w.size()[1] # sequence length
    K = self.num_clusters

    # Embed
    x = self.embed_input(x)

    if self.feeding == 'none':
      tran = F.log_softmax(self.trans(self.dummy).view(N, K, K), dim=-1)

    zero = torch.zeros(N).to(self.device)
    pre_alpha = torch.zeros(N, K).to(self.device)
    cur_alpha = torch.zeros(N, K).to(self.device)
    if self.type == 'hmm+1':
      pre_alpha = self.start(self.dummy).expand(N,K)
    else:
      pre_alpha = F.log_softmax(self.start(self.dummy).expand(N,K), dim=-1)

    #TODO (Jan) is there a cleaner way to implement this?
    #TODO (Jan) not understanding the role of Cs. Shouldn't this be deterministic?
    Emissions = torch.stack([
        F.log_softmax(self.emit(self.emb_cluster(self.Cs[i])), -1)
        for i in range(K)])
    Emissions = Emissions.transpose(0, 1)    # Move batch to the front

    for t in range(1, T):
      # Transition
      if self.type == 'hmm-new':
         if self.feeding == 'none':
           tran = self.trans(zero, pre_alpha)
         else:
           tran = self.trans(x[:,:,t-1], pre_alpha)

      elif self.feeding != 'none': #TODO (Jan) is feeding == none implemented?
        if self.type == 'hmm+1':
          tran = self.trans(x[:,:,t-1]).view(N, K, K)
          cur_alpha = pre_alpha.unsqueeze(-1).expand(N, K, K) + tran
          cur_alpha = U.log_sum_exp(cur_alpha, 1)
          cur_alpha = F.log_softmax(cur_alpha, dim=1)
          
          # Yonatan version:  
          #tran = self.trans(x[:,:,t-1]).view(N, K, K)
          #pre_alpha = pre_alpha.unsqueeze(2)
          #cur_alpha = F.log_softmax(tran @ pre_alpha, dim=-1)
          #cur_alpha = cur_alpha.clone().squeeze()
        else:
          tran = F.log_softmax(self.trans(x[:,:,t-1]).view(N, K, K), dim=-1)
          cur_alpha = pre_alpha.unsqueeze(-1).expand(N, K, K) + tran
          cur_alpha = U.log_sum_exp(cur_alpha, 1)

      # Emission

      # self.Kr is the (full) range of clusters
      word_idx = w[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
      emit_prob = Emissions[:, self.Kr].gather(2, word_idx).squeeze()
      cur_alpha[:, self.Kr] = cur_alpha[:, self.Kr] + emit_prob

      # Update

      # TODO (Jan) I think we are missing a normalization and/or accummulation step (as in my equations)
      pre_alpha = cur_alpha.clone()

    return -1 * torch.mean(U.log_sum_exp(cur_alpha, dim=1))

  def emissions_list(self, index):
    V = F.log_softmax(self.emit(self.emb_cluster(self.Cs[index])), dim=-1)
    e_list = [torch.exp(V[0][j]).data.item() for j in range(self.vocab_size)]
    return e_list

