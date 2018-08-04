import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_utils as U

class RNN(nn.Module):
  def __init__(self, vocab_size, args, device):
    super(RNN, self).__init__()
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

    if args.type == 'jordan':
      self.b_h = nn.Linear(1, self.embed_dim)
      self.start_1hot = torch.zeros(1, vocab_size).to(device)
      #self.start_1hot[0, START] = 1   #TODO is start first symbol in input seq? should not use here
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


  def forward(self, x):
    w = x.clone()
    x = self.embeddings(x)
    if self.condition == 'lstm':
      x, _ = self.cond(x)
    x = x.permute(0, 2, 1)
    N = x.size()[0] # args.batch_size
    T = w.size()[1]

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
        h_t = U.log_sum_exp(h_t, 1)

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


