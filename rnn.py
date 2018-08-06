import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_utils as U
import cell 

class RNN(nn.Module):
  def __init__(self, vocab_size, args, device):
    super(RNN, self).__init__()
    self.hidden_dim = args.hidden_dim
    self.embed_dim = args.embed_dim
    self.vocab_size = vocab_size
    self.type = args.type
    self.feeding = args.feeding 
    self.device = device

    # self.Kr = torch.from_numpy(np.array(range(self.num_clusters))) #(Jan) ?
    self.dummy = torch.ones(args.batch_size, 1).to(device)

    # Embedding parameters
    self.embed = nn.Embedding(vocab_size, self.embed_dim)
    if args.glove_emb:
      self.embed.weight.data.copy_(
          torch.from_numpy(np.load("inference/infer_glove.npy")))
      self.embed.requires_grad = False

    if args.feeding == 'encode-lstm':
      self.encode_context = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)

    # Transition parameters

    if args.type == 'elman' or self.type == 'rnn-1' or self.type == 'rnn-1a': 
      # new implementation
      if args.type == 'rnn-1':
        nonlin = 'softmax'
      elif args.type == 'rnn-1a':
        nonlin = 'sigmoid'
      else: #if args.type == 'elman':
        nonlin = 'tanh'

      self.trans = cell.ElmanCell(self.embed_dim, self.hidden_dim, nonlin, self.feeding != 'none')
    elif args.type == 'gru':
      self.trans = nn.GRUCell(self.embed_dim, self.hidden_dim)
    elif args.type == 'lstm':
      self.trans = nn.LSTMCell(self.embed_dim, self.hidden_dim)

    # still old implementation
    elif args.type == 'jordan':
      self.b_h = nn.Linear(1, self.embed_dim)
      self.start_1hot = torch.zeros(1, vocab_size).to(device)
      #self.start_1hot[0, START] = 1   #TODO is start first symbol in input seq? should not use here
      self.start_1hot.requires_grad = False

    elif args.type == 'rnn-2':
      self.trans = nn.Linear(self.embed_dim, self.hidden_dim)

    elif args.type == 'dist':  #TODO (Jan) unclear
      self.trans = nn.Linear(1, self.embed_dim**2, bias=False)
      self.trans.weight.data.uniform_(-1, 1)

    else: 
      # if args.type == 'ran':
      print(args.type + " is not implemented")
      sys.exit()

    # Emission parameters
    self.emit = nn.Linear(self.hidden_dim, vocab_size, bias=True)    # f(cluster, word)
    self.emit.weight.data.uniform_(-1, 1)                           # Otherwise root(V) is huge

  def embed_input(self, x):
    embed_x = self.embed(x)
    if 'encode' in self.feeding:
      embed_x, _ = self.encode_context(embed_x)
    return embed_x.permute(0, 2, 1) # batch_size x embed_dim x seq_length 

  def forward(self, x):
    w = x.clone()
    N = x.size()[0] # batch size
    T = w.size()[1] # sequence length

    #TODO find better ways of dealing with zeros
    #TODO (Jan) implement proper hidden state initialization
    cur_alpha = torch.zeros(N).to(self.device)
    zeros = torch.zeros(N, self.embed_dim).to(self.device)
    h_tm1 = torch.zeros(N, self.hidden_dim).to(self.device)

    # Embed
    x = self.embed_input(x)

    if self.type == 'lstm':
      c_tm1 = torch.zeros(N, self.hidden_dim).to(self.device)

    if self.type == 'jordan':
      b_h = self.b_h(self.dummy)
      y_tm1 = self.start_1hot.expand(N, self.vocab_size)   # One hot start

    for t in range(1, T):

      if self.type == 'elman' or self.type == 'rnn-1' or self.type == 'rnn-1a': 
        h_t = self.trans(x[:,:,t-1], h_tm1)

      elif self.type == 'gru':
        if self.feeding == 'word':
          h_t = self.trans(x[:,:,t-1], h_tm1)
        else:
          h_t = self.trans(zeros, h_tm1)

      elif self.type == 'lstm': #TODO (Jan) do we need to store c_tm1 seperately?
        if self.feeding == 'word':
          h_t, c_t = self.trans(x[:,:,t-1], (h_tm1, c_tm1))
          c_tm1 = c_t.clone()
        else:
          h_t, c_t = self.trans(zeros, (h_tm1, c_tm1))
          c_tm1 = c_t.clone()

      # Still in old implementation:
      elif self.type == 'jordan': #TODO (Jan) this definition is unclear to me
        # h_t = act(W_h x_t + U_h y_t-1 + b_h)
        #TODO (Jan) don't understand @ notation
        if self.feeding == 'word':
          h_t = F.tanh(x[:,:,t-1] + y_tm1 @ self.embed.weight + b_h)
        else:
          h_t = F.tanh(zeros +  y_tm1 @ self.embedweight + b_h)

      #elif self.type == 'elman': 
      #  h_t = act(W_h x_t + U_h h_t-1 + b_h)  # (Jan) where is W_h?
      #  if self.feeding == 'word':
      #    h_t = F.tanh(x[:,:,t-1] + self.trans(h_tm1))
      #  else:
      #    h_t = F.tanh(zeros + self.trans(h_tm1))

      elif self.type == 'rnn-2':
        assert self.hidden_dim == self.embed_dim
        if self.feeding == 'word': #TODO implement as cell (Jan) unclear
          h_t = x[:,:,t-1] * F.softmax(self.trans(h_tm1), dim=-1)
        else:
          h_t = F.softmax(self.trans(h_tm1), dim=-1)

      elif self.type == 'dist': #TODO implement as seperate cell
        assert self.hidden_dim == self.embed_dim
        # if h_t-1 is a distribution and multiply by transition matrix
        K = self.embed_dim
        tran = F.log_softmax(self.trans(self.dummy).view(N, K, K), dim=-1)
        h_t = h_tm1.unsqueeze(1).expand(N, K, K) + tran
        h_t = U.log_sum_exp(h_t, 1)

      # y_t = act(W_y h_t + b_y)
      y_t = F.log_softmax(self.emit(h_t), -1)        # Emission

      word_idx = w[:, t].unsqueeze(1)
      cur_alpha += y_t.gather(1, word_idx).squeeze()           # Word Prob

      y_tm1 = y_t.clone()
      h_tm1 = h_t.clone() # (Jan) why are we cloning here?
    return -1 * torch.mean(cur_alpha)


