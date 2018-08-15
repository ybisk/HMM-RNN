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

    self.dummy = torch.ones(args.batch_size, 1).to(device)

    # Embedding parameters
    self.embed = nn.Embedding(vocab_size, self.embed_dim)
    if args.glove_emb:
      self.embed.weight.data.copy_(
          torch.from_numpy(np.load("inference/infer_glove.npy")))
      self.embed.requires_grad = False

    if args.feeding == 'encode-lstm':
      self.encode_context = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)

    # Transition cell

    if args.type == 'elman' or args.type == 'jordan' or 'rnn' in args.type:
      if args.type == 'rnn-1' or args.type == 'rnn-2': 
        nonlin = 'softmax'
      elif args.type == 'rnn-1a':
        nonlin = 'sigmoid'
      else: 
        nonlin = 'tanh'

      self.trans = cell.ElmanCell(self.embed_dim, self.hidden_dim, nonlin,
          feed_input = (self.feeding != 'none'), 
          trans_use_input_dim = (args.type == 'jordan'),
          trans_only_nonlin = (args.type == 'rnn-2'))
    elif args.type == 'gru':
      self.trans = nn.GRUCell(self.embed_dim, self.hidden_dim)
    elif args.type == 'lstm':
      self.trans = nn.LSTMCell(self.embed_dim, self.hidden_dim)
    else: 
      print(args.type + " is not implemented")
      sys.exit()

    if args.type == 'jordan':
      self.start_1hot = torch.zeros(1, vocab_size).to(device)
      # self.start_1hot[0, 0] = START #TODO use START symbol index
      self.start_1hot.requires_grad = False

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
      y_tm1 = self.start_1hot.expand(N, self.vocab_size)   # One hot start
      h_tm1 = y_tm1 @ self.embed.weight 

    for t in range(1, T):
      inp = zero if self.feeding == 'none' else x[:,:,t-1] 

      if self.type == 'lstm':
        h_t, c_t = self.trans(inp, (h_tm1, c_tm1))
        c_tm1 = c_t.clone()
      else:
        h_t = self.trans(inp, h_tm1)

      # y_t = act(W_y h_t + b_y)
      logits_t = self.emit(h_t)
      y_t = F.log_softmax(logits_t, -1)        # Emission

      # print(logits_t)
      word_idx = w[:, t].unsqueeze(1)
      x_t = y_t.gather(1, word_idx).squeeze()           # Word Prob
      cur_alpha += x_t

      y_tm1 = y_t.clone()

      if self.type == 'jordan':
        # h_t = act(W_h x_t + U_h y_t-1 + b_h)
        # Replace hidden state so that other computations are equivalent to Elman
        h_tm1 = y_tm1 @ self.embed.weight #TODO why use log probabilities?
      else:
        h_tm1 = h_t.clone() # (Jan) why are we cloning here?
    return -1 * torch.mean(cur_alpha)


