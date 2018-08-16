import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cell 

class RNN(nn.Module):
  def __init__(self, vocab_size, args):
    super(RNN, self).__init__()
    self.hidden_dim = args.hidden_dim
    self.embed_dim = args.embed_dim
    self.vocab_size = vocab_size
    self.type = args.type
    self.feeding = args.feeding 
    self.glove_emb = args.glove_emb

    # Input embedding parameters
    self.embed = nn.Embedding(vocab_size, self.embed_dim)
    if args.glove_emb:
      self.embed.weight.data.copy_(
          torch.from_numpy(np.load("inference/infer_glove.npy")))
      self.embed.requires_grad = False

    if args.feeding == 'encode-lstm':
      self.encode_context = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)

    # Transition cell

    if args.type == 'elman' or 'rnn' in args.type:
      if args.type == 'rnn-1' or args.type == 'rnn-2': 
        nonlin = 'softmax'
      elif args.type == 'rnn-1a':
        nonlin = 'sigmoid'
      else: 
        nonlin = 'tanh'
      self.trans = cell.ElmanCell(self.embed_dim, self.hidden_dim, nonlin,
          feed_input = (self.feeding != 'none'), 
          trans_only_nonlin = (args.type == 'rnn-2'))
    elif args.type == 'jordan':
      nonlin = 'tanh'
      self.trans = cell.JordanCell(self.embed_dim, self.hidden_dim, nonlin,
          feed_input = (self.feeding != 'none'))
    elif args.type == 'gru':
      self.trans = nn.GRUCell(self.embed_dim, self.hidden_dim)
    elif args.type == 'lstm':
      self.trans = nn.LSTMCell(self.embed_dim, self.hidden_dim)
    else: 
      print(args.type + " is not implemented")
      sys.exit()

    # Emission parameters
    self.emit = nn.Linear(self.hidden_dim, vocab_size, bias=True) 

    self.init_weights()

  def init_weights(self, initrange=1.0):
    self.emit.weight.data.uniform_(-initrange, initrange)    # Otherwise root(V) is huge
    self.emit.bias.data.zero_()
    if not self.glove_emb:
      self.embed.weight.data.uniform_(-initrange, initrange) 

  def init_hidden_state(self, batch_size, inp=None):
    weight = next(self.parameters())
    if self.type == 'jordan':
      assert inp is not None
      state = weight.new_zeros((batch_size, self.vocab_size))
      state.scatter_(1, inp.view(batch_size, 1), 1)
      return state
    if self.type == 'lstm':
      return (weight.new_zeros((batch_size, self.hidden_dim)),
              weight.new_zeros((batch_size, self.hidden_dim)))
    else:
      return weight.new_zeros((batch_size, self.hidden_dim))

  def embed_input(self, words):
    emb = self.embed(words)
    if 'encode' in self.feeding:
      emb, _ = self.encode_context(embed)
    return emb.permute(0, 2, 1) # batch_size x embed_dim x seq_length 

  def forward(self, words, hidden_state):
    N = words.size()[0] # batch size
    T = words.size()[1] # sequence length
    emit_marginal = None

    # feeding is none doesn't make sense for RNNs, so why bother for LSTM/GRU?
    if self.type == 'lstm' or self.type == 'gru':
      assert self.feeding != 'none'

    # Embed
    emb = self.embed_input(words)

    for t in range(1, T):
      state_input = emb[:,:,t-1] 

      # Transition
      if self.type == 'lstm':
        hidden_output, hidden_memcell = self.trans(state_input, hidden_state)
      elif self.type == 'jordan':
        # h_t = act(W_h x_t + U_h y_t-1 + b_h)
        #TODO move to cell if we can tie weights
        hidden_input_state = hidden_state @ self.embed.weight
        hidden_output = self.trans(state_input, hidden_input_state)
      else:  
        hidden_output = self.trans(state_input, hidden_state)

      # Emit

      # y_t = act(W_y h_t + b_y)
      logits = self.emit(hidden_output)
      emit_distr = F.log_softmax(logits, -1)        # Emission

      word_idx = words[:, t].unsqueeze(1)
      emit_ll = emit_distr.gather(1, word_idx).squeeze()           # Word Prob

      # State Update
      #TODO do we need to clone here?
      if self.type == 'lstm':
        hidden_state = (hidden_output.clone(), hidden_memcell.clone())
      elif self.type == 'jordan':
        hidden_state = F.softmax(logits, -1)
      else:
        hidden_state = hidden_output.clone() 

      # Accumulate
      if t == 1:
        emit_marginal = emit_ll
      else:
        emit_marginal = emit_marginal + emit_ll

    return emit_marginal, hidden_state

