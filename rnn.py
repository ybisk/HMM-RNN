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
    self.logspace_hidden = True #TODO experiment with this

    # Input embedding parameters
    self.embed = nn.Embedding(vocab_size, self.embed_dim)
    if args.glove_emb:
      self.embed.weight.data.copy_(
          torch.from_numpy(np.load("inference/infer_glove.npy")))
      self.embed.requires_grad = False

    if args.feeding == 'encode-lstm':
      self.encode_context = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)

    # Transition cell
    if 'hmm' in args.type:
      self.trans = cell.HMMCell(self.embed_dim, self.hidden_dim, 
                                logspace_hidden = self.logspace_hidden,
                                feed_input = (self.feeding != 'none'), 
                                delay_trans_softmax = (self.type == 'hmm-1'))
    elif args.type == 'elman' or 'rnn' in args.type:
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
    if 'hmm' in self.type:
      # Uniform distribution over clusters
      return weight.new_full((batch_size, self.hidden_dim), 1/self.hidden_dim)
    elif self.type == 'jordan':
      assert inp is not None
      state = weight.new_zeros((batch_size, self.vocab_size))
      state.scatter_(1, inp.view(batch_size, 1), 1)
      return state
    if self.type == 'lstm':
      return (weight.new_zeros((batch_size, self.hidden_dim)),
              weight.new_zeros((batch_size, self.hidden_dim)))
    else:
      return weight.new_zeros((batch_size, self.hidden_dim))

  def emissions_list(self):
    emit_distr = F.log_softmax(self.emit.weight, 0) # dim vocab_size x hidden_size
    return emit_distr.detach().cpu().numpy()

  def embed_input(self, words):
    emb = self.embed(words)
    if 'encode' in self.feeding:
      emb, _ = self.encode_context(embed)
    return emb.permute(0, 2, 1) # batch_size x embed_dim x seq_length 

  def rnn_step(self, state_input, hidden_state, word_idx):
    # Transition
    if self.type == 'lstm':
      hidden_output, hidden_memcell = self.trans(state_input, hidden_state)
    elif self.type == 'jordan':
      #TODO move to cell if we can tie weights
      hidden_input_state = hidden_state @ self.embed.weight
      hidden_output = self.trans(state_input, hidden_input_state)
    else:  
      hidden_output = self.trans(state_input, hidden_state)

    # Emit
    logits = self.emit(hidden_output)
    emit_distr = F.log_softmax(logits, -1)        # Emission

    emit_ll = emit_distr.gather(1, word_idx).squeeze()           # Word Prob

    # State Update
    if self.type == 'lstm':
      hidden_state = (hidden_output, hidden_memcell)
    elif self.type == 'jordan':
      hidden_state = F.softmax(logits, -1) # not log_softmax
    else:
      hidden_state = hidden_output 

    return emit_ll, hidden_state

  def hmm_step(self, state_input, hidden_state, emit_distr, word_idx):
    N = word_idx.size()[0] # batch size

    # Transition
    hidden_output = self.trans(state_input, hidden_state)

    # Emit 
    word_idx = word_idx.expand(N, self.hidden_dim)
    emit_state_ll = emit_distr.gather(0, word_idx) # batch_size x hidden_dim

    # State Update
    if self.logspace_hidden:
      joint_state_ll = hidden_output + emit_state_ll
      emit_ll = torch.logsumexp(joint_state_ll, 1)
      hidden_state = joint_state_ll - emit_ll.unsqueeze(1).expand(N, self.hidden_dim)
    else:
      joint_state_ll = torch.log(hidden_output) + emit_state_ll
      emit_ll = torch.logsumexp(joint_state_ll, 1)
      hidden_state = torch.exp(joint_state_ll - emit_ll.unsqueeze(1).expand(N, self.hidden_dim))

    return emit_ll, hidden_state

  def forward(self, words, hidden_state):
    N = words.size()[0] # batch size
    T = words.size()[1] # sequence length
    emit_marginal = None

    # feeding is none doesn't make sense for RNNs, so why bother for LSTM/GRU?
    if self.type == 'lstm' or self.type == 'gru':
      assert self.feeding != 'none'

    if 'hmm' in self.type:
      # Emission distribution (input invariant)
      emit_distr = F.log_softmax(self.emit.weight, 0) # dim vocab_size x hidden_size (opposite of layer def order)

    # Embed
    emb = self.embed_input(words[:,:-1])

    for t in range(1, T):
      inp = emb[:,:,t-1]
      word_idx = words[:, t].unsqueeze(1)

      if 'hmm' in self.type:
        emit_ll, hidden_state = self.hmm_step(inp, hidden_state, emit_distr, word_idx)
      else:
        emit_ll, hidden_state = self.rnn_step(inp, hidden_state, word_idx)

      if t == 1:
        emit_marginal = emit_ll
      else:
        emit_marginal = emit_marginal + emit_ll

    return emit_marginal / (T-1), hidden_state

