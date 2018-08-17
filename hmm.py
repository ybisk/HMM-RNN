import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cell

class HMM(nn.Module):
  def __init__(self, vocab_size, args):
    super(HMM, self).__init__()
    self.embed_dim = args.embed_dim
    self.vocab_size = vocab_size
    self.hidden_dim = args.hidden_dim
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
    self.trans = cell.HMMCell(self.embed_dim, self.hidden_dim, 
                              logspace_hidden = self.logspace_hidden,
                              feed_input = (self.feeding != 'none'), 
                              delay_trans_softmax = (self.type == 'hmm-1'))

    # Emission parameters
    self.emit = nn.Linear(self.hidden_dim, vocab_size, bias=True)

    self.init_weights()

  def init_weights(self, initrange=1.0):
    self.emit.weight.data.uniform_(-initrange, initrange)    # Otherwise root(V) is huge
    self.emit.bias.data.zero_()
    if not self.glove_emb:
      self.embed.weight.data.uniform_(-initrange, initrange) 

  def init_hidden_state(self, batch_size):
    weight = next(self.parameters())
    # Uniform distribution over clusters
    return weight.new_full((batch_size, self.hidden_dim), 1/self.hidden_dim)

  def emissions_list(self):
    emit_distr = F.log_softmax(self.emit.weight, 0) # dim vocab_size x hidden_size
    return emit_distr.detach().cpu().numpy()

  def embed_input(self, x):
    embed_x = self.embed(x)
    if 'encode' in self.feeding:
      embed_x, _ = self.encode_context(embed_x)
    return embed_x.permute(0, 2, 1) # batch_size x embed_dim x seq_length 

  def forward(self, words, hidden_state):
    N = words.size()[0] # batch size
    T = words.size()[1] # sequence length
    emit_marginal = None

    # Embed
    emb = self.embed_input(words)

    # Initial hidden state distribution is computed in first time step

    # Emission distribution (input invariant)
    emit_distr = F.log_softmax(self.emit.weight, 0) # dim vocab_size x hidden_size (opposite of layer def order)

    for t in range(1, T):
      # Transition
      state_input = emb[:,:,t-1] 
      hidden_output = self.trans(state_input, hidden_state)

      # Emit 
      word_idx = words[:, t].unsqueeze(1).expand(N, self.hidden_dim)
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

      # Accumulate
      if t == 1:
        emit_marginal = emit_ll
      else:
        emit_marginal = emit_marginal + emit_ll

    return emit_marginal, hidden_state


