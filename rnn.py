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
    self.delay_softmax = args.delay_softmax
    self.feeding = args.feeding 
    self.type = args.type
    self.glove_emb = args.glove_emb
    self.logspace_hidden = True

    self.drop = nn.Dropout(args.dropout)
    # Input embedding parameters
    self.embed = nn.Embedding(vocab_size, self.embed_dim)
    if args.glove_emb:
      self.embed.weight.data.copy_(
          torch.from_numpy(np.load("inference/infer_glove.npy")))
      self.embed.requires_grad = False

    # feeding is none doesn't make sense for RNNs
    if (self.type == 'lstm' or self.type == 'gru' or self.type == 'jordan' or
        self.type.startswith('rnn') or self.type.startswith('rrnn')):
      assert self.feeding != 'none'

    if args.feeding == 'encode-lstm':
      self.encode_context = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)
    elif args.feeding.startswith('encode'):
      assert False, 'Feeding encoding not implemented'

    # Transition cell
    if args.type.startswith('hmm-new'):
      assert self.hidden_dim == self.embed_dim and args.tie_embeddings
      assert (self.feeding == 'none'
              or args.type.startswith('hmm-new-tensor-feed')
              or args.type.startswith('hmm-new-add-feed')
              or args.type.startswith('hmm-new-gate-feed'))
      self.trans = cell.HMMNewCell(self.hidden_dim, 
              args.type.startswith('hmm-new-tensor-feed'),
              args.type.startswith('hmm-new-add-feed'),
              args.type.startswith('hmm-new-gate-feed'),
              args.delay_softmax.startswith('trans'),
              '-sigmoid' in args.type,
              '-prob' in args.type)
    elif args.type.startswith('hmm') or args.type.startswith('rnn-hmm'):
      self.trans = cell.HMMCell(self.embed_dim, self.hidden_dim, 
                                logspace_hidden = self.logspace_hidden,
                                feed_input = (self.feeding != 'none'), 
                                delay_trans_softmax = (self.type == 'hmm+1'),
                                with_trans_gate = (self.type == 'hmm-g'))
    elif args.type.startswith('elman'): 
      if args.type.startswith('elman-linear'):
        nonlin = ''
      elif args.type.startswith('elman-softmax'):
        nonlin = 'softmax'
      else:
        nonlin = 'sigmoid'
      self.trans = cell.ElmanCell(self.embed_dim, self.hidden_dim, nonlin,
          delayed_nonlin = ('-delayed' in args.type), 
          single_trans = ('-single' in args.type), 
          multiplicative = ('-mult' in args.type))
    elif args.type.startswith('rnn'):
      nonlin = 'softmax' if args.type == 'rnn-1' or args.type == 'rnn-2' else 'sigmoid'
      self.trans = cell.ElmanCell(self.embed_dim, self.hidden_dim, nonlin,
          delayed_nonlin = (args.type == 'rnn-2'), multiplicative =
          (args.type == 'rnn-3'))
    elif args.type.startswith('rrnn'):
      nonlin = '' if args.type == 'rrnn-2' or args.type == 'rrnn-r' else 'sigmoid'
      self.trans = cell.RationalCell(self.embed_dim, self.hidden_dim, nonlin)
      if args.type == 'rrnn-r':
        self.reset_tr = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        assert self.hidden_dim == self.embed_dim
    elif args.type == 'bigram':
      self.trans = cell.BigramCell(self.embed_dim, self.hidden_dim, 'sigmoid')
    elif args.type == 'jordan':
      nonlin = 'tanh'
      self.trans = cell.JordanCell(self.embed_dim, self.hidden_dim, nonlin)
    elif args.type == 'gru':
      self.trans = nn.GRUCell(self.embed_dim, self.hidden_dim)
    elif args.type == 'lstm':
      #self.trans = nn.LSTM(self.embed_dim, self.hidden_dim)
      self.trans = nn.LSTMCell(self.embed_dim, self.hidden_dim)
    else: 
      print(args.type + " is not implemented")
      sys.exit()

    # Emission parameters
    self.emit = nn.Linear(self.hidden_dim, vocab_size) 

    if args.tie_embeddings:
      assert self.hidden_dim == self.embed_dim, 'When using tied embeddings, hidden size and embeddings size must be equal.'
      self.emit.weight = self.embed.weight

    self.init_weights(args.initrange)

  def init_weights(self, initrange=1.0):
    if not self.glove_emb:
      self.embed.weight.data.uniform_(-initrange, initrange) 
    self.emit.bias.data.zero_()
    self.emit.weight.data.uniform_(-initrange, initrange) # Otherwise root(V) is huge

  def init_hidden_state(self, batch_size, inp=None):
    weight = next(self.parameters())
    if self.type.startswith('hmm'):
      # Uniform distribution over clusters
      if self.logspace_hidden:
        return weight.new_full((batch_size, self.hidden_dim), -np.log(self.hidden_dim))
      else:
        return weight.new_full((batch_size, self.hidden_dim), 1/self.hidden_dim)
    elif self.type == 'jordan':
      assert inp is not None
      state = weight.new_zeros((batch_size, self.vocab_size))
      state.scatter_(1, inp.view(batch_size, 1), 1)
      return state
    elif self.type == 'lstm':
      #return (weight.new_zeros((1, batch_size, self.hidden_dim)),
      #        weight.new_zeros((1, batch_size, self.hidden_dim)))
      return (weight.new_zeros((batch_size, self.hidden_dim)),
              weight.new_zeros((batch_size, self.hidden_dim)))
    else:
      return weight.new_zeros((batch_size, self.hidden_dim))

  def emissions_list(self):
    emit_distr = F.log_softmax(self.emit.weight, 0) # dim vocab_size x hidden_size
    return emit_distr.detach().cpu().numpy()

  def embed_input(self, words):
    emb = self.embed(words)
    if self.feeding.startswith('encode'):
      emb, _ = self.encode_context(emb)
    return emb.permute(0, 2, 1) # batch_size x embed_dim x seq_length 

  def rnn_step(self, state_input, hidden_state, word_idx, emit_distr_fixed=None, 
               compute_emit=False):
    N = word_idx.size()[0] # batch size
    # Transition
    if self.type == 'lstm':
      #hidden_output, hidden_cell_state = self.trans(state_input.unsqueeze(0), 
      #                                           hidden_state)
      #hidden_output = hidden_output.squeeze(0)
      hidden_output, hidden_memstate = self.trans(state_input, hidden_state)
    elif self.type == 'jordan':
      #TODO move to cell if we can tie weights
      hidden_input_state = hidden_state @ self.embed.weight
      hidden_output = self.trans(state_input, hidden_input_state)
    else:  
      hidden_output = self.trans(state_input, hidden_state)

    # Emit
    if self.type == 'rrnn':
      output = torch.sigmoid(hidden_output.clone())
      #output = torch.tanh(hidden_output.clone())
    elif self.type == 'rrnn-r':
      # apply reset gate
      reset_gate = torch.sigmoid(self.reset_tr(state_input))
      output = reset_gate * hidden_output + (1 - reset_gate) * state_input
    else:
      output = hidden_output.clone()

    if self.type.startswith('elman') and self.type.endswith('-hmm-emit'):
      # First normalize sigmoid hidden output
      #output = torch.log(F.normalize(output, 1, 1))
      # Now assuming softmax in cell
      if self.type.startswith('elman-softmax'):
        output = torch.log(output)
      else:
        output = torch.log(F.normalize(output, 1, 1))
      current_embed = emit_distr_fixed.gather(0, word_idx.expand(N, self.hidden_dim))
      joint_state_ll = output + current_embed
      emit_ll = torch.logsumexp(joint_state_ll, 1)

      if compute_emit:
        emit_distr = torch.exp(emit_distr_fixed)
        use_marginal = False
        if use_marginal:
          marginal = emit_distr @ hidden_output.transpose(0,1)
        else:
          marginal = emit_distr[:,torch.argmax(hidden_output, dim=1)]
    else:    
      if self.type.startswith('elman') and '-delayed' in self.type:
        output = F.softmax(output, 1) 
      elif self.type.startswith('elman') and '-norm' in self.type:
        output = F.normalize(output, 1, 1) 
      else:
        output = self.drop(output)

      logits = self.emit(output)
      emit_distr = F.log_softmax(logits, -1)        # Emission
      emit_ll = emit_distr.gather(1, word_idx).squeeze()           # Word Prob
      if compute_emit:
        marginal = torch.exp(emit_distr)

    # State Update
    if self.type == 'lstm':
      #hidden_state = hidden_cell_state
      hidden_state = (hidden_output, hidden_memstate)
    elif self.type == 'jordan':
      hidden_state = F.softmax(logits, -1) # not log_softmax
    else:
      hidden_state = hidden_output 

    if compute_emit:
      return emit_ll, hidden_state, marginal.squeeze()
    else:
      return emit_ll, hidden_state, None

  def hmm_step(self, state_input, hidden_state, emit_distr, word_idx, compute_emit=False):
    N = word_idx.size()[0] # batch size

    # Transition
    hidden_output = self.trans(state_input, hidden_state)

    if not self.logspace_hidden: # need log prob for any subsequent computation
      hidden_output = torch.log(hidden_output)

    # Emit 
    if self.type == 'hmm+2': # delay emit softmax
      #TODO this consumes a lot of memory, and is slow.
      emit_distr = emit_distr.unsqueeze(0).expand(N, self.vocab_size, self.hidden_dim)
      hidden_output = hidden_output.view(N, 1, self.hidden_dim).expand(
              N, self.vocab_size, self.hidden_dim)
      emit_weight = emit_distr + hidden_output
      emit_distr = torch.log_softmax(emit_weight.view(N, -1), dim=1).view(
              N, self.vocab_size, self.hidden_dim)

      word_idx = word_idx.unsqueeze(1).expand(N, 1, self.hidden_dim)
      emit_state_ll = emit_distr.gather(1, word_idx).view(N, self.hidden_dim)
    else:
      word_idx = word_idx.expand(N, self.hidden_dim)
      emit_state_ll = emit_distr.gather(0, word_idx) # batch_size x hidden_dim

    # State Update
    # TODO standard output dropout won't work here; maybe try fixed mask (like variational dropout)

    if self.type == 'hmm+2':
      joint_state_ll = emit_state_ll
    else:
      joint_state_ll = hidden_output + emit_state_ll

    emit_ll = torch.logsumexp(joint_state_ll, 1)
    hidden_state = joint_state_ll - emit_ll.unsqueeze(1).expand(N, self.hidden_dim)

    if not self.logspace_hidden:
      hidden_state = torch.exp(hidden_state)

    if compute_emit:
      # Hidden state is now a distribution
      # emit distr is now a distribution
      emit_distr = torch.exp(emit_distr)

      use_marginal = False
      if use_marginal:
        marginal = emit_distr @ hidden_state.transpose(0,1)
      else:
        marginal = emit_distr[:,torch.argmax(hidden_state, dim=1)]

      return emit_ll, hidden_state, marginal.squeeze()
    else:
      return emit_ll, hidden_state, None

  def new_hmm_step(self, prev_embed_unnorm, prev_embed, current_embed,
          hidden_state, emit_distr_fixed, word_idx, compute_emit=False):
    N = hidden_state.size()[0] # batch size

    # Transition
    hidden_output = self.trans(prev_embed_unnorm, prev_embed, hidden_state)
    output = hidden_output.clone()

    # Emit 
    if self.delay_softmax.endswith('emit'): # delay_softmax emit = rnn-emit
      if not '-prob' in self.type:
        output = torch.exp(output)
      output = self.drop(output)
      logits = self.emit(output) 
      emit_distr = F.log_softmax(logits, -1)
      emit_ll = emit_distr.gather(1, word_idx).squeeze()
      if compute_emit:
        marginal = torch.exp(emit_distr)
    else:
      if 'sigmoid' in self.type:
        if '-prob' in self.type:
          output = F.normalize(output, 1, 1)
          output = torch.log(output)
        else:
          output = F.log_softmax(output, 1)
      else:
        if '-prob' in self.type:
          output = torch.log(output)

      joint_state_ll = output + current_embed
      emit_ll = torch.logsumexp(joint_state_ll, 1)

      if compute_emit:
        if '-prob' not in self.type:
          emit_distr = torch.exp(emit_distr_fixed)
        else:
          emit_distr = emit_distr_fixed
        output_prob = torch.exp(output)

        use_marginal = False
        if use_marginal:
          marginal = emit_distr @ output_prob.transpose(0,1)
        else:
          marginal = emit_distr[:,torch.argmax(output, dim=1)]

    if compute_emit:
      return emit_ll, hidden_output, marginal.squeeze()
    else:
      return emit_ll, hidden_output, None


  def forward(self, words, hidden_state, compute_emit=False):
    N = words.size()[0] # batch size
    T = words.size()[1] # sequence length
    num_tags = len(self.corpus.dict.i2tag)
    emit_marginal = None
    emissions_over_tags = np.zeros((N,T-1,num_tags))

    if (self.type.startswith('hmm') or (self.type.startswith('elman') 
        and self.type.endswith('-hmm-emit'))):
      # Emission distribution (input invariant)
      # dim vocab_size x hidden_size (opposite of layer def order)
      emit_weight = self.emit.weight + self.emit.bias.view(self.vocab_size, 1).expand(self.vocab_size, self.hidden_dim) 
      if self.type.startswith('hmm-new') and '-prob' in self.type:
        emit_weight = F.softmax(emit_weight, 0) 
      elif not self.type == 'hmm+2': # delay emit softmax
        emit_weight = F.log_softmax(emit_weight, 0) 

    # Embed
    if self.type.startswith('hmm-new'):
      word_idx = words[:, 0].unsqueeze(1).expand(N, self.hidden_dim)
      current_embed = emit_weight.gather(0, word_idx) # batch_size x hidden_dim
        
    emb = self.embed_input(words[:,:-1])
    emb = self.drop(emb)

    for t in range(1, T):
      word_idx = words[:, t].unsqueeze(1)
      
      inp = emb[:,:,t-1] # unnormalized prev embed
      if self.type.startswith('hmm-new'): 
        prev_embed = current_embed
        current_embed = emit_weight.gather(0, word_idx.expand(N, self.hidden_dim))
        emit_ll, hidden_state, emit_distr = self.new_hmm_step(inp, prev_embed,
                current_embed, hidden_state, emit_weight, word_idx, compute_emit=compute_emit)
      elif self.type.startswith('hmm'):
        emit_ll, hidden_state, emit_distr = self.hmm_step(inp, hidden_state,
                emit_weight, word_idx, compute_emit=compute_emit)
      else:
        emit_ll, hidden_state, emit_distr = self.rnn_step(inp, hidden_state, 
                word_idx, emit_weight if (self.type.startswith('elman') 
                  and self.type.endswith('-hmm-emit')) else None, compute_emit=compute_emit)

      if t == 1:
        emit_marginal = emit_ll
      else:
        emit_marginal = emit_marginal + emit_ll

      if compute_emit:
        emissions_over_tags[:, t-1, :] = emit_distr.cpu().numpy() @ self.corpus.dict.tag_embed

    if compute_emit:
      return emit_marginal / (T - 1), hidden_state, emissions_over_tags
    else:
      return emit_marginal / (T - 1), hidden_state

