import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_nonlin(state, nonlin):
  if nonlin == 'tanh':
    state = torch.tanh(state)
  elif nonlin == 'softmax':
    state = F.softmax(state, dim=-1)
  elif nonlin == 'sigmoid':
    state = torch.sigmoid(state)
  return state


class BigramCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin='sigmoid'):
    super(BigramCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin

    self.input_tr = nn.Linear(input_dim, hidden_dim, bias=True)

  def forward(self, inp, state):
    state = self.input_tr(inp)
    state = apply_nonlin(state, self.nonlin)
    return state


class ElmanCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin='sigmoid', 
               delayed_nonlin=False,
               single_trans=False,
               multiplicative=False):
    super(ElmanCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin
    self.delayed_nonlin = delayed_nonlin
    self.single_trans = single_trans
    self.multiplicative = multiplicative

    self.transition = nn.Linear(hidden_dim, hidden_dim, bias=True)
    if not single_trans:
      self.input_tr = nn.Linear(input_dim, hidden_dim, bias=(multiplicative))

  def forward(self, inp, state):
    if self.delayed_nonlin:
      state = apply_nonlin(state, self.nonlin)

    if self.multiplicative:
      if self.single_trans:
        state = self.transition(state * inp) # not working in logspace here 
      else:
        state = self.transition(state) * self.input_tr(inp)
    else:
      state = self.transition(state)
      if self.single_trans:
        state = state + inp
      else:
        state = state + self.input_tr(inp)

    if not self.delayed_nonlin:
      state = apply_nonlin(state, self.nonlin)

    return state


class RationalCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin=''):
    super(RationalCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin

    self.input_tr = nn.Linear(input_dim, hidden_dim, bias=True)

    self.forget_tr = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def forward(self, inp, state):
    inp_rep = self.input_tr(inp)
    inp_rep = apply_nonlin(inp_rep, self.nonlin)
    forget_gate = torch.sigmoid(self.forget_tr(inp))
    state = forget_gate * state + (1 - forget_gate) * inp_rep
    return state


class JordanCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin='sigmoid'):
    super(JordanCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin

    self.input_tr = nn.Linear(input_dim, hidden_dim, bias=False)

    self.transition = nn.Linear(input_dim, hidden_dim, bias=True)

  def forward(self, inp, state):
    # Note: Assume weighted embedding averaging has already been done
    state = self.transition(state) + self.input_tr(inp)
    state = apply_nonlin(state, self.nonlin)
    return state


class HMMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, logspace_hidden=False,
               feed_input=True, delay_trans_softmax=False,
               with_trans_gate=False):
    super(HMMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.logspace_hidden = logspace_hidden
    self.feed_input = feed_input
    self.delay_trans_softmax = delay_trans_softmax
    self.with_trans_gate = with_trans_gate
   
    assert not (with_trans_gate and (delay_trans_softmax or not logspace_hidden))

    if feed_input:
      self.transition = nn.Linear(self.input_dim, self.hidden_dim**2, bias=True)
    else:
      print("No feed input HMM cell.")
      self.transition = nn.Linear(1, self.hidden_dim**2, bias=False)

    if with_trans_gate:
      self.gate_tr = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.init_weights()

  def init_weights(self, initrange=1.0):
    self.transition.weight.data.uniform_(-initrange, initrange)   # 0 to e (logspace)

  def forward(self, inp, state):
    batch_size = state.size()[0]

    trans_inp = inp if self.feed_input else inp.new_ones((batch_size, 1))
    trans_distr = self.transition(trans_inp).view(batch_size, self.hidden_dim, 
                                            self.hidden_dim)

    if self.delay_trans_softmax:
      if self.logspace_hidden:
        state = trans_distr + state.view(batch_size, 1, self.hidden_dim).expand(batch_size, self.hidden_dim, self.hidden_dim)
        state = torch.logsumexp(state, 1) 
        state = F.log_softmax(state, dim=1)
      else:    
        state = trans_distr @ state.unsqueeze(2)
        state = F.softmax(state, dim=1)
    else:     
      if self.logspace_hidden:
        if self.with_trans_gate:
          gate = torch.sigmoid(self.gate_tr(inp)) # each gate element scales distrubution given previous hidden state
          trans_distr = trans_distr * gate.view(batch_size, self.hidden_dim, 1).expand(
                  batch_size, self.hidden_dim, self.hidden_dim) 
          # Note that at the moment we have to apply the gate before softmaxing
          trans_distr = F.log_softmax(trans_distr, dim=1)
        else:
          trans_distr = F.log_softmax(trans_distr, dim=1)


        state = trans_distr + state.view(batch_size, 1, self.hidden_dim).expand(batch_size, self.hidden_dim, self.hidden_dim)
        state = torch.logsumexp(state, 2)
      else:    
        trans_distr = F.softmax(trans_distr, dim=1)
        state = trans_distr @ state.unsqueeze(2) 

    return state.view(batch_size, self.hidden_dim)


class HMMNewCell(nn.Module):
  def __init__(self, hidden_dim, tensor_feed_input=False, add_feed_input=False,
               gate_feed_input=False, delay_trans_softmax=False,
               sigmoid_trans=False, probspace=False):
    super(HMMNewCell, self).__init__()
    self.hidden_dim = hidden_dim
    self.tensor_feed_input = tensor_feed_input
    self.add_feed_input = add_feed_input
    self.gate_feed_input = gate_feed_input
    self.delay_trans_softmax = delay_trans_softmax
    self.sigmoid_trans = sigmoid_trans
    self.probspace = probspace
   
    if tensor_feed_input:
      self.transition = nn.Linear(self.hidden_dim, self.hidden_dim**2, bias=True)
    else:
      self.transition = nn.Linear(1, self.hidden_dim**2, bias=True)

    if add_feed_input or gate_feed_input:
      self.input_tr = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.init_weights()

  def init_weights(self, initrange=1.0):
    self.transition.weight.data.uniform_(-initrange, initrange)   # 0 to e (logspace)

  def forward(self, inp, inp_sm, state):
    # assume inp is E[prev_word]
    # assume inp_sm is log_softmax(E)[prev_word]
    batch_size = state.size()[0]

    trans_inp = inp if self.tensor_feed_input else inp.new_ones((batch_size, 1))
    trans_distr = self.transition(trans_inp).view(batch_size, self.hidden_dim, 
                                            self.hidden_dim)

    if self.add_feed_input: 
      # each input transtions element scales distrubution given previous hidden state
      trans_distr += self.input_tr(inp).view(batch_size, self.hidden_dim, 1)
    elif self.gate_feed_input:
      # each gate element scales distribution independent of previous hidden state
      gate = torch.sigmoid(self.input_tr(inp)) 
      trans_distr = trans_distr * gate.view(batch_size, self.hidden_dim, 1).expand(
                  batch_size, self.hidden_dim, self.hidden_dim) 

    if self.probspace:
      inp_state = inp_sm + state
      inp_state = F.softmax(inp_state, dim=1)
    else:
      inp_state = inp_sm * state
      inp_state = F.normalize(inp_state, 1, 1)
    
    if self.sigmoid_trans:
      next_state = trans_distr @ inp_state.unsqueeze(2)
      next_state = torch.sigmoid(next_state)
    if self.delay_trans_softmax:
      next_state = trans_distr @ inp_state.unsqueeze(2)
      next_state = F.softmax(next_state, dim=1)
    else:
      trans_distr = F.softmax(trans_distr, dim=1)
      next_state = trans_distr @ inp_state.unsqueeze(2) 

    if not self.probspace:
      next_state = torch.log(next_state)

    return next_state.view(batch_size, self.hidden_dim)

