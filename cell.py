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


class ElmanCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin='sigmoid', 
               trans_only_nonlin=False):
    super(ElmanCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin
    self.trans_only_nonlin = trans_only_nonlin

    self.input_tr = nn.Linear(input_dim, hidden_dim, bias=trans_only_nonlin)

    self.transition = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def forward(self, inp, state):
    state = self.transition(state)

    if self.trans_only_nonlin:
      state = apply_nonlin(state, self.nonlin)
      state = state + self.input_tr(inp)
      # state = self.input_tr(inp) * state  # Yonatan's formulation
    else:
      state = state + self.input_tr(inp)
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
  def __init__(self, hidden_dim, logspace_hidden=False,
               feed_input=False, combine_input=False):
    super(HMMNewCell, self).__init__()
    self.hidden_dim = hidden_dim
    self.logspace_hidden = logspace_hidden
    self.feed_input = feed_input
    self.combine_input = combine_input
    assert not (feed_input and combine_input)
   
    if feed_input:
      self.transition = nn.Linear(self.hidden_dim, self.hidden_dim**2, bias=True)
    else:
      self.transition = nn.Linear(1, self.hidden_dim**2, bias=False)

    if combine_input:
      self.input_tr = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.init_weights()

  def init_weights(self, initrange=1.0):
    self.transition.weight.data.uniform_(-initrange, initrange)   # 0 to e (logspace)

  def forward(self, inp, state):
    # assume input is log_softmax(E)[prev_word]
    batch_size = state.size()[0]

    trans_inp = inp if self.feed_input else inp.new_ones((batch_size, 1))
    trans_distr = self.transition(trans_inp).view(batch_size, self.hidden_dim, 
                                            self.hidden_dim)

    inp_state = inp + state
    if self.logspace_hidden:
      inp_state = F.log_softmax(inp_state, dim=1)
    else:
      inp_state = F.softmax(inp_state, dim=1)

    if self.combine_input:
      # each input transtions element scales distrubution given previous hidden state
      trans_distr += self.input_tr(torch.exp(inp)).view(batch_size, self.hidden_dim, 1) #expand

    if self.logspace_hidden:
      trans_distr = F.log_softmax(trans_distr, dim=1)
      next_state = trans_distr + inp_state.view(batch_size, 1, self.hidden_dim)
            #.expand(batch_size, self.hidden_dim, self.hidden_dim)
      next_state = torch.logsumexp(next_state, 2)
    else:
      trans_distr = F.softmax(trans_distr, dim=1)
      next_state = trans_distr @ inp_state.unsqueeze(2) 
      next_state = torch.log(next_state)

    return next_state.view(batch_size, self.hidden_dim)

