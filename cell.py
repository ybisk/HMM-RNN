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
  def __init__(self, input_dim, hidden_dim, nonlin='tanh', feed_input=True,
        trans_only_nonlin=False):
    super(ElmanCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin
    self.feed_input = feed_input
    self.trans_only_nonlin = trans_only_nonlin

    self.input_tr = nn.Linear(input_dim, hidden_dim, bias=trans_only_nonlin)

    self.transition = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def forward(self, inp, state):
    state = self.transition(state)

    if self.trans_only_nonlin:
      state = apply_nonlin(state, self.nonlin)
      if self.feed_input: 
        state = state + self.input_tr(inp) 
        # state = self.input_tr(inp) * state  # Yonatan's formulation
    else:
      if self.feed_input:
        state = state + self.input_tr(inp)
      state = apply_nonlin(state, self.nonlin)

    return state


class JordanCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin='tanh', feed_input=True):
    super(JordanCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin
    self.feed_input = feed_input

    self.input_tr = nn.Linear(input_dim, hidden_dim, bias=False)

    self.transition = nn.Linear(input_dim, hidden_dim, bias=True)

  def forward(self, inp, state):
    # Note: Assume weighted embedding averaging has already been done
    state = self.transition(state)
    if self.feed_input:
      state = state + self.input_tr(inp)
    state = apply_nonlin(state, self.nonlin)
    return state


class HMMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, feed_input=True,
               delay_trans_softmax=False):
    super(HMMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.feed_input = feed_input
    self.delay_trans_softmax = delay_trans_softmax

    if feed_input:
      self.transition = nn.Linear(self.input_dim, self.hidden_dim**2, bias=True)
    else:
      self.transition = nn.Linear(1, self.hidden_dim**2, bias=False)

    self.init_weights()

  def init_weights(self, initrange=1.0):
    self.transition.weight.data.uniform_(-initrange, initrange)   # 0 to e (logspace)

  def forward(self, inp, state):
    batch_size = state.size()[0]

    trans_inp = inp if self.feed_input else inp.new_ones((batch_size, 1))
    trans_distr = self.transition(trans_inp).view(batch_size, self.hidden_dim, 
                                            self.hidden_dim)
    # Implementing this in prob space

    if self.delay_trans_softmax:
      state = trans_distr @ state.unsqueeze(2)
      state = F.softmax(state, dim=1)
    else:    
      trans_distr = F.softmax(trans_distr, dim=1)
      state = trans_distr @ state.unsqueeze(2) 

    #trans_distr = F.log_softmax(transition_distr, dim=2)
    #state = trans_distr + state.unsqueeze(-1).expand(batch_size, self.hidden_dim, self.hidden_dim).clone()
    #state = torch.logsumexp(state, 1) # dim batch_size x hidden_dim

    return state.view(batch_size, self.hidden_dim)

