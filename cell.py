import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_utils as U

class ElmanCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin='tanh', feed_input=True):
    super(ElmanCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.feed_input = feed_input
    self.nonlin = nonlin

    self.input_tr = nn.Linear(self.input_dim, self.hidden_dim) # TODO bias?
    self.transition = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

  def forward(self, inp, state):
    new_state = self.transition(state)
    if self.feed_input:
      new_state += self.input_tr(inp)
    if self.nonlin == 'tanh':
      new_state = F.tanh(new_state)
    elif self.nonlin == 'softmax':
      new_state = F.softmax(new_state, dim=-1)
    elif self.nonlin == 'sigmoid':
      new_state = F.sigmoid(new_state)
    # default is no non-linearity

    return new_state

class HMMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, feed_input=True):
    super(HMMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.feed_input = feed_input

    if feed_input:
      self.transition = nn.Linear(self.input_dim, self.hidden_dim**2, bias=True)
    else:
      self.transition = nn.Linear(1, self.hidden_dim**2, bias=False)
    self.transition.weight.data.uniform_(-1, 1)           # 0 to e (logspace)

  def forward(self, inp, state):
    batch_size = state.size()[0]

    # assume that inp is a zero if not feed_input (TODO assert this)
    #TODO implement delayed softmax (hmm+1)
    transition_distr = self.transition(inp).view(batch_size, self.hidden_dim, self.hidden_dim)
    transition_distr = F.log_softmax(transition_distr, dim=-1) # double check dimension

    new_state = transition_distr + state.unsqueeze(-1).expand(batch_size, self.hidden_dim, self.hidden_dim)
    new_state = U.log_sum_exp(new_state, 1) # dim batch_size x hidden_dim

    return new_state

