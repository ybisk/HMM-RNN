import torch
import torch.nn as nn
import torch.nn.functional as F

class ElmanCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, nonlin='tanh', feed_input=True,
        trans_use_input_dim=False, trans_only_nonlin=False):
    super(ElmanCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.nonlin = nonlin
    self.feed_input = feed_input
    self.trans_use_input_dim = trans_use_input_dim
    self.trans_only_nonlin = trans_only_nonlin

    self.input_tr = nn.Linear(self.input_dim, self.hidden_dim,
                              bias=trans_only_nonlin)

    tr_input_dim = input_dim if trans_use_input_dim else hidden_dim
    self.transition = nn.Linear(tr_input_dim, self.hidden_dim, bias=True)

  def apply_nonlin(self, state):
    # non-linearity (default is none)
    if self.nonlin == 'tanh':
      new_state = torch.tanh(state)
    elif self.nonlin == 'softmax':
      new_state = F.softmax(state, dim=-1)
    elif self.nonlin == 'sigmoid':
      new_state = torch.sigmoid(state)
    else:
      new_state = state
    return new_state

  def forward(self, inp, state):
    new_state = self.transition(state)

    if self.trans_only_nonlin:
      new_state = self.apply_nonlin(new_state)
      if self.feed_input: #TODO in Yonatan's formulation this is element-wise multiply
        new_state += self.input_tr(inp) #TODO use bias here?
        # new_state = self.input_tr(inp) * new_state
    else:
      if self.feed_input:
        new_state += self.input_tr(inp)
      new_state = self.apply_nonlin(new_state)

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
    new_state = torch.logsumexp(new_state, 1) # dim batch_size x hidden_dim

    return new_state

