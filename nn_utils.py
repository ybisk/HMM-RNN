import torch

# Jan Buys Code
def log_sum_exp(vec, dim):
   # sum over dim
   max_score, _ = torch.max(vec, dim, keepdim=True)
   max_score_broadcast = max_score.expand(vec.size())
   return max_score.squeeze(dim) + torch.log(torch.sum(
       torch.exp(vec - max_score_broadcast), dim))

