import pickle, gzip
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
#torch.manual_seed(1)
import sys
torch.set_printoptions(threshold=1000, edgeitems=10)

def expand(text, max_len):
  while len(text) < max_len:
    text.append(NONE) 
  return text[:max_len]

tdata = pickle.load(gzip.open('data/train.freq.pkl.gz','rb'))
voc2i = pickle.load(gzip.open('data/v2i.freq.pkl.gz','rb'))
i2voc = pickle.load(gzip.open('data/i2v.freq.pkl.gz','rb'))

log = "HMM"
batch_size = 32
max_len = 20
NONE = voc2i["<PAD>"]
UNK = voc2i["#UNK#"]

data = []
for seq in tdata:
  data.append(expand(seq, max_len))
data = np.array(data)
print("Train  {:5}".format(len(data)))

def to_string(seq):
  return " ".join([i2voc[s] for s in seq]).replace("<PAD>","")

# Jan Buys Code
def log_sum_exp(vec, dim):
   # sum over dim
   max_score, _ = torch.max(vec, dim, keepdim=True)
   max_score_broadcast = max_score.expand(vec.size())
   return max_score.squeeze(dim) + torch.log(torch.sum(
       torch.exp(vec - max_score_broadcast), dim))

"""
  Model
"""
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hidden = 32
    self.embed_dim = 100
    self.vocab_dim = len(voc2i)
    self.num_clusters = 128

    self.Kr = torch.from_numpy(np.array(range(self.num_clusters))).cuda()
    
    # vocab x 100
    self.embeddings = nn.Embedding(len(voc2i), self.embed_dim).cuda()
    self.embeddings.weight.data.copy_(
        torch.from_numpy(np.load("inference/infer_glove.npy")))
    self.embeddings.requires_grad = False

    # Cluster vectors
    self.Cs = []
    for i in range(self.num_clusters):
      self.Cs.append(Variable(torch.zeros(batch_size).long().cuda() + i))

    # Embed clusters
    self.one_hot = False
    if self.one_hot:
      self.emb_cluster = nn.Embedding(self.num_clusters, self.num_clusters).cuda()
      self.emb_cluster.weight.data = torch.eye(self.num_clusters)
      self.emb_cluster.requires_grad = False
    else:
      self.emb_cluster = nn.Embedding(self.num_clusters, self.num_clusters).cuda()
  
    # Global HMM statistics
    self.start = nn.Linear(1, self.num_clusters)
    self.cluster_vocab = nn.Linear(self.num_clusters, len(voc2i), bias=False)     # f(cluster, word)
    self.cluster_vocab.weight.data.uniform_(-1, 1)     # Otherwise root(V) is huge
    self.cluster_trans = nn.Linear(1, self.num_clusters**2, bias=False) # f(cluster, cluster)
    self.cluster_trans.weight.data.uniform_(-1, 1)    # 0 to e (logspace)


  def forward(self, x):
    w = x.clone()
    #x = self.embeddings(x).permute(0, 2, 1)     

    N = batch_size
    T = w.size()[1]
    K = self.num_clusters

    dummy = Variable(torch.ones(N, 1)).cuda()

    tran = F.log_softmax(self.cluster_trans(dummy).view(N, K, K), dim=-1)

    pre_alpha = Variable(torch.zeros(N, K))
    cur_alpha = Variable(torch.zeros(N, K))
    pre_alpha = self.start(dummy).expand(N,K) #x_dist.clone()   # init w/ p(z)

    Emissions = torch.stack([
        F.log_softmax(self.cluster_vocab(self.emb_cluster(self.Cs[i])), -1)
        for i in range(K)])
    Emissions = Emissions.transpose(0, 1)    # Move batch to the front

    Kr = self.Kr
    for t in range(0, T):
      # Transition
      cur_alpha = pre_alpha.unsqueeze(-1).expand(N, K, K) + tran   # TODO: Sanity check
      cur_alpha = log_sum_exp(cur_alpha, 1)                        # TODO: Correct dim?

      # Emission
      word_idx = w[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
      cur_alpha[:, Kr] = cur_alpha[:, Kr] + \
                         Emissions[:, Kr].gather(2, word_idx).squeeze()

      # Update
      pre_alpha = cur_alpha.clone()

    return -1 * torch.mean(log_sum_exp(cur_alpha, dim=1))


  def step(self, src, train=True):
    src = Variable(torch.from_numpy(src).cuda())
    optimizer.zero_grad()
    net.train() if train else net.eval()

    LL = self(src)

    if train:
      loss = LL 
      loss.backward()
      optimizer.step()
    return LL

  def print_emissions(self):
    o = open("Emissions.{}.b{}.k{}.h{}.txt".format(log, batch_size, self.num_clusters, self.hidden),'w')
    for i in range(self.num_clusters):
      V = F.log_softmax(self.cluster_vocab(self.emb_cluster(self.Cs[i])), dim=-1)
      listed = [(torch.exp(V[0][j]).data.item(), str(i2voc[j])) for j in range(len(voc2i))]
      listed.sort()
      listed.reverse()
      o.write("\n%d\n" % i)
      for prob, word in listed[:50]:
        o.write("   {:10.8f}  {:10s}\n".format(100*prob, str(word)))
    o.close()

net = Net()
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)#, weight_decay=1e-3)

print("Starting Training")
"""
  Train
"""
for epoch in range(21):

  # Print # Training
  net.print_emissions()

  # Training
  inds = list(range(len(data)))
  random.shuffle(inds)
  iterate = tqdm.tqdm(range(0, len(inds) - len(inds)%batch_size, batch_size), ncols=80)
  for i in iterate:
    r = inds[i:i+batch_size]
    LL = net.step(data[r])
    iterate.set_description("Loss {:8.4f}".format(LL[0].data.item()))

