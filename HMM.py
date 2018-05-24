import sys, pickle, gzip, random, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#torch.manual_seed(1)
import argparse
torch.set_printoptions(threshold=1000, edgeitems=10)
from torchviz import make_dot, make_dot_from_trace

from tensorboardX import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='HMM')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--max-len', type=int, default=20, help='max seq len')
parser.add_argument('--hidden-dim', type=int, default=64, help='hidden dim')
parser.add_argument('--clusters', type=int, default=64, help='num clusters')
parser.add_argument('--condition', type=str, default='none', help='Condition on none|word|lstm')
parser.add_argument('--one-hot', action='store_true', default=False, help='1-hot clusters')
parser.add_argument('--log', type=str, default='', help='Log name')
parser.add_argument('--type', type=str, default='hmm', help='hmm|jordan|elman')
args = parser.parse_args()


fname = ".c{}.h{}.l{}".format(args.clusters, args.hidden_dim, args.max_len)
if args.one_hot:
  fname += ".1hot"
fname += ".{}".format(args.condition)

writer = SummaryWriter("./log/{}".format(args.type) + args.log + fname)

def expand(text):
  text.insert(START, 0)
  while len(text) < args.max_len:
    text.append(NONE)
  return text[:args.max_len]

tdata = pickle.load(gzip.open('data/train.freq.pkl.gz','rb'))
voc2i = pickle.load(gzip.open('data/v2i.freq.pkl.gz','rb'))
i2voc = pickle.load(gzip.open('data/i2v.freq.pkl.gz','rb'))

NONE = voc2i["<PAD>"]
UNK = voc2i["#UNK#"]
START = voc2i["<S>"]

data = []
for seq in tdata:
  data.append(expand(seq))
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
    self.hidden = args.hidden_dim
    self.embed_dim = 100
    self.vocab_dim = len(voc2i)
    self.num_clusters = args.clusters

    self.Kr = torch.from_numpy(np.array(range(self.num_clusters)))
    self.dummy = torch.ones(args.batch_size, 1).to(device)

    # vocab x 100
    self.embeddings = nn.Embedding(len(voc2i), self.embed_dim)
    self.embeddings.weight.data.copy_(
        torch.from_numpy(np.load("inference/infer_glove.npy")))
    self.embeddings.requires_grad = False

    if args.condition == 'lstm':
      self.enc = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True)

    if args.type != 'hmm': 

      if args.type == 'jordan':
        self.b_h = nn.Linear(1, self.embed_dim)
        self.start_1hot = torch.zeros(1, len(voc2i)).to(device)
        self.start_1hot[0, START] = 1
        self.start_1hot.requires_grad = False

      if args.type == 'elman':
        self.trans = nn.Linear(self.embed_dim, self.embed_dim) 

      self.vocab = nn.Linear(self.embed_dim, len(voc2i), bias=True)    # f(cluster, word)
      self.vocab.weight.data.uniform_(-1, 1)                           # Otherwise root(V) is huge

    else:
      # Cluster vectors
      self.Cs = []
      for i in range(self.num_clusters):
        self.Cs.append(torch.zeros(args.batch_size).long().to(device) + i)

      # Embed clusters
      self.one_hot = False
      if self.one_hot:
        self.emb_cluster = nn.embedding(self.num_clusters, self.num_clusters)
        self.emb_cluster.weight.data = torch.eye(self.num_clusters)
        self.emb_cluster.requires_grad = False
      else:
        self.emb_cluster = nn.Embedding(self.num_clusters, self.num_clusters)

      # HMM Start Probabilities
      self.start = nn.Linear(1, self.num_clusters)

      self.vocab = nn.Linear(self.num_clusters, len(voc2i), bias=False)   # f(cluster, word)
      self.vocab.weight.data.uniform_(-1, 1)                              # Otherwise root(V) is huge
      if args.condition == 'none':
        self.trans = nn.Linear(1, self.num_clusters**2, bias=False)         # f(cluster, cluster)
      else:
        self.trans = nn.Linear(self.embed_dim, self.num_clusters**2, bias=False)         # f(cluster, cluster)
      self.trans.weight.data.uniform_(-1, 1)                              # 0 to e (logspace)


  def forward(self, x):
    w = x.clone()
    x = self.embeddings(x)
    if args.condition == 'lstm':
      x, _ = self.enc(x)
    x = x.permute(0, 2, 1)
    N = args.batch_size
    T = w.size()[1]

    if args.type != 'hmm':
      cur_alpha = torch.zeros(N).to(device)
      h_tm1 = torch.zeros(N, self.embed_dim).to(device)

      if args.type == 'jordan':
        b_h = self.b_h(self.dummy)
        y_tm1 = self.start_1hot.expand(N, len(voc2i))   # One hot start

      for t in range(1, T):
        if args.type == 'elman':
          # h_t = act(W_h x_t + U_h h_t-1 + b_h)
          h_t = F.tanh(x[:,:,t-1] + self.trans(h_tm1))

        if args.type == 'jordan':
          # h_t = act(W_h x_t + U_h y_t-1 + b_h)
          h_t = F.tanh(x[:,:,t-1] +  y_tm1 @ self.embeddings.weight + b_h)

        # y_t = act(W_y h_t + b_y)
        y_t = F.log_softmax(self.vocab(h_t), -1)        # Emission

        word_idx = w[:, t].unsqueeze(1)
        cur_alpha += y_t.gather(1, word_idx).squeeze()           # Word Prob

        y_tm1 = y_t.clone()
        h_tm1 = h_t.clone()
      return -1 * torch.mean(cur_alpha)

    if args.type == 'hmm':
      K = self.num_clusters
      
      if args.condition == 'none':
        tran = F.log_softmax(self.trans(self.dummy).view(N, K, K), dim=-1)
      
      pre_alpha = torch.zeros(N, K)
      cur_alpha = torch.zeros(N, K)
      pre_alpha = F.log_softmax(self.start(self.dummy).expand(N,K), dim=-1)
      
      Emissions = torch.stack([
          F.log_softmax(self.vocab(self.emb_cluster(self.Cs[i])), -1)
          for i in range(K)])
      Emissions = Emissions.transpose(0, 1)    # Move batch to the front
      
      for t in range(1, T):
        if args.condition != 'none':
          tran = F.log_softmax(self.trans(x[:,:,t-1]).view(N, K, K), dim=-1)
        
        # Transition
        cur_alpha = pre_alpha.unsqueeze(-1).expand(N, K, K) + tran
        cur_alpha = log_sum_exp(cur_alpha, 1)
        
        # Emission
        word_idx = w[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
        cur_alpha[:, self.Kr] = cur_alpha[:, self.Kr] + \
                           Emissions[:, self.Kr].gather(2, word_idx).squeeze()
        
        # Update
        pre_alpha = cur_alpha.clone()
      
      # TODO – Perplexity 2^-Sum(p * log2(p))
      return -1 * torch.mean(log_sum_exp(cur_alpha, dim=1))


  def step(self, src, train=True):
    src = torch.from_numpy(src).to(device)
    optimizer.zero_grad()
    net.train() if train else net.eval()

    LL = self(src)

    if train:
      loss = LL
      loss.backward()
      optimizer.step()
    return LL

  def print_emissions(self):
    o = open("Emissions.{}.txt".format(fname),'w')
    for i in range(self.num_clusters):
      V = F.log_softmax(self.vocab(self.emb_cluster(self.Cs[i])), dim=-1)
      listed = [(torch.exp(V[0][j]).data.item(), str(i2voc[j])) for j in range(len(voc2i))]
      listed.sort()
      listed.reverse()
      o.write("\n%d\n" % i)
      for prob, word in listed[:50]:
        o.write("   {:10.8f}  {:10s}\n".format(100*prob, str(word)))
    o.close()

net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)#, weight_decay=1e-3)

print("Starting Training")
"""
  Train
"""
#writer.add_graph(model=Net(), input_to_model=torch.ones((2,2)), verbose=True)

step = 0
for epoch in range(args.epochs):

  # Print # Training
  if args.type == 'hmm':
    net.print_emissions()

  # Training
  inds = list(range(len(data)))
  random.shuffle(inds)
  iterate = tqdm.tqdm(range(0, len(inds) - len(inds)%args.batch_size, args.batch_size), ncols=80)
  for i in iterate:
    r = inds[i:i + args.batch_size]
    LL = net.step(data[r])
    iterate.set_description("Loss {:8.4f}".format(LL.item()))

    step += 1
    writer.add_scalar('Loss', LL.item(), step)

    #out = open("graph.dot",'w')
    #out.write(str(make_dot(LL, params=dict(net.named_parameters()))))
    #out.close()
    #print("generated")
    #sys.exit()
writer.close()
