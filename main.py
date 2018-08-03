import sys, pickle, gzip, random, tqdm
import numpy as np
import torch
import argparse

from tensorboardX import SummaryWriter

import model

# TODO: Embed dim is fixed according to GloVe, these should be decoupled
parser = argparse.ArgumentParser(description='HMM-RNN')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--max-len', type=int, default=20, help='max seq len')
parser.add_argument('--hidden-dim', type=int, default=64, help='hidden dim')
parser.add_argument('--clusters', type=int, default=64, help='num clusters')
parser.add_argument('--condition', type=str, default='none',
                    help='none|word|lstm')
parser.add_argument('--one-hot', action='store_true', default=False,
                    help='1-hot clusters')
parser.add_argument('--glove-emb', action='store_true', default=False,
                    help='Use GloVe embeddings instead of learning embeddings.')
parser.add_argument('--log', type=str, default='./log/', help='Log dir')
parser.add_argument('--type', type=str, default='hmm',
                    help='hmm|hmm+1|jordan|elman|dist|gru|lstm')
parser.add_argument('--note', type=str, default='',
                    help='extra note on fname')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(threshold=1000, edgeitems=10)
#torch.manual_seed(1)

fname = "{}".format(args.type)
fname += ".{}".format(args.note) if len(args.note) > 0 else ""
if args.one_hot:
  fname += "_1hot"
fname += "_{}".format(args.condition)
fname += "_l{}".format(args.max_len)
if 'hmm' in args.type:
  fname += "_c{}_h{}".format(args.clusters, args.hidden_dim)
if args.glove_emb:
  fname += "_fixedE"

writer = SummaryWriter(args.log + fname)

def expand(text):
  text.insert(START, 0)
  while len(text) < args.max_len:
    text.append(NONE)
  return text[:args.max_len]

#TODO (Jan): Do we have validation data?
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

vocab_size = len(voc2i)
net = model.Net(vocab_size, args, device).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)#, weight_decay=1e-3)

print("Starting Training")
"""
  Train
"""
#writer.add_graph(model=model.Net(), input_to_model=torch.ones((2,2)), verbose=True)

step = 0
for epoch in range(args.epochs):

  # Print # Training
  if 'hmm' in args.type:
    net.print_emissions(fname, i2voc)

  # Training
  inds = list(range(len(data)))
  random.shuffle(inds)
  iterate = tqdm.tqdm(range(0, len(inds) - len(inds)%args.batch_size, args.batch_size), ncols=80)
  for i in iterate:
    r = inds[i:i + args.batch_size]
    src = torch.from_numpy(data[r]).to(device)

    optimizer.zero_grad()
    net.train()
    LL = net(src)

    loss = LL
    loss.backward()
    optimizer.step()

    iterate.set_description("Loss {:8.4f}".format(LL.item()))

    step += 1
    writer.add_scalar('Loss', LL.item(), step)

    #out = open("graph.dot",'w')
    #out.write(str(make_dot(LL, params=dict(net.named_parameters()))))
    #out.close()
    #print("generated")
    #sys.exit()
writer.close()
