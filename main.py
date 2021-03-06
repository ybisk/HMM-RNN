import sys, pickle, gzip, random, tqdm
import numpy as np
import torch
import argparse

from tensorboardX import SummaryWriter

import rnn

def print_emissions(net, fname, i2voc):
  o = open("Emissions.{}.txt".format(fname),'w')
  e_list = net.emissions_list()
  for i in range(net.hidden_dim):
    listed = [(float(e_list[j][i]), str(i2voc[j])) for j in range(net.vocab_size)]
    listed.sort()
    listed.reverse()
    o.write("\n%d\n" % i)
    for prob, word in listed[:50]:
      o.write("   {:10.8f}  {:10s}\n".format(100*prob, str(word)))
  o.close()

parser = argparse.ArgumentParser(description='HMM-RNN')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--max-len', type=int, default=20, help='max seq len')
parser.add_argument('--hidden-dim', type=int, default=100, help='hidden dim (num clusters for HMM)')
parser.add_argument('--embed-dim', type=int, default=100, help='embedding dim')
parser.add_argument('--feeding', type=str, default='word',
                    help='none|word|encode-lstm')
parser.add_argument('--one-hot', action='store_true', default=False, 
                    help='1-hot clusters') #TODO not implemented any more
parser.add_argument('--glove-emb', action='store_true', default=False,
                    help='Use GloVe embeddings instead of learning embeddings.')
parser.add_argument('--log', type=str, default='./log/', help='Log dir')
parser.add_argument('--type', type=str, default='hmm',
                    help='hmm|hmm+1|jordan|elman|dist|gru|lstm')
parser.add_argument('--note', type=str, default='',
                    help='extra note on fname')
parser.add_argument('--write-graph', action='store_true', default=False,
                    help='Write out computation graph.')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(threshold=1000, edgeitems=10)
#torch.manual_seed(1)

fname = "{}".format(args.type)
fname += ".{}".format(args.note) if len(args.note) > 0 else ""
if args.one_hot:
  fname += "_1hot"
fname += "_{}".format(args.feeding)
fname += "_l{}".format(args.max_len)
fname += "_h{}".format(args.hidden_dim)
if args.glove_emb:
  assert args.embed_dim == 100
  fname += "_fixedE100"
else:
  fname += "_e{}".format(args.embed_dim)

writer = SummaryWriter(args.log + fname)

def expand(text):
  text.insert(START, 0)
  while len(text) < args.max_len:
    text.append(NONE)
  return text[:args.max_len]

#TODO (Jan): Use Mikolov PTB
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

net = rnn.RNN(vocab_size, args).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4) #, weight_decay=1e-3)

print("Starting Training")
"""
  Train
"""
if args.write_graph:
  writer.add_graph(model=model.Net(), input_to_model=torch.ones((2,2)), verbose=True)

step = 0
for epoch in range(args.epochs):
  # Training
  print("Epoch %d" % epoch)
  if 'hmm' in args.type:
    print_emissions(net, fname, i2voc)

  #TODO implement truncated backprop with fixed length subsequences

  # Training
  inds = list(range(len(data)))
  random.shuffle(inds)
  iterate = tqdm.tqdm(range(0, len(inds) - len(inds)%args.batch_size, args.batch_size), ncols=80)
  for i in iterate:
    data_range = inds[i:i + args.batch_size]
    data_tensor = torch.from_numpy(data[data_range]).to(device)

    optimizer.zero_grad()
    net.train() 
    
    #TODO carry over hidden state between batches
    if args.type == 'jordan':
      hidden_state = net.init_hidden_state(args.batch_size, data_tensor[:,0]) 
    else:
      hidden_state = net.init_hidden_state(args.batch_size) 
    
    emit_marginal, hidden_state = net(data_tensor, hidden_state)

    loss = -1 * torch.mean(emit_marginal)
    loss.backward()
    optimizer.step()

    iterate.set_description("Loss {:8.4f}".format(loss.item()))

    step += 1
    writer.add_scalar('Loss', loss.item(), step)

    #TODO compute loss on validation data
    #TODO Compute Perplexity 2^-Sum(p * log2(p))

    if args.write_graph:
      out = open("graph.dot",'w')
      out.write(str(make_dot(loss, params=dict(net.named_parameters()))))
      out.close()
      print("generated")
      sys.exit()

writer.close()
