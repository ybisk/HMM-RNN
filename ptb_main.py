# some code taken from https://github.com/pytorch/examples/blob/master/word_language_model/main.py

import sys, pickle, random, tqdm
import time, math, logging
import argparse
import numpy as np

import torch
from tensorboardX import SummaryWriter

import corpus_data
import rnn

parser = argparse.ArgumentParser(description='HMM-RNN')
parser.add_argument('--data-dir', type=str, default='./data/ptb-mikolov',
                    help='location of the data corpus')
parser.add_argument('--save', type=str, default='model',
                    help='path to save the final model')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

parser.add_argument('--type', type=str, default='hmm',
                    help='hmm|hmm+1|hmm+2|hmm-1|jordan|elman|rnn-hmm|rrnn|rrnn-1|rrnn-2|gru|lstm|rnn-1|rnn-2')
parser.add_argument('--feeding', type=str, default='word',
                    help='none|word|encode-lstm')
parser.add_argument('--glove-emb', action='store_true', default=False,
                    help='Use GloVe embeddings instead of learning embeddings.')
parser.add_argument('--test', action='store_true', default=False,
                    help='Evaluate on test data')

parser.add_argument('--max-len', type=int, default=35, help='max / BPTT seq len')
parser.add_argument('--batch-size', type=int, default=20, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--hidden-dim', type=int, default=100, help='hidden dim (num clusters for HMM)')
parser.add_argument('--embed-dim', type=int, default=100, help='embedding dim')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr-decay-rate', type=float, default=4.0,
                    help='learning rate decay per epoch')

parser.add_argument('--fixed-decay', action='store_true', 
                    help='follow fixed lr decay schedule (as in Zaremba et al)')
parser.add_argument('--num-init-lr-epochs', type=int, default=6, 
                    help='number of epochs before fixed learning rate decay')

parser.add_argument('--initrange', type=float, default=1.0, 
                    help='initial param range')
parser.add_argument('--tie-embeddings', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--optim', type=str, default='adam',
                    help='adam|sgd')

parser.add_argument('--patience', type=int, default=0, 
                    help='Stop training if not improving for some number of epochs')
parser.add_argument('--headless', action='store_true', help='kill prints')
parser.add_argument('--log', type=str, default='./log/', help='Log dir')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--note', type=str, default='',
                    help='extra note on fname')
parser.add_argument('--write-graph', action='store_true', default=False,
                    help='Write out computation graph.')
args = parser.parse_args()

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG) 
sys.stdout.flush()
sys.stderr.flush()

# Set the random seed manually for reproducibility.
if args.seed > 0:
  torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(threshold=1000, edgeitems=10)

def output_fname():
  fname = "{}".format(args.type)
  fname += ".{}".format(args.note) if len(args.note) > 0 else ""
  fname += "_{}".format(args.feeding)
  fname += "_l{}".format(args.max_len)
  fname += "_h{}".format(args.hidden_dim)
  if args.glove_emb:
    fname += "_fixedE100"
  else:
    fname += "_e{}".format(args.embed_dim)
  fname += "_lr{}".format(args.lr)
  fname += "_drop{}".format(args.dropout)
  fname += "_{}".format(args.optim)
  if args.tie_embeddings:
    fname += "_tieE"
  return fname

if args.glove_emb:
  assert args.embed_dim == 100

if args.headless:
  log_file = open(output_fname() + ".log", 'w')

def h_print(s):
  if not args.headless:
    print(s)
  else:
    log_file.write(s + "\n")

writer = SummaryWriter(args.log + output_fname())

h_print("Reading the data")
corpus = corpus_data.Corpus(args.data_dir)

def batchify(data, N):
  # Work out how cleanly we can divide the dataset into bsz parts.
  num_batches = data.size(0) // N
  # Trim off any extra elements that wouldn't cleanly fit (remainders).
  data = data.narrow(0, 0, num_batches * N)
  # Evenly divide the data across the bsz batches.
  data = data.view(N, -1).contiguous()
  return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

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

vocab_size = len(corpus.dict)

net = rnn.RNN(vocab_size, args).to(device)

def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""
  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
  seq_len = min(args.max_len, source.size()[1] - 1 - i)
  data = source[:,i:i+seq_len+1]
  return data

def evaluate(data_source):
  net.eval()
  total_loss = 0.0

  if args.type == 'jordan':
    hidden_state = net.init_hidden_state(eval_batch_size, data_source[:,0])
  else:
    hidden_state = net.init_hidden_state(eval_batch_size) 

  with torch.no_grad():
    for i in range(0, data_source.size(1) - 1, args.max_len):
      data_tensor = get_batch(data_source, i)
      emit_marginal, hidden_state = net(data_tensor, hidden_state)  
      loss = -1 * torch.mean(emit_marginal)
      total_loss += loss.item() * data_tensor.size()[1]
      hidden_state = repackage_hidden(hidden_state)
  return total_loss / (data_source.size(1) - 1)

num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
h_print("Trainable parameters: %d" % num_parameters)

h_print("Starting Training")
"""
  Train
"""
if args.write_graph:
  writer.add_graph(model=net.Net(), input_to_model=torch.ones((2,2)), verbose=True)

lr = args.lr

if args.optim == 'adam':
  optimizer = torch.optim.Adam(net.parameters(), lr=lr) #, weight_decay=1e-3)
else:
  optimizer = torch.optim.SGD(net.parameters(), lr=lr) #, weight_decay=1e-3)

best_val_loss = None

TBstep = 0
patience_count = 0
for epoch in range(args.epochs):
  step = 0
  # Training
  #if 'hmm' in args.type:
  #  print_emissions(net, output_fname(), corpus.dict.i2voc)

  net.train() 
  total_loss = 0.0
  start_time = time.time()

  if args.type == 'jordan':
    hidden_state = net.init_hidden_state(args.batch_size, train_data[:,0]) 
  else:
    hidden_state = net.init_hidden_state(args.batch_size) 
 
  inds = list(range(train_data.size(1) - 1))
  iterate = tqdm.tqdm(range(0, len(inds) - len(inds)%args.max_len, args.max_len), ncols=80, disable=args.headless)
  for i in iterate:
    data_tensor = get_batch(train_data, i)
    hidden_state = repackage_hidden(hidden_state)
    net.zero_grad()
    
    emit_marginal, hidden_state = net(data_tensor, hidden_state)

    # emit_marginal[emit_marginal != emit_marginal] = 0 # hack to avoid NaNs
    loss = -1 * torch.mean(emit_marginal)
    loss.backward()

    if args.clip > 0:
      torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
    optimizer.step()

    # Rather using SGD optimizer  
    #if args.optim == 'sgd':
    #  for p in net.parameters():
    #    assert hasattr(p.grad, "data"), "Network parameter not in computation graph."
    #    p.data.add_(-lr, p.grad.data)

    total_loss += loss.item()

    iterate.set_description("Loss {:8.4f}".format(loss.item()))

    step += 1
    TBstep += 1
    writer.add_scalar('Loss', loss.item(), TBstep)

  cur_loss = total_loss / step # loss calculation might be approximate
  elapsed = time.time() - start_time
  h_print('| epoch {:3d} | {:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, step, lr, elapsed * 1000 / step, cur_loss, math.exp(cur_loss)))
    
  val_loss = evaluate(val_data)
  h_print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
        epoch, (time.time() - start_time), val_loss, math.exp(val_loss)))

  writer.add_scalar('Prp_Train', math.exp(cur_loss), epoch)
  writer.add_scalar('Prp_Val', math.exp(val_loss), epoch)

  # Save the model if the validation loss is the best we've seen so far.
  if not best_val_loss or val_loss < best_val_loss:
    with open(args.save + '.' + output_fname() + '.pt', 'wb') as f:
      torch.save(net, f)
    best_val_loss = val_loss
    patience_count = 0
  else:
    patience_count += 1
    # Anneal the learning rate if no improvement has been seen in the validation dataset.
    if args.optim == 'sgd' and not args.fixed_decay:
      lr /= args.lr_decay_rate

  if (args.optim == 'sgd' and args.fixed_decay 
      and epoch >= args.num_init_lr_epochs):
    lr /= args.lr_decay_rate

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  if args.write_graph:
    out = open("graph.dot",'w')
    out.write(str(make_dot(loss, params=dict(net.named_parameters()))))
    out.close()
    h_print("generated graph")
    sys.exit()

  if args.patience > 0 and patience_count >= args.patience:
    break

if args.test:
  # Load the best saved model.
  with open(args.save + '.' + output_fname() + '.pt', 'rb') as f:
      net = torch.load(f)
      # after load the rnn params are not a continuous chunk of memory
      # this makes them a continuous chunk, and will speed up forward pass
      # net.rnn.flatten_parameters() #TODO

  # Run on test data.
  test_loss = evaluate(test_data)
  h_print('=' * 89)
  h_print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
      test_loss, math.exp(test_loss)))
  h_print('=' * 89)

writer.close()
