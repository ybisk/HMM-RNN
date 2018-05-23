#!/usr/bin/env python3

import argparse
from functools import reduce
import gzip
import itertools
import numpy as np
import os
import pickle
from pprint import pprint
import random
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import tqdm
#torch.manual_seed(1)

from inference_dataset import get_dataset, get_negative_examples

UNK = None
PAD = None
DOT = None
SEP = None
VOCAB_SIZE = None
STOPWORDS = None

# TODO need <S> and </S>
# TODO try batchnorm
# TODO we're not gradient clipping
# TODO need to save dataset
# TODO need to save model (particularly after stopping based on val performance)

class LogReg(nn.Module):
  def __init__(self, input_dim):
    print("Logistic Regression model has {} features...".format(input_dim))
    super(LogReg, self).__init__()
    self.input_dim = input_dim
    self.linear = nn.Linear(input_dim, 2, bias=False).cuda()

  def forward(self, x):
    #print(self.linear.weight.data)
    out = self.linear(x)
    return out

class TENet(nn.Module):
  ''' Textual Entailment Model.
  '''
  def __init__(self):
    super(TENet, self).__init__()
    self.hidden_dim = FLAGS_hidden_dim
    self.embed_dim = FLAGS_embedding_dim
    self.output_dim = 2

    # load pretrained embeddings:
    self.embeddings = nn.Embedding(VOCAB_SIZE, self.embed_dim).cuda()
    self.embeddings.weight.data.copy_(
        torch.from_numpy(np.load(FLAGS_glove_embeddings)))
    # freeze embeddings:
    self.embeddings.requires_grad = False
    #self.dropout_embeddings = nn.Dropout(p=FLAGS_dropout_rate)
    self.embeddings_FC = nn.Linear(self.embed_dim, self.embed_dim)
    self.dropout_embeddings_FC = nn.Dropout(p=FLAGS_dropout_rate)
    self.concrete_rnn = nn.LSTM(self.embed_dim, self.hidden_dim,
                                dropout=FLAGS_dropout_rate,
                                batch_first=True,
                                bidirectional=FLAGS_bidirectional)
    if FLAGS_tie_rnns:
      self.abstract_rnn = self.concrete_rnn
    else:
      self.abstract_rnn = nn.LSTM(self.embed_dim, self.hidden_dim,
                                  dropout=FLAGS_dropout_rate,
                                  batch_first=True,
                                  bidirectional=FLAGS_bidirectional)
    self.total_hidden_dim = 2 * self.hidden_dim
    if FLAGS_bidirectional:
      self.total_hidden_dim *= 2
    self.FC1 = nn.Linear(self.total_hidden_dim, self.total_hidden_dim)
    self.dropout_FC1 = nn.Dropout(p=FLAGS_dropout_rate)
    self.FC2 = nn.Linear(self.total_hidden_dim, self.output_dim)

  def forward(self, x):
    ''' the first FLAGS_concrete_length steps should be the concrete seqs; the
        next FLAGS_concrete_length ones should be the abstract seqs.
    '''
    x = F.relu(self.dropout_embeddings_FC(
               self.embeddings_FC(
               self.embeddings(x))))
    #x = self.dropout_embeddings(self.embeddings(x)) # batch, len, dim
    out_conc, _ = self.concrete_rnn(x.narrow(1, 0, FLAGS_concrete_length))
    out_abst, _ = self.abstract_rnn(x.narrow(1, FLAGS_concrete_length,
                                             FLAGS_abstract_length))
    #print(out_conc.size())
    #print(out_abst.size())
    #print(torch.sum(out_conc, dim=1).size())
    #print(torch.sum(out_abst, dim=1).size())
    #print('---')
    #out = torch.cat((torch.squeeze(out_conc[:,-1,:]),
    #                 torch.squeeze(out_abst[:,-1,:])),
    #                dim=1)

    # sum over time in both LSTMs, concat sums together:
    out = torch.cat((torch.sum(out_conc, dim=1),
                     torch.sum(out_abst, dim=1)),
                    dim=1)
    out = F.relu(self.dropout_FC1(self.FC1(out)))
    out = self.FC2(out)
    return out

class TENet_Series(nn.Module):
  ''' Textual Entailment Model. Runs long text through LSTM first, outputs the
      hidden state from that net into the LSTM on the short text.
  '''
  # TODO why isn't this thing training afs;l;asfdjklsdafkjlsadfj;sadfj;sfda;j
  def __init__(self):
    super(TENet_Series, self).__init__()
    self.hidden_dim = FLAGS_hidden_dim
    self.embed_dim = FLAGS_embedding_dim
    self.output_dim = 2

    # load pretrained embeddings:
    self.embeddings = nn.Embedding(VOCAB_SIZE, self.embed_dim).cuda()
    self.embeddings.weight.data.copy_(
        torch.from_numpy(np.load(FLAGS_glove_embeddings)))
    # freeze embeddings:
    self.embeddings.requires_grad = False
    #self.dropout_embeddings = nn.Dropout(p=FLAGS_dropout_rate)
    self.embeddings_FC = nn.Linear(self.embed_dim, self.embed_dim)
    self.dropout_embeddings_FC = nn.Dropout(p=FLAGS_dropout_rate)
    #self.total_hidden_dim = 2 * self.hidden_dim
    self.total_hidden_dim = self.hidden_dim
    if FLAGS_bidirectional:
      self.total_hidden_dim *= 2
    # input longer concrete RNN first
    self.concrete_rnn = nn.LSTM(self.embed_dim, self.hidden_dim,
                                dropout=FLAGS_dropout_rate,
                                batch_first=True,
                                bidirectional=FLAGS_bidirectional)
    self.abstract_rnn = nn.LSTM(self.embed_dim + self.total_hidden_dim,
                                self.hidden_dim,
                                dropout=FLAGS_dropout_rate,
                                batch_first=True,
                                bidirectional=FLAGS_bidirectional)
    self.FC1 = nn.Linear(self.total_hidden_dim, self.total_hidden_dim)
    self.dropout_FC1 = nn.Dropout(p=FLAGS_dropout_rate)
    if True:
      self.FC1 = nn.Linear(2 * self.total_hidden_dim, 2 * self.total_hidden_dim)
    self.FC2 = nn.Linear(self.total_hidden_dim, self.output_dim)
    if True:
      self.FC2 = nn.Linear(2 * self.total_hidden_dim, self.output_dim)
    self.dropout_hidden = nn.Dropout(p=FLAGS_dropout_rate)

  def forward(self, x):
    ''' the first FLAGS_abstract_length steps should be the concrete seqs; the
        next FLAGS_concrete_length ones should be the abstract seqs.
    '''
    x = F.relu(self.dropout_embeddings_FC(
               self.embeddings_FC(
               self.embeddings(x))))
    #x = self.dropout_embeddings(self.embeddings(x)) # batch, len, dim
    out_conc, _ = self.concrete_rnn(x.narrow(1, 0, FLAGS_concrete_length))

    # input abstract embeddings concatted with concrete output states summed
    # over time:
    #print(x.narrow(1, FLAGS_abstract_length, FLAGS_concrete_length).size())
    #print(torch.sum(out_conc, dim=1).size())
    #print(torch.sum(out_conc, dim=1)
    #      .unsqueeze(1).size())
    #print((FLAGS_batch_size, FLAGS_abstract_length, self.total_hidden_dim))
    #print(torch.sum(out_conc, dim=1)
    #      .unsqueeze(1)
    #      .expand(FLAGS_batch_size, FLAGS_abstract_length, self.total_hidden_dim)
    #      .size())
    broadcasted_concrete_states = (
        #torch.sum(out_conc, dim=1)
        self.dropout_hidden(out_conc[:,-1,:])
        .unsqueeze(1)
        .expand(FLAGS_batch_size, FLAGS_abstract_length, self.total_hidden_dim))
    #print(x.narrow(1, FLAGS_concrete_length, FLAGS_abstract_length))
    #print(broadcasted_concrete_states)
    #print('------------')
    out_abst, _ = self.abstract_rnn(torch.cat(
      [x.narrow(1, FLAGS_concrete_length, FLAGS_abstract_length),
       broadcasted_concrete_states],
      dim=2))
    if True:
      out_conc2, _ = self.concrete_rnn(x.narrow(1, FLAGS_concrete_length,
                                                FLAGS_abstract_length))
      broadcasted_concrete_states2 = (
          self.dropout_hidden(out_conc2[:,-1,:])
          .unsqueeze(1)
          .expand(FLAGS_batch_size, FLAGS_concrete_length,
                  self.total_hidden_dim))
      out_abst2, _ = self.abstract_rnn(torch.cat(
        [x.narrow(1, 0, FLAGS_concrete_length),
         broadcasted_concrete_states2],
        dim=2))
    #print(out_conc.size())
    #print(out_abst.size())
    #print(torch.sum(out_conc, dim=1).size())
    #print(torch.sum(out_abst, dim=1).size())
    #print('---')
    #out = torch.cat((torch.squeeze(out_conc[:,-1,:]),
    #                 torch.squeeze(out_abst[:,-1,:])),
    #                dim=1)

    #out = torch.sum(out_abst, dim=1)
    #out = out_abst[:,-1,:]
    #print(out_abst[:,-1,:].size())
    #print(out_abst2[:,-1,:].size())
    out = torch.cat([out_abst[:,-1,:], out_abst2[:,-1,:]], dim=1)

    #out = torch.cat((torch.sum(out_conc, dim=1),
    #                 torch.sum(out_abst, dim=1)),
    #                dim=1)
    out = F.relu(self.dropout_FC1(self.FC1(out)))
    out = self.FC2(out)
    return out

def main():
  tdata, vdata = read_data()
  voc2i, i2voc = read_vocab()
  populate_global_word_constants(voc2i)
  print_dataset_vocab_info(tdata, vdata, voc2i, i2voc)
  negative_train_examples = get_negative_examples(tdata, FLAGS_negative_ratio)
  negative_valid_examples = get_negative_examples(vdata, FLAGS_negative_ratio)
  train_conc, train_abst, train_targ = get_dataset(
      positive=tdata,
      negative=negative_train_examples,
      abstract_length=FLAGS_abstract_length,
      concrete_length=FLAGS_concrete_length,
      pad=PAD,
      sep=SEP)
  valid_conc, valid_abst, valid_targ = get_dataset(
      positive=vdata,
      negative=negative_valid_examples,
      abstract_length=FLAGS_abstract_length,
      concrete_length=FLAGS_concrete_length,
      pad=PAD,
      sep=SEP)
  assert len(train_conc) == len(train_abst)
  assert len(train_conc) == len(train_targ)
  assert len(valid_conc) == len(valid_abst)
  assert len(valid_conc) == len(valid_targ)
  #pprint(list(zip((' '.join(i2voc[w] for w in seq) for seq in valid_conc),
  #                (' '.join(i2voc[w] for w in seq) for seq in valid_abst),
  #                valid_targ)))
  print("With negative examples, Train is {} seqs, validation is {} seqs."
        .format(len(train_conc), len(valid_conc)))
  data = {'train_conc': train_conc, 'train_abst': train_abst,
          'train_targ': train_targ, 'valid_conc': valid_conc,
          'valid_abst': valid_abst, 'valid_targ': valid_targ,
          'orig_tdata': tdata}
  get_baseline_performance(data)
  print('-' * 80)
  print("Training LSTM...")
  print('-' * 80)
  #net = TENet_Series()
  net = TENet()
  #net = TENet()
  net.cuda()
  train(net, data)

def get_baseline_performance(data):
  print("Training baseline Logistic Regression model...")
  train_x, train_y, val_x, val_y = get_logreg_features(data)
  model = LogReg(train_x.shape[1])
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  nsteps = train_x.shape[0] // FLAGS_batch_size
  for epoch in range(FLAGS_n_epochs):
    print("Logreg epoch {}...".format(epoch))
    total_loss = 0.0
    acc = 0
    total = 0
    for step in range(nsteps):
      lb = step*FLAGS_batch_size
      ub = lb + FLAGS_batch_size
      batch_x = Variable(torch.from_numpy(train_x[lb:ub, :]).cuda()).float()
      batch_y = Variable(torch.from_numpy(train_y[lb:ub]).cuda()).long()
      optimizer.zero_grad()
      model.train(mode=True)
      logits = model(batch_x)
      loss = F.cross_entropy(logits, batch_y)
      total_loss += loss.data[0]
      loss.backward()
      optimizer.step()
      acc += (torch.max(logits, 1)[1] == batch_y).sum().data[0]
      total += FLAGS_batch_size
    acc *= 100.0/total

    # Evaluation
    model.train(mode=False)
    v_loss = 0.0
    v_acc = 0
    v_total = 0
    nsteps = val_x.shape[0] // FLAGS_batch_size
    for step in range(nsteps):
      lb = step*FLAGS_batch_size
      ub = lb + FLAGS_batch_size
      batch_x = Variable(torch.from_numpy(val_x[lb:ub, :]).cuda()).float()
      batch_y = Variable(torch.from_numpy(val_y[lb:ub]).cuda()).long()
      v_logits = model(batch_x)
      v_loss += F.cross_entropy(v_logits, batch_y).data[0]
      v_acc += (torch.max(v_logits, 1)[1] == batch_y).sum().data[0]
      v_total += FLAGS_batch_size
    v_acc *= 100.0 / total

    print("Train acc {:.2f}; loss {:.2f}\tVal acc {:.2f}; loss {:.2f}"
          .format(acc, total_loss, v_acc, v_loss))
  output_logreg_confidences(train_x, train_y, data['train_conc'],
                            data['train_abst'], model, 'train_logreg.txt')

    #print("PARAM GRAD/NORM ratios:")
    #print([np.linalg.norm(x.grad.data.cpu().numpy().flatten()) /
    #       np.linalg.norm(x.data.cpu().numpy().flatten())
    #       for x in model.parameters()])

def output_logreg_confidences(x, y, concs, absts, model, out_fname):
  model.train(mode=False)
  nsteps = x.shape[0] // FLAGS_batch_size
  y_onehot = np.zeros(shape=(y.shape[0], 2), dtype=np.float32)
  y_onehot[np.arange(y.shape[0]), y] = 1
  # a list of the amount or probability mass the model _should_ have assigned
  # to the right answer but didn't:
  wrong_pct = np.zeros(shape=y.shape[0], dtype=np.float32)
  for step in range(nsteps):
    lb = step*FLAGS_batch_size
    ub = lb + FLAGS_batch_size
    if step == nsteps - 1:
      ub = x.shape[0]
    batch_x = Variable(torch.from_numpy(x[lb:ub, :]).cuda()).float()
    batch_y = Variable(torch.from_numpy(y[lb:ub]).cuda()).long()
    probs = F.softmax(model(batch_x))
    prob_diffs = y_onehot[lb:ub, :] - probs.data.cpu().numpy()
    guesses = np.argmax(prob_diffs, axis=1)
    wrong_pct[lb:ub] = prob_diffs[np.arange(prob_diffs.shape[0]),
                                  np.argmax(prob_diffs, axis=1)]
  pct_idxs = np.flip(np.argsort(wrong_pct), axis=0)
  with open(out_fname, 'w') as outf:
    for i in pct_idxs:
      outf.write('{}\t{}\t{}\n'.format(wrong_pct[i], y[i], i))
      outf.write('{}\n'.format(' '.join(I2V[w] for w in concs[i] if w != PAD)))
      outf.write('{}\n'.format(' '.join(I2V[w] for w in absts[i] if w != PAD)))

def get_logreg_features(data):
  ''' return trainx, trainy, valx, valy 4-tuple.

  The first 2 and last 2 will stand in correspondence to each other.
  '''
  return (
      logreg_features_from_dataset(data['train_conc'], data['train_abst']),
      data['train_targ'],
      logreg_features_from_dataset(data['valid_conc'], data['valid_abst']),
      data['valid_targ'])

def logreg_features_from_dataset(conc, abst):
  ''' returns ndarray with same shape[0] as the input params
  '''
  assert conc.shape[0] == abst.shape[0]
  feats = {}
  feats['unigram_overlap_pct'] = get_unigram_overlap_pct(conc, abst)
  feats['unigram_overlap_rev'] = get_unigram_overlap_pct(abst, conc)
  #print(feats)
  return np.concatenate(list(feats.values()), axis=1)

def get_unigram_overlap_pct(conc, abst):
  res = np.ndarray(shape=(len(conc), 1), dtype=np.float32)
  for i, (this_conc, this_abst) in enumerate(zip(conc, abst)):
    this_conc_set = set(x for x in this_conc if x != PAD and x not in STOPWORDS)
    this_abst_li = [x for x in this_abst if x != PAD and x not in STOPWORDS]
    res[i] = (len([x for x in this_abst_li if x in this_conc_set]) /
              len(this_abst_li))
  return res


def train(net, data):
  optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS_lr,
                               weight_decay=FLAGS_weight_decay)
  #optimizer = torch.optim.SGD(net.parameters(),
  #                            lr=FLAGS_lr,
  #                            momentum=FLAGS_momentum,
  #                            weight_decay=FLAGS_weight_decay)
  for epoch in range(FLAGS_n_epochs):
    print("Epoch {}...".format(epoch))
    if FLAGS_new_samples_each_epoch:
      negative_train_examples = get_negative_examples(data['orig_tdata'],
                                                      FLAGS_negative_ratio)
      data['train_conc'], data['train_abst'], data['train_targ'] = \
          get_dataset(positive=data['orig_tdata'],
                      negative=negative_train_examples,
                      abstract_length=FLAGS_abstract_length,
                      concrete_length=FLAGS_concrete_length,
                      pad=PAD,
                      sep=SEP)
    indices = list(range(len(data['train_conc'])))
    steps = int(len(indices)/FLAGS_batch_size)
    total_loss = 0
    acc = 0
    total = 0
    for step in tqdm.tqdm(range(steps)):
      r = indices[:FLAGS_batch_size]
      indices = indices[FLAGS_batch_size:]

      optimizer.zero_grad()
      inputs = Variable(torch.from_numpy(
        np.concatenate((data['train_conc'][r], data['train_abst'][r]),
                       axis=1)).cuda())
      labels = Variable(torch.from_numpy(data['train_targ'][r]).cuda())

      net.train(mode=True)
      logits = net(inputs)
      loss = F.cross_entropy(logits, labels)
      total_loss += loss.data[0]
      loss.backward()
      optimizer.step()

      #print("PARAM GRAD/NORM ratios:")
      #print([np.linalg.norm(x.grad.data.cpu().numpy().flatten()) /
      #       np.linalg.norm(x.data.cpu().numpy().flatten())
      #       for x in net.parameters()])

      acc += (torch.max(logits, 1)[1] == labels).sum().data[0]
      total += FLAGS_batch_size
    acc *= 100.0/total

    # Evaluation
    net.train(mode=False)
    v_inds = range(len(data['valid_conc']))
    v_loss = 0
    v_acc = 0
    v_total = 0
    while len(v_inds) > FLAGS_batch_size:
      v_r = v_inds[:FLAGS_batch_size]
      v_inds = v_inds[FLAGS_batch_size:]
      v_inputs = Variable(torch.from_numpy(
        np.concatenate((data['valid_conc'][v_r], data['valid_abst'][v_r]),
                       axis=1)).cuda())
      v_labels = Variable(torch.from_numpy(data['valid_targ'][v_r]).cuda())
      v_logits = net(v_inputs)
      v_loss += F.cross_entropy(v_logits, v_labels).data[0]
      v_acc += (torch.max(v_logits, 1)[1] == v_labels).sum().data[0]
      v_total += FLAGS_batch_size
      #print_examples(v_inputs.data, v_logits.data, v_labels.data)
    v_acc *= 100.0/v_total

    print("Train acc {:.2f}; loss {:.2f}\tVal acc {:.2f}; loss {:.2f}"
          .format(acc, total_loss, v_acc, v_loss))
    sys.stdout.flush()

def print_examples(v_inputseqs, v_logits, golds):
  for seq, logits, gold in zip(v_inputseqs, v_logits, golds):
    print('C: ' + ' '.join(I2V[x]
                           for x in seq[:FLAGS_concrete_length]
                           if x != PAD))
    print('A: ' + ' '.join(I2V[x]
                           for x in seq[FLAGS_concrete_length:]
                           if x != PAD))
    logits = logits.cpu().numpy()
    print('preds: {} (gold: {})'.format(
      np.exp(logits) / np.sum(np.exp(logits)),
      gold
      ))


def read_data():
  ''' returns (train, val) pair
  '''
  return (pickle.load(gzip.open(os.path.join(FLAGS_data_dir, FLAGS_train_pkl),
                                'rb')),
          pickle.load(gzip.open(os.path.join(FLAGS_data_dir, FLAGS_val_pkl),
                                'rb')))

def read_vocab():
  ''' returns (v2i, i2v) pair
  '''
  return (pickle.load(gzip.open(os.path.join(FLAGS_data_dir,
                                             FLAGS_vocab_to_idx_pkl),
                                'rb')),
          pickle.load(gzip.open(os.path.join(FLAGS_data_dir,
                                             FLAGS_idx_to_vocab_pkl),
                                'rb')))

def populate_global_word_constants(voc2i):
  from stopwords import stopwords
  global UNK
  global PAD
  global DOT
  global SEP
  global VOCAB_SIZE
  global I2V
  global V2I
  global STOPWORDS
  UNK = voc2i["#UNK#"]
  PAD = voc2i["<PAD>"]
  DOT = voc2i["."]
  SEP = voc2i["<SEP>"]
  VOCAB_SIZE = len(voc2i)
  V2I = voc2i
  I2V = {v:k for k,v in voc2i.items()}
  STOPWORDS = set([V2I[x] for x in stopwords if x in V2I])


def print_dataset_vocab_info(tdata, vdata, voc2i, i2voc):
  def print_data_info(tuples):
    nwords = sum([sum([len(seq) for seq in seqs])
                  for seqs in tuples])
    n_oov = sum([sum([len([w for w in seq if w == UNK ]) for seq in seqs])
                 for seqs in tuples])
    print("{:.4f} OOV, out of {} tokens in {} examples.".format(n_oov / nwords,
                                                                nwords,
                                                                len(tuples)))
  print("Using vocab of {} words.".format(VOCAB_SIZE))
  print("Training set:")
  print_data_info(tdata)
  print("Dev set:")
  print_data_info(vdata)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Data:
  parser.add_argument('--data_dir', type=str, required=False,
                      default="../data", help="train/val/vocab pickle dir.")
  parser.add_argument('--train_pkl', type=str, required=False,
                      default="train.freq.pkl.gz",
                      help="gzipped train set pickle")
  parser.add_argument('--val_pkl', type=str, required=False,
                      default="valid.freq.pkl.gz", help="gzipped val set pickle")
  parser.add_argument('--vocab_to_idx_pkl', type=str, required=False,
                      default="v2i.freq.pkl.gz", help="vocab -> index dict")
  parser.add_argument('--idx_to_vocab_pkl', type=str, required=False,
                      default="i2v.freq.pkl.gz", help="vocab -> index dict")
  parser.add_argument('--negative_ratio', type=float, required=False,
                      default=0.5,
                      help="percent of the dataset that will imposters.")
  # Training setup stuff:
  parser.add_argument('--abstract_length', type=int, required=False, default=20,
                      help="length we clip/pad concrete YouCook annotations to")
  parser.add_argument('--concrete_length', type=int, required=False, default=50,
                      help="length we clip/pad collected annotations to")
  parser.add_argument('--batch_size', type=int, required=False, default=32,
                      help="Sometimes grownups use a thing called batch SGD")
  parser.add_argument('--dropout_rate', type=float, required=False, default=0.5,
                      help="Sometimes grownups use a thing called Dropout")
  parser.add_argument('--lr', type=float, required=False, default=1e-3,
                      help="learning rate")
  parser.add_argument('--momentum', type=float, required=False, default=0.9,
                      help="momentum param")
  parser.add_argument('--weight_decay', type=float, required=False,
                      default=1e-4,
                      help="L2 regularization weight")
  parser.add_argument('--n_epochs', type=int, required=False, default=100,
                      help="how many times this widening gyre of trash turns")
  parser.add_argument('--new_samples_each_epoch', dest='new_samples_each_epoch',
                      action='store_true',
                      help="Should we sample new negative examples each pass?")
  parser.add_argument('--no_new_samples_each_epoch',
                      dest='new_samples_each_epoch',
                      action='store_false')
  parser.set_defaults(new_samples_each_epoch=True)
  # Model Params:
  parser.add_argument('--hidden_dim', type=int, required=False, default=25,
                      help="RNN hidden dim")
  parser.add_argument('--embedding_dim', type=int, required=False, default=300,
                      help="size of embeddings. If using pretrained GloVe "
                      "vectors, this needs to match up with that dim.")
  parser.add_argument('--glove_embeddings', type=str,
                      default="infer_glove.npy",
                      help="output of get_glove_embeddings.py")
  parser.add_argument('--bidirectional', dest='bidirectional',
                      action='store_true')
  parser.add_argument('--unidirectional', dest='bidirectional',
                      action='store_false')
  parser.set_defaults(bidirectional=True)
  parser.add_argument('--tie_rnns', dest='tie_rnns',
                      action='store_true')
  parser.add_argument('--dont_tie_rnns', dest='tie_rnns',
                      action='store_false')
  parser.set_defaults(tie_rnns=True)
  args = parser.parse_args()
  for k,v in vars(args).items():
    globals()['FLAGS_%s' % k] = v
  assert FLAGS_negative_ratio > 0 and FLAGS_negative_ratio < 1
  main()
