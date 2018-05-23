#!/usr/bin/env python3

''' Like pickle_data, but selects vocabulary based on unigram frequency, rather
than just selecting all the train words---this can hopefully allow us to learn a
decent <UNK> embedding...
'''

# TODO try replacing global <UNK> with spacy's POS for uncommon words.

import argparse
import collections
import csv
import gzip
import os
import pandas
import pickle
#import spacy
import sys

#nlp = spacy.load('en')


SPECIAL_TOKENS = ["#UNK#", "<PAD>", "<SEP>", "<S>", "</S>"]
special_v2i = {w:i for i,w in enumerate(SPECIAL_TOKENS)}
UNK_IDX = special_v2i["#UNK#"]

# not including "?" or ";" because they're not common enough in training set

def main():
  PTB = [line.strip().split() for line in gzip.open("../wsj00-18.raw.txt.gz")]

  """
    Build Vocabulary and training
  """
  print("Building vocab...")
  sys.stdout.flush()

  v2i = {v:i for i,v in enumerate(SPECIAL_TOKENS)}
  i2v = {v:k for k,v in v2i.items()}
  counts = collections.defaultdict(int)
  for line in PTB:
    line = [w.lower() for w in line]
    for word in line:
      counts[word] += 1
  ct_li = list(counts.items())
  ct_li.sort(key = lambda x: x[1], reverse=True)
  ct_li = [(w, None) for w in SPECIAL_TOKENS] + ct_li[:FLAGS_vocab_size]
  v2i = {pair[0]: i for i,pair in enumerate(ct_li)}
  i2v = {v:k for k,v in v2i.items()}

  def txt_to_idx_seq(txt):
    ''' tokenizes, replaces OOV with UNK
    '''
    return [v2i.get(t.lower(), UNK_IDX) for t in txt]

  print("Processing Train...")
  sys.stdout.flush()

  train = []
  for line in PTB:
    train.append(txt_to_idx_seq(line))

  print("Writing to file...")
  sys.stdout.flush()
  pickle.dump(v2i, gzip.open(os.path.join(FLAGS_data_dir,
                                          FLAGS_vocab_to_idx_pkl), 'wb'))
  pickle.dump(i2v, gzip.open(os.path.join(FLAGS_data_dir,
                                          FLAGS_idx_to_vocab_pkl),
                             'wb'))
  pickle.dump(train, gzip.open(os.path.join(FLAGS_data_dir, FLAGS_train_pkl),
                               'wb'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, required=False,
                      default="../", help="train/val/vocab pickle dir.")
  parser.add_argument('--train_pkl', type=str, required=False,
                      default="train.freq.pkl.gz",
                      help="gzipped train set pickle")
  parser.add_argument('--vocab_to_idx_pkl', type=str, required=False,
                      default="v2i.freq.pkl.gz", help="vocab -> index dict")
  parser.add_argument('--idx_to_vocab_pkl', type=str, required=False,
                      default="i2v.freq.pkl.gz", help="vocab -> index dict")
  parser.add_argument('--vocab_size', type=int, required=False,
                      default=1000, help="num words to put in vocab")
  args = parser.parse_args()
  for k,v in vars(args).items():
    globals()['FLAGS_%s' % k] = v
  main()

