#!/usr/bin/env python3

import argparse
import gzip
import numpy as np
import pickle

def main():
  v2i = pickle.load(gzip.open(FLAGS_vocab_to_idx_pkl, 'rb'))
  #v2i = {k.lower(): v for k,v in v2i.items()}
  i2v = {v:k for k,v in v2i.items()}
  embeddings = np.zeros((len(v2i), FLAGS_embedding_dim), dtype=np.float32)
  for line in open(FLAGS_glove_file, 'r'):
    w = line[:line.find(' ')]
    if w in v2i:
      embeddings[v2i[w],:] = [float(x) for x in line.split()[1:]]
  # dim-wise moments:
  mean = np.mean(embeddings, axis=0)
  std = np.std(embeddings, axis=0)
  for i,v in enumerate(embeddings):
    if np.all(v == 0):
      print("No embedding found for {}; inserting random one".format(i2v[i]))
      embeddings[i,:] = np.random.normal(mean, std)
  with open(FLAGS_output, 'wb') as outf:
    np.save(outf, embeddings)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Data:
  parser.add_argument('--glove_file', type=str, required=True,
                      help="one of the downloaded GloVe txt files")
  parser.add_argument('--vocab_to_idx_pkl', type=str, required=False,
                      default="v2i.freq.pkl.gz", help="vocab -> index dict")
  parser.add_argument('--output', type=str, required=False,
                      default="infer_glove.npy",
                      help="where to put embeddings")
  parser.add_argument('--embedding_dim', type=int, default=300)
  parser.set_defaults(tie_rnns=True)
  args = parser.parse_args()
  for k,v in vars(args).items():
    globals()['FLAGS_%s' % k] = v
  main()
