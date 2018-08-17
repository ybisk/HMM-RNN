# Source: https://github.com/pytorch/examples/blob/master/word_language_model/data.py

import os
import torch
import gzip

class Dictionary(object):
  def __init__(self):
    self.voc2i = {}
    self.i2voc = []

  def add_word(self, word):
    if word not in self.voc2i:
      self.i2voc.append(word)
      self.voc2i[word] = len(self.i2voc) - 1
    return self.voc2i[word]

  def __len__(self):
    return len(self.i2voc)


class Corpus(object):
  def __init__(self, path):
    self.dict = Dictionary()
    self.train = self.tokenize(os.path.join(path, 'train.txt.gz'))
    self.valid = self.tokenize(os.path.join(path, 'valid.txt.gz'))
    self.test = self.tokenize(os.path.join(path, 'test.txt.gz'))

  def tokenize(self, path):
    """Tokenizes a text file."""
    assert os.path.exists(path)

    # Add words to the dictionary
    with gzip.open(path, 'r') as f:
      self.dict.add_word('<eos>') # add as start symbol
      tokens = 1
      for line in f:
        words = line.split() + ['<eos>']
        tokens += len(words)
        for word in words:
          self.dict.add_word(word)

    # Tokenize file content
    with gzip.open(path, 'r') as f:
      ids = torch.LongTensor(tokens)
      ids[0] = self.dict.voc2i['<eos>']
      token = 1
      for line in f:
        words = line.split() + ['<eos>']
        for word in words:
          ids[token] = self.dict.voc2i[word]
          token += 1

    return ids

