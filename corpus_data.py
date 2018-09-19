# Source: https://github.com/pytorch/examples/blob/master/word_language_model/data.py

import os
import torch
import gzip

class Dictionary(object):
  def __init__(self):
    self.voc2i = {}
    self.i2voc = []
    self.tagDict = {0:-1}

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
    self.train, _ = self.tokenize(os.path.join(path, 'train.txt.gz'))
    self.build_tag_dict(os.path.join(path, '../ptb-tagged/train.tagged.gz'))

    self.valid, tokens = self.tokenize(os.path.join(path, 'valid.txt.gz'))
    self.valid_tags = self.tags(os.path.join(path, '../ptb-tagged/valid.aligned.txt.gz'), tokens)
    self.test, tokens = self.tokenize(os.path.join(path, 'test.txt.gz'))
    self.test_tags = self.tags(os.path.join(path, '../ptb-tagged/test.aligned.txt.gz'), tokens)


  def build_tag_dict(self, path):
    assert os.path.exists(path)
    tagset = set()
    D = {}
    for line in gzip.open(path, 'r'):
      for w in line.lower().strip().split():
        w = w.rsplit("/".encode(),1)
        tag = w[1].split("|".encode())[0]
        word = w[0]
        if tag == "CD".encode() or word.decode('utf-8').isnumeric():
          word = "N".encode()
        if word not in self.dict.voc2i:
          word = "<unk>".encode()

        if word not in D:
          D[word] = {}
        if tag not in D[word]:
          D[word][tag] = 0
        D[word][tag] += 1
        tagset.add(tag)

    # Mapping
    self.dict.i2tag = list(tagset)
    print(self.dict.i2tag)
    self.dict.tag2i = {}
    for i in range(len(self.dict.i2tag)):
      self.dict.tag2i[self.dict.i2tag[i]] = i

    # Top tag per word in training
    for w in D:
      v = [(D[w][t],t) for t in D[w]]
      v.sort()
      self.dict.tagDict[self.dict.voc2i[w]] = self.dict.tag2i[v[-1][1]]

  def tags(self, path, tokens):
    assert os.path.exists(path)
    ids = torch.LongTensor(tokens)
    ids[0] = self.dict.voc2i['<eos>']
    token = 1
    for line in gzip.open(path, 'r'):
      for word in line.strip().split():
        tag = word.lower().rsplit("/".encode(),1)[1].split("|".encode())[0]
        ids[token] = self.dict.tag2i[tag]
        token += 1
      ids[token] = -1
      token += 1
    return ids

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

    return ids, tokens

