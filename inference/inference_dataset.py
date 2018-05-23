
''' Some code to handle datasets
'''

import numpy as np
import random

def get_dataset(positive, negative, abstract_length, concrete_length, pad, sep):
  ''' takes two seqs of 6-tuples; returns a triple of
      (concrete-idx-seqs, abstract-idx-seqs, target-labels), each of which is an
      np.array (the elements of which stand in 1-1 correspondence, but which are
      permuted in order w.r.t. the input arguments).
  '''
  abstracts = np.concatenate((
      _get_array_from_individual_seqs([t[0] for t in positive],
                                     abstract_length,
                                     pad),
      _get_array_from_individual_seqs([t[0] for t in negative],
                                     abstract_length,
                                     pad)))
  # smash the concrete sentences together with periods, stack them up:
  concretes = np.concatenate(
      (_get_array_from_individual_seqs(
          [t[1] + [sep] + t[2] + [sep] + t[3] + [sep] + t[4] + [sep] + t[5]
           for t in positive],
          concrete_length,
          pad),
      _get_array_from_individual_seqs(
          [t[1] + [sep] + t[2] + [sep] + t[3] + [sep] + t[4] + [sep] + t[5]
           for t in negative],
          concrete_length,
          pad)))
  # needs to be int64 because F.cross_entropy expects a LongTensor:
  labels = np.array(([1] * len(positive)) + ([0] * len(negative)),
                    dtype=np.int64)
  idxs = np.random.permutation(range(len(labels)))
  return (concretes[idxs], abstracts[idxs], labels[idxs])

def get_negative_examples(data_tuples, negative_ratio):
  ''' returns a list of (youcooktxt, s1, s2, ...) 6-tuples, just like the input,
  but with sampled "imposter" youcooktxt first elements.
  '''
  dlen = len(data_tuples)
  num_negs = int(dlen * negative_ratio / (1 - negative_ratio))
  if num_negs <= dlen:
    idxs = random.sample(range(dlen), num_negs)
  else:
    idxs = random.sample(range(dlen), dlen)
    while len(idxs) < num_negs:
      idxs.extend(random.sample(range(dlen),
                                min(dlen, num_negs - len(idxs))))
  return [tuple([head] + list(data_tuples[idx][1:]))
          for idx,head in zip(idxs, _get_imposter_heads(idxs, data_tuples))]


def _get_array_from_individual_seqs(seqs, targ_len, pad):
  ''' returns an np.array of the elems of seq )padded/trimmed and stacked up.
  '''
  # needs to be int64 so it turns into a LongTensor, as nn.Embedding expects.
  return np.array([_get_seq_of_len(seq, targ_len, pad) for seq in seqs],
                  dtype=np.int64)

def _get_seq_of_len(seq, targ_len, pad):
  seq = seq[:targ_len]
  return seq + [pad for _ in range(targ_len - len(seq))]

def _get_imposter_heads(idxs, data_tuples):
  deranged_idxs = _get_random_derangement(idxs, data_tuples)
  assert len(deranged_idxs) == len(idxs)
  return [data_tuples[idx][0] for idx in deranged_idxs]

def _get_random_derangement(idxs, data_tuples):
  ''' returns a random permutation such that no index is mapped to something
  with the same first element in data_tuples (this would mean they have the same
  concrete annotation---since the point is to sample negative concrete
  representations, we can't be sampling identical representations).
  '''
  # TODO there is, like, for sure a better way to do this---the prob of finding
  # a derangement approaches 1/e, but this extra constraint damps...something by
  # a constant factor.
  idxs = np.array(idxs, dtype=np.int32)
  perm = np.random.permutation(idxs)
  while any(data_tuples[i][0] == data_tuples[x][0] for i,x in zip(idxs,perm)):
    perm = np.random.permutation(idxs)
  return perm
