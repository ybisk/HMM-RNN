import os, sys

prefix = "elman"

results = []

for root, dirs, files in os.walk("."):
  for fname in files:
    if prefix in fname and ".log" in fname:
      valid = 1e10
      train = 1e10
      for line in open(fname,'r'):
        line = line.strip().split()
        if "valid" in line:
          valid = min(valid, float(line[-1]))
        if "batches" in line:
          train = min(train, float(line[-1]))

      results.append((valid, train, fname))

results.sort()
for valid, train, fname in results:
  print("{:60} {:7.3f}   {:7.3f}".format(fname, train, valid))

