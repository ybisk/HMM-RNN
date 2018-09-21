import os, sys

prefix = sys.argv[1]
no_log = "--no-log" in sys.argv[2:]
bpc = "--bpc" in sys.argv[2:]

results = []

for root, dirs, files in os.walk("."):
  for fname in files:
    if fname.startswith(prefix) and (".log" in fname or no_log):
      valid = 1e10
      train = 1e10
      try:
        for line in open(fname,'r'):
          line = line.strip().split()
          if "ppl" in line:
            if bpc:
              ind = line.index("bpc") + 1
            else:
              ind = line.index("ppl") + 1
            if "valid" in line:
              if line[ind] != "nan":
                valid = min(valid, float(line[ind]))
            elif "batches" in line:
              if line[ind] != "nan":
                train = min(train, float(line[ind]))
        results.append((valid, train, fname))
      except FileNotFoundError:
        continue

results.sort()
for valid, train, fname in results:
  fname = fname.replace("l35_","")
  #fname = fname.replace(prefix + "_", "")
  fname = fname.replace(".log","")
  fname = fname.split("_")
  fname = " ".join(["{:8s}".format(v) for v in fname])

  print("{:50} {:9.3f}   {:9.3f}".format(fname, train, valid))

