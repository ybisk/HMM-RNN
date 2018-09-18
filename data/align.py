import gzip, sys, os

PTB = [line.replace(" U.S/"," U.S./").strip().split() for line in gzip.open(sys.argv[1],'rt')]
v = []
for val in PTB:
  v.extend(val)
PTB = v
Mik = [line.strip().split() for line in gzip.open(sys.argv[2],'rt')]

def equal(m, p):
  if m == "<unk>":
    return True
  p = p.rsplit("/",1)
  if m == "N" and (p[1] == "CD" or p[0] == "%" or p[1] == "LS"):
    return True
  p = p[0].lower()
  return p == m
  

idx = 0
out = open("out.txt",'w')
for i in range(len(Mik)):
  m = Mik[i]
  if equal(m[0], PTB[idx]) and equal(m[-1], PTB[idx + len(m) - 1]):
    out.write(" ".join(PTB[idx:idx+len(m)]) + "\n")
    idx += len(m)
  else:
    print(m)
    print(PTB[idx:idx+len(m)])
    print(m[0], PTB[idx], m[-1], PTB[idx + len(m) - 1])
    print(i)
    sys.exit()
