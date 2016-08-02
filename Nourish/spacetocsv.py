X = open('train.dat','r').read().splitlines()
f = open('input.txt','w')
for i in range(len(X)):
  f.write(X[i].replace(' ',',').rstrip(','))
  f.write('\n')
f.close()
