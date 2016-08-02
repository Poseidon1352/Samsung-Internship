import numpy as np

def cast(string):
    if string == '':
        return np.nan
    else:
        return float(string)

#Data Input
X = open('input.txt', 'r').read().splitlines()
X = [x.split(',') for x in X]
X = [[cast(a) for a in x] for x in X]

# hyperparameters
datum_size = len(X[0])
n_fact = 1 #no of factors
n_nutr = datum_size - n_fact
hidden_size = 100 # size of hidden layer of neurons
seq_length = 5 # number of steps to unroll the RNN for
learning_rate = 1e-2

#Data Preprocessing
X_mean = np.nanmean(X,axis=0)
X_range = np.nanmax(X,axis=0) - np.nanmin(X,axis=0)
X_range[X_range == 0] = 1
X = (X - X_mean)/X_range

# model parameters
Wxh = np.random.randn(hidden_size, datum_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(n_nutr, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size,1)) # hidden bias
by = np.zeros((n_nutr,1)) # output bias


def lossFun(inputs, hprev, n_iter):
  x, h, y = {}, {}, {}
  h[-1] = np.copy(hprev)
  loss = 0
  x[0] = np.copy(np.reshape(inputs[0],(datum_size,1)))
  # forward pass
  for t in range(len(inputs)-1):
    x[t+1] = np.copy(np.reshape(inputs[t+1],(datum_size,1)))
    h[t] = np.tanh(np.dot(Wxh,x[t]) + np.dot(Whh, h[t-1]) + bh) # hidden state
    y[t] = np.dot(Why, h[t]) + by
    if np.isnan(x[t+1][n_fact][0]):
      x[t+1][n_fact:] = np.copy(y[t])
      if n_iter % 100 == 0:
        X_temp = x[t+1].T*X_range + X_mean
        with open("predictions.txt", "ab") as out_file:
          np.savetxt(out_file,X_temp,fmt='%f',delimiter=',')
    loss += 0.5*np.sum(np.square(y[t] - x[t+1][n_fact:])) # linear (sum of squares loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(h[0])
  for t in reversed(range(len(inputs)-1)):
    dy = y[t] - x[t+1][n_fact:]
    dWhy += np.dot(dy, h[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - h[t] * h[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, x[t].T)
    dWhh += np.dot(dhraw, h[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, h[len(inputs)-2]

n_iter, p, olap = 0, 0, 1
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(X) or n_iter == 0: 
    hprev = np.random.normal(0,1,(hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = X#[p:p+seq_length+1]
  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, hprev,n_iter)
  if n_iter % 100 == 0:
    print('iter'+str(n_iter)+' loss:'+str(loss)) # print progress
    with open("predictions.txt", "a") as out_file:
      out_file.write('iter'+str(n_iter)+'\n')
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += int(seq_length*olap) # move data pointer
  n_iter += 1 # iteration counter
