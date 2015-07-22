import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from Layers import hidden,lstm,gru,BiDirectionLSTM,decoder
from Models import RNN,ENC_DEC

print 'Testing model with softmax outputs'

#theano.config.exception_verbosity='high'

n_u = 2
n_h = 6
n_y = 3
time_steps = 10
n_seq= 100
n_epochs = 1000

np.random.seed(0)

seq = np.random.randn(n_seq, time_steps, n_u)
seq=np.cast[theano.config.floatX](seq)
# Note that is this case `targets` is a 2d array
targets = np.zeros((n_seq, time_steps), dtype=np.int)

thresh = 0.5
# Comparisons to assing a class label in output
targets[:, 2:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + thresh] = 1
targets[:, 2:][seq[:, 1:-1, 1] < seq[:, :-2, 0] - thresh] = 2
# otherwise class is 0

targets_onehot=np.zeros((n_seq, time_steps,n_y), dtype=np.int)

targets_onehot[:,:,0][targets[:,:]==0]=1
targets_onehot[:,:,1][targets[:,:]==1]=1
targets_onehot[:,:,2][targets[:,:]==2]=1


model = ENC_DEC(n_u,n_h,n_y,0.001,200)
model.add(BiDirectionLSTM(n_u,n_h))
model.add(decoder(n_h,n_y))
'''
model = RNN(n_u,n_h,n_y,0.001,200)
model.add(hidden(n_u,n_h))
model.add(hidden(n_h,n_h))
'''

model.build('softmax')
model.fit(seq,targets)


plt.close('all')
fig = plt.figure()
ax1 = plt.subplot(311)
plt.plot(seq[1])
plt.grid()
ax1.set_title('input')
ax2 = plt.subplot(312)

plt.scatter(xrange(time_steps), targets[1], marker = 'o', c = 'b')
plt.grid()

guess = model.predict_proba(seq[1])
guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
ax2.set_title('blue points: true class, grayscale: model output (white mean class)')

ax3 = plt.subplot(313)
plt.plot(model.errors)
plt.grid()
ax3.set_title('Training error')