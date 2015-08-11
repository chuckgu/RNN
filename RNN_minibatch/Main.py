import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from Layers import hidden,lstm,gru,BiDirectionLSTM,decoder
from Models import ENC_DEC

print 'Testing model with softmax outputs'

#theano.config.exception_verbosity='high'

theano.config.optimizer='None' 

n_u = 10
n_h = 20
time_steps_x = 10

n_d = 30
n_y = 3 
time_steps_y = 10

n_seq= 105
n_epochs = 100

n_batch=15
n_maxlen=10

np.random.seed(0)

seq = np.random.randn(n_seq, time_steps_x, n_u)
seq=np.cast[theano.config.floatX](seq)
# Note that is this case `targets` is a 2d array

targets = np.zeros((n_seq, time_steps_y), dtype=np.int)

thresh = 0.5
# Comparisons to assing a class label in output

targets[:, 0][seq[:, 0, 1] > seq[:, 0, 0] + thresh] = 1
targets[:, 0][seq[:, 0, 1] < seq[:, 0, 0] - thresh] = 2
targets[:, -1][seq[:, -1, 1] > seq[:, -2, 0] + thresh] = 1
targets[:, -1][seq[:, -1, 1] < seq[:, -2, 0] - thresh] = 2
targets[:, 1:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + thresh] = 1
targets[:, 1:][seq[:, 1:-1, 1] < seq[:, :-2, 0] - thresh] = 2
# otherwise class is 0

targets_onehot=np.zeros((n_seq, time_steps_y,n_y), dtype=np.int)

targets_onehot[:,:,0][targets[:,:]==0]=1
targets_onehot[:,:,1][targets[:,:]==1]=1
targets_onehot[:,:,2][targets[:,:]==2]=1

targets_onehot=np.cast[theano.config.floatX](targets_onehot)


mode='tr'

seq=seq.transpose(1,0,2)
targets_onehot=targets_onehot.transpose(1,0,2)

model = ENC_DEC(n_u,n_h*2,n_d,n_y,0.001,n_epochs,n_batch,n_maxlen)
model.add(BiDirectionLSTM(n_u,n_h))
model.add(decoder(n_h*2,n_d,n_y))

model.build('softmax')



if mode=='tr':
    model.train(seq,targets_onehot)
    model.save('encdec_new.pkl')
else:model.load('encdec_new.pkl')

i=20
plt.close('all')
fig = plt.figure()
ax1 = plt.subplot(311)
plt.plot(seq[:,i])
plt.grid()
ax1.set_title('input')
ax2 = plt.subplot(312)

plt.scatter(xrange(time_steps_y), targets[i], marker = 'o', c = 'b')
plt.grid()

a=seq[:,i]
b=seq[:, i][:, np.newaxis]
guess = model.gen_sample(b)

aa=guess[1]

guess=np.asarray(guess[0],dtype=np.float).reshape((10,3))




guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
ax2.set_title('blue points: true class, grayscale: model output (white mean class)')

ax3 = plt.subplot(313)
plt.plot(model.errors)
plt.grid()
ax3.set_title('Training error')


'''

model = RNN(n_u,n_h,n_y,0.001,100)
model.add(gru(n_u,n_h))

model.build('softmax')

model.train(seq,targets)

i=1
plt.close('all')
fig = plt.figure()
ax1 = plt.subplot(311)
plt.plot(seq[i])
plt.grid()
ax1.set_title('input')
ax2 = plt.subplot(312)

plt.scatter(xrange(time_steps_y), targets[i], marker = 'o', c = 'b')
plt.grid()


guess = model.predict_proba(seq[i])

#guess = model.gen_sample(seq[i])

#guess=np.asarray(guess,dtype=np.float)

guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
ax2.set_title('blue points: true class, grayscale: model output (white mean class)')

ax3 = plt.subplot(313)
plt.plot(model.errors)
plt.grid()
ax3.set_title('Training error')

'''