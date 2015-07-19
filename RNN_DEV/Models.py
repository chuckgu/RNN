import theano
import theano.tensor as T
import numpy as np
import logging
import matplotlib.pyplot as plt
from initializations import glorot_uniform,zero,alloc_zeros_matrix
from Layers import hidden,lstm,gru
from Loss import mean_squared_error

mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c

def ndim_tensor(ndim):
    if ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()
    


class Model(object):
    
    def __init__(self,n_in,n_hidden,n_out,lr=0.001,n_epochs=400):
        
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_out=int(n_out)
        self.lr=float(lr)
        self.L1_reg=0
        self.L2_reg=0
        self.x = T.matrix(name = 'x', dtype = theano.config.floatX)
        self.y = T.matrix(name = 'y', dtype = theano.config.floatX) 
         
        self.layers = []
        self.params=[]
        self.errors=[]
        
        self.n_epochs=n_epochs
        self.W_hy = glorot_uniform((self.n_hidden,self.n_out))
        self.b_hy = zero((self.n_out,))
        

        self.initial_momentum=0.5
        self.final_momentum=0.9
        self.momentum_switchover=5
        self.learning_rate_decay=0.999
        self.updates = {}
        self.n_layers=0
        
        self.L1= T.scalar('L1', dtype = theano.config.floatX)
        #self.L2_sqr= T.scalar('L2_sqr', dtype = theano.config.floatX)
        
        
    def add(self,layer):                                  
        self.layers.append(layer)

        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        else:
            self.layers[0].set_input(self.x)
        
        self.n_layers=self.n_layers+1
        self.params+=layer.params
        #self.L1+=layer.L1
        #self.L2_sqr+=layer.L2_sqr
        
        
    
    def get_output(self):
        return self.layers[-1].get_output()

    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = l.input.ndim
                self.layers[0].input = ndim_tensor(ndim)
                break

    def get_input(self, train=False):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input()     
        
    def build(self):      
        
        #### set up parameter         
        self.params+=[self.W_hy, self.b_hy]
        for param in self.params:
            self.updates[param] = theano.shared(
                                      value = np.zeros(
                                                  param.get_value(
                                                      borrow = True).shape,
                                                      dtype = theano.config.floatX),
                                      name = 'updates')
                                             
        ### fianl prediction formular
        self.y_pred = T.dot(self.get_output(), self.W_hy) + self.b_hy   
        
        for layer in self.layers:        
            self.L1 += layer.L1
            
        self.L1 += abs(self.W_hy.sum())
    
    def predict(self,input):
        
        predict = theano.function(inputs = [self.x, ],
                                           outputs = self.y_pred,
                                           mode = mode, allow_input_downcast=True)
        self.y_pred=predict(input)                                   
       
        return self.y_pred
    
    def loss(self,y):
        
        loss=mean_squared_error(y,self.y_pred)
               
        return loss
    
    def fit(self,X_train,Y_train):
        train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        
        
        index = T.lscalar('index')    # index to a case    
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum
   
        cost = self.loss(self.y)+self.L1_reg * self.L1
                       
        gparams = []
        for param in self.params:
            gparams.append(T.grad(cost, param))

        # zip just concatenate two lists
        updates = {}
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom*weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd
            
        compute_train_error = theano.function(inputs = [index, ],
                                              outputs = self.loss(self.y),
                                              givens = {
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]},
                                              mode = mode)    
       
        train_model =theano.function(inputs = [index, lr, mom],
                                      outputs = cost,
                                      updates = updates,
                                      givens = {
                                          self.x: train_set_x[index], # [:, batch_start:batch_stop]
                                          self.y: train_set_y[index]},
                                      mode = mode, allow_input_downcast=True)
                     
        ###############
        # TRAIN MODEL #
        ###############
        print 'Training model ...'
        epoch = 0
        n_train = train_set_x.get_value(borrow = True).shape[0]

        while (epoch < self.n_epochs):
            epoch = epoch + 1
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                                     if epoch > self.momentum_switchover \
                                     else self.initial_momentum
                example_cost = train_model(idx,
                                           self.lr,
                                           effective_momentum)
                #this_train_loss = np.mean(example_cost)   
                
                                  
            # compute loss on training set
            train_losses = [compute_train_error(i)
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
        
            self.errors.append(this_train_loss)
           
            print('epoch %i, train loss %f ''lr: %f' % \
                  (epoch, this_train_loss, self.lr))

            self.lr *= self.learning_rate_decay

    
    

if __name__ == "__main__":

    
    print 'Testing model with real outputs'
    n_u = 3 # input vector size (not time at this point)
    n_h = 10 # hidden vector size
    n_y = 3 # output vector size
    time_steps = 15 # number of time-steps in time
    n_seq = 100 # number of sequences for training

    np.random.seed(0)
    
    # generating random sequences
    seq = np.random.randn(n_seq, time_steps, n_u)
    seq=np.cast[theano.config.floatX](seq)
    targets = np.zeros((n_seq, time_steps, n_y))

    targets[:, 1:, 0] = seq[:, :-1, 0] # 1 time-step delay between input and output
    targets[:, 2:, 1] = seq[:, :-2, 1] # 2 time-step delay
    targets[:, 3:, 2] = seq[:, :-3, 2] # 3 time-step delay

    targets += 0.01 * np.random.standard_normal(targets.shape)
    

    
    model = Model(n_u,n_h,n_y,0.001,200)
    model.add(hidden(n_u,n_h))
    #model.add(lstm(n_h,n_h))
    
    
    model.build()
    model.fit(seq,targets)
    
    
    # We just plot one of the sequences
    plt.close('all')
    fig = plt.figure()

    # Graph 1
    ax1 = plt.subplot(311) # numrows, numcols, fignum
    plt.plot(seq[0])
    plt.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = plt.subplot(312)
    true_targets = plt.plot(targets[0])

    guess = model.predict(seq[0])
    guessed_targets = plt.plot(guess, linestyle='--')
    plt.grid()
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    # Graph 3
    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')

    # Save as a file
    #plt.savefig('real.png')

    
  