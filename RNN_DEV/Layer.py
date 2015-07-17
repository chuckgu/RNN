import theano
import theano.tensor as T
import numpy as np
import logging
import matplotlib.pyplot as plt
from initializations import glorot_uniform,zero,alloc_zeros_matrix

mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c

class normal_hidden(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.x= T.tensor3()
        
        
        self.W_hh=glorot_uniform((n_hidden,n_hidden))
        self.W_in=glorot_uniform((n_in,n_hidden))
        self.bh=zero((n_hidden,))
        
        self.params=[self.W_hh,self.W_in,self.bh]
        
    def set_previous(self,layer):
        self.previous = layer
        self.x=layer.get_out()
    
    def _step(self,x_t, h_tm1):
        return T.tanh(T.dot(h_tm1, self.W_hh) + T.dot(x_t, self.W_in) + self.bh)
        
    
    def get_output(self):
        h, _ = theano.scan(self._step, 
                             sequences = self.x,
                             outputs_info =T.unbroadcast(alloc_zeros_matrix(self.x.shape[0], self.n_hidden), 0) )
                            # outputs_info = [T.unbroadcast(T.alloc(self.h0_tm1,
                             #self.n_layers,self.n_hidden),0),])
        return h

class Model(object):
    def __init__(self,n_in,n_hidden,n_out,lr,n_epochs):
        
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_out=int(n_out)
        self.lr=float(lr)
        self.x = T.matrix(name = 'x', dtype = theano.config.floatX)
        self.y = T.matrix(name = 'y', dtype = theano.config.floatX) 
         
        #self.hidden=normal_hidden(n_in,n_hidden)
        self.hidden = []
        
        self.n_epochs=n_epochs
        self.W_hy = glorot_uniform((self.n_hidden,self.n_out))
        self.b_hy = zero((self.n_out,))
        
        
        
        self.params=[]
        self.errors=[]
        self.initial_momentum=0.5
        self.final_momentum=0.9
        self.momentum_switchover=5
        self.learning_rate_decay=0.999
        self.updates = {}
        self.n_layers=0
        
    def add(self,layer):                                  
        self.hidden.append(layer)

        if len(self.hidden) > 1:
            self.hidden[-1].set_previous(self.hidden[-2])

        
        self.params.extend(layer.params)
        self.n_layers=self.n_layers+1
        #self.regularizers += regularizers
        #self.constraints += constraints
        
        
    def build(self):
        #self.h0_tm1 = zero((self.n_layers*self.n_hidden))
         
            
                                                     
        self.params.extend([self.W_hy, self.b_hy])
        for param in self.params:
            self.updates[param] = theano.shared(
                                      value = np.zeros(
                                                  param.get_value(
                                                      borrow = True).shape,
                                                      dtype = theano.config.floatX),
                                      name = 'updates')
                           
    
    def fit(self,X_train,Y_train):
        train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        '''
        def step(x_t, h_tm1):
            num=0
            h_tm1_c=[]
            for layer in self.hidden:     
                h_t=layer.activation(x_t,h_tm1[num])      
                x_t=h_t
                h_tm1_c.append(h_t)
                num=num+1
            return h_tm1_c
        '''    
        
        index = T.lscalar('index')    # index to a case    
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum
        
    

        self.y_pred = T.dot(self.hidden[-1].get_output(), self.W_hy) + self.b_hy
        
        self.loss = lambda y: T.mean((self.y_pred - y) ** 2)
        cost = self.loss(self.y)
        
        self.predict = theano.function(inputs = [self.x, ],
                                           outputs = self.y_pred,
                                           mode = mode, allow_input_downcast=True)
        '''                                
        compute_train_error = theano.function(inputs = [index, ],
                                              outputs = self.loss(self.y),
                                              givens = {
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]},
                                              mode = mode)
        '''       
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
                this_train_loss = np.mean(example_cost)   
                
            '''                        
            # compute loss on training set
            train_losses = [compute_train_error(i)
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
            '''
            self.errors.append(this_train_loss)
           
            print('epoch %i, train loss %f ''lr: %f' % \
                  (epoch, this_train_loss, self.lr))

            self.lr *= self.learning_rate_decay

    
    

if __name__ == "__main__":

    
    n_u = 3
    n_h = 10
    n_y = 3
    time_steps = 20
    n_seq= 100
    n_epochs = 500
    
    print 'Testing model with real outputs'
    np.random.seed(0)
    
    # generating random sequences
    seq = np.random.randn(n_seq, time_steps, n_u)
    seq=np.cast[theano.config.floatX](seq)
    targets = np.zeros((n_seq, time_steps, n_y))

    targets[:, 1:, 0] = seq[:, :-1, 0] # 1 time-step delay between input and output
    targets[:, 4:, 1] = seq[:, :-4, 1] # 2 time-step delay
    targets[:, 8:, 2] = seq[:, :-8, 2] # 3 time-step delay

    targets += 0.01 * np.random.standard_normal(targets.shape)
        
    
    
    model = Model(n_u,n_h,n_y,0.001,n_epochs)
    model.add(normal_hidden(n_u,n_h))
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
    ax1.set_title('Training error')

    # Save as a file
    plt.savefig('real_' + str(model.activation) + '_Epoch: ' + str(n_epochs) + '.png')
    
  