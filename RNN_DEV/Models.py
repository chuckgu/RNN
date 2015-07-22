import theano
import theano.tensor as T
import numpy as np
import logging
import matplotlib.pyplot as plt
import Loss
from initializations import glorot_uniform,zero,alloc_zeros_matrix
from Layers import hidden,lstm,gru,BiDirectionLSTM


mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c

def ndim_tensor(ndim):
    if ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()
    


class RNN(object):
    
    def __init__(self,n_in,n_hidden,n_out,lr=0.001,n_epochs=400,L1_reg=0,L2_reg=0):
        
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_out=int(n_out)
        self.lr=float(lr)

        self.x = T.matrix(name = 'x', dtype = theano.config.floatX)
         
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
        
        self.L1_reg=L1_reg
        self.L2_reg=L2_reg    
        self.L1= 0
        self.L2_sqr= 0
        
        
    def add(self,layer):                                  
        self.layers.append(layer)

        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        else:
            self.layers[0].set_input(self.x)
        
        self.n_layers=self.n_layers+1
        self.params+=layer.params
        self.L1 += layer.L1
        self.L2_sqr += layer.L2_sqr
        
        
    
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
        
    def build(self,output_type):      
        
        #### set up parameter         
        self.params+=[self.W_hy, self.b_hy]
        for param in self.params:
            self.updates[param] = theano.shared(
                                      value = np.zeros(
                                                  param.get_value(
                                                      borrow = True).shape,
                                                      dtype = theano.config.floatX),
                                      name = 'updates')
                                      
        ### set up regularizer                               
        
         
        self.L1 += T.sum(abs(self.W_hy))    
        self.L2_sqr += T.sum(self.W_hy**2)               
                                             
        ### fianl prediction formular
                                             
        self.y_pred = T.dot(self.get_output(), self.W_hy) + self.b_hy
                                     
        self.output_type = output_type
        if self.output_type == 'real':
            self.y = T.matrix(name = 'y', dtype = theano.config.floatX) 
            self.loss = lambda y: Loss.mse(self.y_pred,y) # y is input and self.mse(y) is output
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = self.y_pred,
                                           mode = mode)

        elif self.output_type == 'binary':
            self.y = T.matrix(name = 'y', dtype = 'int32')
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: Loss.nll_binary(self.p_y_given_x,y)
            self.predict_proba = theano.function(inputs = [self.x, ],
                                                 outputs = self.p_y_given_x,
                                                 mode = mode)
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = T.round(self.p_y_given_x),
                                           mode = mode)
        
        elif self.output_type == 'softmax':
            self.y = T.vector(name = 'y', dtype = 'int32')
            self.p_y_given_x = T.nnet.softmax(self.y_pred)
            self.y_out = T.argmax(self.p_y_given_x, axis = -1)
            self.loss = lambda y: Loss.nll_multiclass(self.p_y_given_x,y)
            self.predict_proba = theano.function(inputs = [self.x, ],
                                                 outputs = self.p_y_given_x,
                                                 mode = mode)
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = self.y_out, # y-out is calculated by applying argmax
                                           mode = mode)
        else:
            raise NotImplementedError
        

                                      
   
    def fit(self,X_train,Y_train):
        train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        if self.output_type in ('binary', 'softmax'):
            train_set_y = T.cast(train_set_y, 'int32')
        
        index = T.lscalar('index')    # index to a case    
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum
        

   
        cost = self.loss(self.y) +self.L1_reg * self.L1
        
                       
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
               
                                  
            # compute loss on training set
            train_losses = [compute_train_error(i)
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
        
            self.errors.append(this_train_loss)
           
            print('epoch %i, train loss %f ''lr: %f' % \
                  (epoch, this_train_loss, self.lr))

            self.lr *= self.learning_rate_decay




class ENC_DEC(object):
    
    def __init__(self,n_in,n_hidden,n_out,lr=0.001,n_epochs=400,L1_reg=0,L2_reg=0):
        
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_out=int(n_out)
        self.lr=float(lr)

        self.x = T.matrix(name = 'x', dtype = theano.config.floatX)
         
        self.layers = []
        self.decoder=[]
        self.params=[]
        self.errors=[]
        
        self.n_epochs=n_epochs
        
        

        self.initial_momentum=0.5
        self.final_momentum=0.9
        self.momentum_switchover=5
        self.learning_rate_decay=0.999
        self.updates = {}
        self.n_layers=0
        
        self.L1_reg=L1_reg
        self.L2_reg=L2_reg    
        self.L1= 0
        self.L2_sqr= 0
        
        
        
    def add(self,layer): 
  
        self.layers.append(layer)
    
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        else:
            self.layers[0].set_input(self.x)
        
        self.n_layers=self.n_layers+1    
        self.params+=layer.params
        self.L1 += layer.L1
        self.L2_sqr += layer.L2_sqr
        
        
    
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
        
    def build(self,output_type):      
        

        for param in self.params:
            self.updates[param] = theano.shared(
                                      value = np.zeros(
                                                  param.get_value(
                                                      borrow = True).shape,
                                                      dtype = theano.config.floatX),
                                      name = 'updates')
                                      
            
                                             
        ### fianl prediction formular
                                             
 
        self.y = T.vector(name = 'y', dtype = 'int32')
        
        self.p_y_given_x=self.get_output()
        

        self.y_out = T.argmax(self.p_y_given_x, axis = -1)
        self.loss = lambda y: Loss.nll_multiclass(self.p_y_given_x,y)
        
        self.predict_proba = theano.function(inputs = [self.x,],
                                             outputs = self.p_y_given_x,
                                             mode = mode)
        self.predict = theano.function(inputs = [self.x, ],
                                       outputs = self.y_out, # y-out is calculated by applying argmax
                                       mode = mode)
                                      

                                      
   
    def fit(self,X_train,Y_train):
        train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        train_set_y = T.cast(train_set_y, 'int32')
        
        index = T.lscalar('index')    # index to a case    
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum
        

   
        cost = self.loss(self.y) +self.L1_reg * self.L1
        
                       
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
                T.pprint(self.p_y_given_x)
                                  
            # compute loss on training set
            train_losses = [compute_train_error(i)
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
        
            self.errors.append(this_train_loss)
           
            print('epoch %i, train loss %f ''lr: %f' % \
                  (epoch, this_train_loss, self.lr))

            self.lr *= self.learning_rate_decay


    
  