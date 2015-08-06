import theano
import theano.tensor as T
import numpy as np
import logging
import matplotlib.pyplot as plt
import time
import os
import datetime
import cPickle as pickle
import Loss
from initializations import glorot_uniform,zero,alloc_zeros_matrix
from Layers import hidden,lstm,gru,BiDirectionLSTM

logger = logging.getLogger(__name__)

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
    
    def set_params(self,**params):
        return
    
    def __getstate__(self):
        """ Return state sequence."""
        params = self.params  # parameters set in constructor
        weights = [p.get_value() for p in self.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights = state
        #self.set_params(**params)
        #self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()   
        
    
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
        

                                      
   
    def train(self,X_train,Y_train):
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
    
    def __init__(self,n_in,n_hidden,n_decoder,n_out,
                 time_steps_x,time_steps_y,lr=0.001,n_epochs=400,L1_reg=0,L2_reg=0):
        
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_decoder=int(n_decoder)
        self.n_out=int(n_out)
        self.lr=float(lr)
        self.time_steps_x=int(time_steps_x)
        self.time_steps_y=int(time_steps_y)
        self.maxlen= self.time_steps_y       
        

        self.x = T.matrix(name = 'x', dtype = theano.config.floatX)
        
        self.W_hy = glorot_uniform((self.n_decoder,self.n_out))
        self.b_hy = zero((n_out,))
        
        self.W_hi = glorot_uniform((self.n_hidden,self.n_decoder))
        self.b_hi = zero((n_decoder,))
         
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
    

    def set_params(self,**params):
        return
    
    def __getstate__(self):
        """ Return state sequence."""
        params = self.params  # parameters set in constructor
        weights = [p.get_value() for p in self.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights = state
        #self.set_params(**params)
        #self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()       
        
    
    def get_output(self):
        init_state=T.tanh(T.dot(self.layers[-1].get_input().mean(0), self.W_hi) + self.b_hi)
        return self.layers[-1].get_output(self.y,init_state)
        
    def get_sample(self,y,h):
        return self.layers[-1].get_sample(y,h)    

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
        self.params+=[self.W_hy, self.b_hy,self.W_hi, self.b_hi]
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
                                             
        self.y = T.vector(name = 'y', dtype = 'int32')
        
        self.y_result = T.vector(name = 'y', dtype = 'int32')
               
        self.y_pred = T.dot(self.get_output(), self.W_hy) + self.b_hy
                
        self.p_y_given_x = T.nnet.softmax(self.y_pred)
        
        self.loss = lambda y: Loss.nll_multiclass(self.p_y_given_x,y)
        


        
                                      
    def gen_sample(self,X_test):

        sample=[]
        sample_proba=[]

        
        next_w=-1*T.ones((1,))       
        h_w=T.tanh(T.dot(self.layers[-1].get_input().mean(0), self.W_hi) + self.b_hi)

        for i in xrange(self.maxlen):
            h_t,logit=self.get_sample(next_w,h_w)
            
            
            y_gen = T.dot(logit, self.W_hy) + self.b_hy
            

            
            p_y_given_x_gen = T.nnet.softmax(y_gen)
            
            sample_proba.append(p_y_given_x_gen)
            
            next_w = T.argmax(p_y_given_x_gen, axis = -1)  
            
            sample.append(next_w)        
            
            h_w=h_t
        
        #Todo : implement Beam Search Algorithm here
        
        predict_proba = theano.function(inputs = [self.x,],
                                             outputs = sample_proba,
                                             mode = mode)
                                             
        predict = theano.function(inputs = [self.x,],
                                       outputs = sample, # y-out is calculated by applying argmax
                                       mode = mode)  
                                       
        return  predict_proba(X_test),predict(X_test)                          
   
    def train(self,X_train,Y_train):
        train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        train_set_y_result = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        train_set_y_result = T.cast(train_set_y_result, 'int32')
        
        index = T.lscalar('index')    # index to a case    
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum
        
        ### shift 1 sequence backward
        y_shifted=T.zeros_like(train_set_y_result)
        y_shifted=T.set_subtensor(y_shifted[:,1:],train_set_y_result[:,:-1])
        train_set_y=y_shifted        

        
        cost = self.loss(self.y_result) #+self.L1_reg * self.L1
                       
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
                                              outputs = self.loss(self.y_result),
                                              givens = {
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index],
                                                  self.y_result: train_set_y_result[index]},
                                              mode = mode)    
       
        train_model =theano.function(inputs = [index, lr, mom],
                                      outputs = cost,
                                      updates = updates,
                                      givens = {
                                            self.x: train_set_x[index],
                                            self.y: train_set_y[index],
                                            self.y_result: train_set_y_result[index]},
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


    
  