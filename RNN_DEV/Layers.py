import theano
import theano.tensor as T
import numpy as np
from initializations import glorot_uniform,zero,alloc_zeros_matrix



class hidden(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        
        
        self.W_hh=glorot_uniform((n_hidden,n_hidden))
        self.W_in=glorot_uniform((n_in,n_hidden))
        self.bh=zero((n_hidden,))
        
        self.params=[self.W_hh,self.W_in,self.bh]
        
        self.L1 = abs(self.W_hh.sum())+abs(self.W_in.sum())

    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
    
    def _step(self,x_t, h_tm1):
        return T.tanh(T.dot(h_tm1, self.W_hh) + T.dot(x_t, self.W_in) + self.bh)
        
    def set_input(self,x):
        self.input=x
        
    def get_input(self):
        if hasattr(self, 'previous'):
            return self.previous.get_output()
        else:
            return self.input    
    
    def get_output(self):
        X=self.get_input()
        h, _ = theano.scan(self._step, 
                             sequences = X,
                             outputs_info = alloc_zeros_matrix(self.n_hidden))
                            # outputs_info =T.unbroadcast(alloc_zeros_matrix(self.input.shape[0],
                             #                                               self.n_hidden), 0) )
                            # outputs_info = [T.unbroadcast(T.alloc(self.h0_tm1,
                             #self.n_layers,self.n_hidden),0),])
        return h


class lstm(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        
        
        self.W_i = glorot_uniform((n_in,n_hidden))
        self.U_i = glorot_uniform((n_hidden,n_hidden))
        self.b_i = zero((n_hidden,))

        self.W_f = glorot_uniform((n_in,n_hidden))
        self.U_f = glorot_uniform((n_hidden,n_hidden))
        self.b_f = zero((n_hidden,))

        self.W_c = glorot_uniform((n_in,n_hidden))
        self.U_c = glorot_uniform((n_hidden,n_hidden))
        self.b_c = zero((n_hidden,))

        self.W_o = glorot_uniform((n_in,n_hidden))
        self.U_o = glorot_uniform((n_hidden,n_hidden))
        self.b_o = zero((n_hidden,))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]
        
        
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
    
        
    def _step(self, x_t, h_tm1, c_tm1): 
        i_t = T.tanh(T.dot(x_t, self.W_i) + self.b_i + T.dot(h_tm1, self.U_i))
        f_t = T.tanh(T.dot(x_t, self.W_f) + self.b_f + T.dot(h_tm1, self.U_f))
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_c) + self.b_c + T.dot(h_tm1, self.U_c))
        o_t = T.tanh( T.dot(x_t, self.W_o) + self.b_o + T.dot(h_tm1, self.U_o))
        h_t = o_t * T.tanh(c_t)
        
        return h_t, c_t
       
    def set_input(self,x):
        self.input=x   

        
    def get_input(self):
        if hasattr(self, 'previous'):
            return self.previous.get_output()
        else:
            return self.input    
    
    def get_output(self):
        X=self.get_input()
        [h,c], _ = theano.scan(self._step, 
                             sequences = X,
                             outputs_info = [alloc_zeros_matrix(self.n_hidden),
                                             alloc_zeros_matrix(self.n_hidden)])
                            # outputs_info =T.unbroadcast(alloc_zeros_matrix(self.input.shape[0],
                             #                                               self.n_hidden), 0) )

        return h

class gru(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        
        
        self.W_z = glorot_uniform((n_in,n_hidden))
        self.U_z = glorot_uniform((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_in,n_hidden))
        self.U_r = glorot_uniform((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.U_h = glorot_uniform((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]
        
        
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
    
    def _step(self,x_t, h_tm1):
        
        z = T.tanh(T.dot(x_t, self.W_z) + self.b_z + T.dot(h_tm1, self.U_z))
        r = T.tanh(T.dot(x_t, self.W_r) + self.b_r + T.dot(h_tm1, self.U_r))
        hh_t = T.tanh(T.dot(x_t, self.W_h) + self.b_h + T.dot(r * h_tm1, self.U_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        
        return h_t
        
    def set_input(self,x):
        self.input=x
        
    def get_input(self):
        if hasattr(self, 'previous'):
            return self.previous.get_output()
        else:
            return self.input    
    
    def get_output(self):
        X=self.get_input()
        h, _ = theano.scan(self._step, 
                             sequences = X,
                             outputs_info = alloc_zeros_matrix(self.n_hidden))
                            # outputs_info =T.unbroadcast(alloc_zeros_matrix(self.input.shape[0],
                             #                                               self.n_hidden), 0) )
                            # outputs_info = [T.unbroadcast(T.alloc(self.h0_tm1,
                             #self.n_layers,self.n_hidden),0),])
        return h

