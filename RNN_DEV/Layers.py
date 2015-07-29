import theano
import theano.tensor as T
import numpy as np
from initializations import glorot_uniform,zero,alloc_zeros_matrix
import theano.typed_list



class hidden(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()

        
        
        self.W_hh=glorot_uniform((n_hidden,n_hidden))
        self.W_in=glorot_uniform((n_in,n_hidden))
        self.bh=zero((n_hidden,))
        
        self.params=[self.W_hh,self.W_in,self.bh]
        
        
        self.L1 = T.sum(abs(self.W_hh))+T.sum(abs(self.W_in))
        self.L2_sqr = T.sum(self.W_hh**2) + T.sum(self.W_in**2)

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
        
        self.L1 = T.sum(abs(self.W_i))+T.sum(abs(self.U_i))+\
                  T.sum(abs(self.W_f))+T.sum(abs(self.U_f))+\
                  T.sum(abs(self.W_c))+T.sum(abs(self.U_c))+\
                  T.sum(abs(self.W_o))+T.sum(abs(self.U_o))
        
        self.L2_sqr = T.sum(self.W_i**2) + T.sum(self.U_i**2)+\
                      T.sum(self.W_f**2) + T.sum(self.U_f**2)+\
                      T.sum(self.W_c**2) + T.sum(self.U_c**2)+\
                      T.sum(self.W_o**2) + T.sum(self.U_o**2)
        
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

        self.L1 = T.sum(abs(self.W_z))+T.sum(abs(self.U_z))+\
                  T.sum(abs(self.W_r))+T.sum(abs(self.U_r))+\
                  T.sum(abs(self.W_h))+T.sum(abs(self.U_h))
        
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)        
        
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




class BiDirectionLSTM(object):
    def __init__(self,n_in,n_hidden,output_mode='concat'):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.output_mode = output_mode
        self.input= T.tensor3()
        
        # forward weights
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
        
        # backward weights
        self.Wb_i = glorot_uniform((n_in,n_hidden))
        self.Ub_i = glorot_uniform((n_hidden,n_hidden))
        self.bb_i = zero((n_hidden,))

        self.Wb_f = glorot_uniform((n_in,n_hidden))
        self.Ub_f = glorot_uniform((n_hidden,n_hidden))
        self.bb_f = zero((n_hidden,))

        self.Wb_c = glorot_uniform((n_in,n_hidden))
        self.Ub_c = glorot_uniform((n_hidden,n_hidden))
        self.bb_c = zero((n_hidden,))
        
        self.Wb_o = glorot_uniform((n_in,n_hidden))
        self.Ub_o = glorot_uniform((n_hidden,n_hidden))
        self.bb_o = zero((n_hidden,))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,

            self.Wb_i, self.Ub_i, self.bb_i,
            self.Wb_c, self.Ub_c, self.bb_c,
            self.Wb_f, self.Ub_f, self.bb_f,
            self.Wb_o, self.Ub_o, self.bb_o,
        ]

        self.L1 = T.sum(abs(self.W_i))+T.sum(abs(self.U_i))+\
                  T.sum(abs(self.W_f))+T.sum(abs(self.U_f))+\
                  T.sum(abs(self.W_c))+T.sum(abs(self.U_c))+\
                  T.sum(abs(self.W_o))+T.sum(abs(self.U_o))+\
                  T.sum(abs(self.Wb_i))+T.sum(abs(self.Ub_i))+\
                  T.sum(abs(self.Wb_f))+T.sum(abs(self.Ub_f))+\
                  T.sum(abs(self.Wb_c))+T.sum(abs(self.Ub_c))+\
                  T.sum(abs(self.Wb_o))+T.sum(abs(self.Ub_o))
        
        self.L2_sqr = T.sum(self.W_i**2) + T.sum(self.U_i**2)+\
                      T.sum(self.W_f**2) + T.sum(self.U_f**2)+\
                      T.sum(self.W_c**2) + T.sum(self.U_c**2)+\
                      T.sum(self.W_o**2) + T.sum(self.U_o**2)+\
                      T.sum(self.Wb_i**2) + T.sum(self.Ub_i**2)+\
                      T.sum(self.Wb_f**2) + T.sum(self.Ub_f**2)+\
                      T.sum(self.Wb_c**2) + T.sum(self.Ub_c**2)+\
                      T.sum(self.Wb_o**2) + T.sum(self.Ub_o**2)

        
        
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
    
        
    def _fstep(self, x_t, h_tm1, c_tm1): 
        i_t = T.tanh(T.dot(x_t, self.W_i) + self.b_i + T.dot(h_tm1, self.U_i))
        f_t = T.tanh(T.dot(x_t, self.W_f) + self.b_f + T.dot(h_tm1, self.U_f))
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_c) + self.b_c + T.dot(h_tm1, self.U_c))
        o_t = T.tanh( T.dot(x_t, self.W_o) + self.b_o + T.dot(h_tm1, self.U_o))
        h_t = o_t * T.tanh(c_t)
        
        return h_t, c_t


    def _bstep(self, x_t, h_tm1, c_tm1): 
        i_t = T.tanh(T.dot(x_t, self.Wb_i) + self.bb_i + T.dot(h_tm1, self.Ub_i))
        f_t = T.tanh(T.dot(x_t, self.Wb_f) + self.bb_f + T.dot(h_tm1, self.Ub_f))
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.Wb_c) + self.bb_c + T.dot(h_tm1, self.Ub_c))
        o_t = T.tanh( T.dot(x_t, self.Wb_o) + self.bb_o + T.dot(h_tm1, self.Ub_o))
        h_t = o_t * T.tanh(c_t)
        
        return h_t, c_t        
       
    def set_input(self,x):
        self.input=x   

        
    def get_input(self):
        if hasattr(self, 'previous'):
            return self.previous.get_output()
        else:
            return self.input    
    
    def get_forward_output(self):
        X=self.get_input()
        [h,c], _ = theano.scan(self._fstep, 
                             sequences = X,
                             outputs_info = [alloc_zeros_matrix(self.n_hidden),
                                             alloc_zeros_matrix(self.n_hidden)])
                            # outputs_info =T.unbroadcast(alloc_zeros_matrix(self.input.shape[0],
                             #                                               self.n_hidden), 0) )

        return h
        
    def get_backward_output(self):
        X=self.get_input()
        [h,c], _ = theano.scan(self._bstep, 
                             sequences = X,
                             outputs_info = [alloc_zeros_matrix(self.n_hidden),
                                             alloc_zeros_matrix(self.n_hidden)],
                                            go_backwards = True)
                            # outputs_info =T.unbroadcast(alloc_zeros_matrix(self.input.shape[0],
                             #                                               self.n_hidden), 0) )

        return h  


    def get_output(self):
        forward = self.get_forward_output()
        backward = self.get_backward_output()
        if self.output_mode is 'sum':
            return forward + backward
        elif self.output_mode is 'concat':
            return T.concatenate([forward, backward], axis=1)
        else:
            raise Exception('output mode is not sum or concat')

class decoder(object):
    def __init__(self,n_in,n_out,time_steps_x,time_steps_y):
        self.n_in=int(n_in)
        self.n_out=int(n_out)
        self.input= T.tensor3()
        self.output= T.tensor3()
        self.time_steps_x=int(time_steps_x)
        self.time_steps_y=int(time_steps_y)
        
        self.W_z = glorot_uniform((n_in,n_out))
        self.U_z = glorot_uniform((n_out,n_out))
        self.b_z = zero((n_out,))

        self.W_r = glorot_uniform((n_in,n_out))
        self.U_r = glorot_uniform((n_out,n_out))
        self.b_r = zero((n_out,))

        self.W_h = glorot_uniform((n_in,n_out)) 
        self.U_h = glorot_uniform((n_out,n_out))
        self.b_h = zero((n_out,))

        
        self.W_hh=glorot_uniform((n_out,n_out))
        self.W_ys=glorot_uniform((self.n_out,n_out))

        
        
        self.W_cy = glorot_uniform((self.n_in,self.n_out))
        self.W_cs= glorot_uniform((self.n_in,self.n_out))
        
        self.W_ha = glorot_uniform((self.n_in,))
        self.W_sa= glorot_uniform((self.n_out,self.time_steps_x))
        
        self.params=[self.W_hh,self.W_ys,self.W_cs,self.W_ha,self.W_sa]
        #self.params=[self.W_hh]
         
        self.L1 = T.sum(abs(self.W_hh))+T.sum(abs(self.W_ys))
        self.L2_sqr = T.sum(self.W_hh**2) + T.sum(self.W_ys**2)

    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
    
        
    def set_input(self,x):
        self.input=x
    
    
        
    def get_input(self):
        if hasattr(self, 'previous'):
            return self.previous.get_output()
        else:
            return self.input    

    def _step(self,y_t,s_tm1,h):
        
        
        e=T.tanh(T.dot(h,self.W_ha)+T.dot(s_tm1,self.W_sa).T)
                
        e=T.exp(e)/T.sum(T.exp(e))

        c=T.dot(e.T,h)

        
        s_t=T.tanh(T.dot(s_tm1, self.W_hh)  +T.dot(y_t,self.W_ys) +T.dot(c,self.W_cs))
        
        return T.cast(s_t,dtype =theano.config.floatX)  

    def get_sample(self,y,h_tm1):
        X=self.get_input()
        Y=T.switch(y[0]<0,alloc_zeros_matrix(self.n_out),self.one_hot(y,self.n_out)[0])

        h=self._step(Y,h_tm1,X)        
        return h

    
    def get_output(self,y):
        X=self.get_input()
        Y=y                    
        Y=self.one_hot(Y,self.n_out)
        
        h, _ = theano.scan(self._step, 
                             sequences = Y,
                             outputs_info = alloc_zeros_matrix(self.n_out),
                             non_sequences=X)
                             #n_steps=self.time_steps_y)

        return h
        
        
    def one_hot(self,t, r=None):
        if r is None:
            r = T.max(t) + 1
            
        ranges = T.shape_padleft(T.arange(r), t.ndim)
        
        return T.cast(T.eq(ranges, T.shape_padright(t, 1)) ,dtype =theano.config.floatX)       