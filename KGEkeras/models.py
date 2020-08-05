### KG embedding version 3.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, Lambda, Multiply, Reshape, Concatenate, BatchNormalization, Conv2D, Activation, Dense, Dropout, Conv3D, Flatten
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def l3_reg(weight_matrix, w = 0.01):
    return w * tf.norm(weight_matrix,ord=3)**3

class EmbeddingModel(tf.keras.Model):
    def __init__(self, 
                 e_dim, 
                 r_dim, 
                 num_entities, 
                 num_relations, 
                 negative_samples=2, 
                 batch_size=16,
                 loss_function = 'pointwize_hinge',
                 use_bn = True, 
                 dp = 0.2,
                 margin = 1,
                 loss_weight=1,
                 regularization = 0.01,
                 use_batch_norm = True,
                 entity_embedding_args = None,
                 relational_embedding_args = None,
                 name='embedding_model',
                 **kwargs):
        """
        Base class for embedding models. 
        
        Parameters
        ----------
        e_dim : int 
            Entity embedding dimension
            
        r_dim : int 
            Relation embedding dimension
        
        num_entities : int 
        
        num_relations : int
        
        negative_samples : int 
            Number of negative triples per BATCH.
            
        loss_function : string
            hinge, logistic, or square
        
        loss_type : string 
            pointwize or pairwize
        
        use_bn : bool 
            Batch norm. 
            
        use_dp : bool 
            Use dropout.
        """
        super(EmbeddingModel, self).__init__()
        self.regularization = regularization
        if regularization != 0.0:
            reg = lambda x: l3_reg(x,regularization)
        else:
            reg = None
            
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn_e = BatchNormalization()
            self.bn_r = BatchNormalization()
            
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        init_e = tf.keras.initializers.GlorotUniform()
        init_r = tf.keras.initializers.GlorotUniform()
        self.entity_embedding = Embedding(num_entities,
                                          e_dim,
                                          embeddings_initializer=init_e, 
                                          embeddings_regularizer=reg, 
                                          name=name+'_entity_embedding')
        self.relational_embedding = Embedding(num_relations,
                                              r_dim,
                                              embeddings_initializer=init_r, 
                                              embeddings_regularizer=reg, 
                                              name=name+'_relational_embedding')
        
        self.dp = Dropout(dp)
        self.e_dim = e_dim
        self.r_dim = r_dim
        self.margin = margin
        self.loss_weight = loss_weight
        self.regularization = regularization
        self.use_batch_norm = use_batch_norm
       
        self.__dict__.update(kwargs)
    
    def get_config(self):
        return self.__dict__
    
    def call(self,inputs,training=False):
        """
        Parameters
        ----------
        inputs : tensor, shape = (batch_size, 3)
        """
        
        s,p,o = tf.unstack(inputs,axis=1)
        
        def lookup_entity(a):
            if self.use_batch_norm:
                return self.dp(self.bn_e(self.entity_embedding(a)))
            else:
                return self.dp(self.entity_embedding(a))
            
        def lookup_relation(a):
            if self.use_batch_norm:
                return self.dp(self.bn_r(self.relational_embedding(a)))
            else:
                return self.dp(self.relational_embedding(a))
            
        s,p,o = lookup_entity(s),lookup_relation(p),lookup_entity(o)
        score = self.func(s,p,o,training)
        
        return score
    
class DistMult(EmbeddingModel):
    def __init__(self,
                 name='DistMult', 
                 **kwargs):
        """DistMult implmentation."""
        super(DistMult, self).__init__(name=name,**kwargs)
        
    def func(self, s,p,o, training = False):
        return tf.reduce_sum(s*p*o, axis=-1)
        

class TransE(EmbeddingModel):
    def __init__(self, 
                 name='TransE',
                 norm=1,
                 gamma=12,
                 **kwargs):
        """TransE implmentation."""
        super(TransE, self).__init__(name=name,**kwargs)
        self.gamma = gamma
        self.norm = norm
        
    def func(self, s,p,o, training = False):
        if self.gamma > 0:
            return self.gamma - tf.norm(s+p-o, axis=1, ord=self.norm)
        else:
            return tf.norm(s+p-o, axis=1, ord=self.norm)
    
class CosinE(EmbeddingModel):
    def __init__(self,
                 name='CosinE',
                 **kwargs):
        """CosinE implmentation."""
        super(CosinE, self).__init__(name=name,**kwargs)
        
    def func(self, s,p,o, training = False):
        return -(1+2*cosine_similarity(s+p,o,axis=1))
        

class ComplEx(EmbeddingModel):
    def __init__(self,
                 name='ComplEx', 
                 **kwargs):
        """ComplEx implmentation."""
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 2*kwargs['r_dim']
        super(ComplEx, self).__init__(name=name,**kwargs)
        
    def func(self, s,p,o, training = False):
        split2 = lambda x: tf.split(x,num_or_size_splits=2,axis=-1)
        s_real, s_img = split2(s)
        p_real, p_img = split2(p)
        o_real, o_img = split2(o)
        
        s1 = s_real*p_real*o_real
        s2 = p_real*s_img*o_img
        s3 = p_img*s_real*o_img
        s4 = p_img*s_img*o_real
        return tf.reduce_sum(s1+s2+s3-s4,axis=-1)

class HolE(EmbeddingModel):
    def __init__(self,
                 name='HolE', 
                 **kwargs):
        """HolE implmentation."""
        super(HolE, self).__init__(name=name,**kwargs)
        
    def func(self, s,p,o, training = False):
        def circular_cross_correlation(x, y):
            return tf.math.real(tf.signal.ifft(
            tf.multiply(tf.math.conj(tf.signal.fft(tf.cast(x, tf.complex64))), tf.signal.fft(tf.cast(y, tf.complex64)))))
        
        x = circular_cross_correlation(s,o)
        return tf.reduce_sum(p*x,axis=-1)

class ConvE(EmbeddingModel):
    def __init__(self,
                 name='ConvE', 
                 hidden_dp=0.2,
                 conv_filters=8,
                 conv_size_w=3, 
                 conv_size_h=3,
                 **kwargs):
        """ConvE implmentation."""
        super(ConvE, self).__init__(name=name,**kwargs)
        
        self.dim = kwargs['e_dim']
        factors = lambda val: [(int(i), int(val / i)) for i in range(1, int(val**0.5)+1) if val % i == 0]
        self.w, self.h = factors(self.dim).pop(-1)
    
        self.ls = [Conv2D(conv_filters,(conv_size_w,conv_size_h)),
                   BatchNormalization(),
                    Activation('relu'),
                    Dropout(hidden_dp),
                    Flatten(),
                    Dense(self.dim),
                    BatchNormalization(),
                    Activation('relu'),
                    Dropout(hidden_dp)]
        
    def func(self, s,p,o, training = False):
        s = tf.reshape(s,(-1,self.w,self.h,1))
        p = tf.reshape(p,(-1,self.w,self.h,1))
        x = tf.concat([s,p],axis=1)
        
        for l in self.ls:
            x = l(x)
        
        return tf.reduce_sum(x * o, axis=-1)
    
class ConvR(EmbeddingModel):
    def __init__(self,
                 name='ConvR', 
                 hidden_dp=0.2,
                 conv_filters=8,
                 conv_size_w=3, 
                 conv_size_h=3,
                 **kwargs):
        """ConvR implmentation."""
        kwargs['r_dim'] = conv_filters*conv_size_w*conv_size_h
        super(ConvR, self).__init__(name=name,**kwargs)
        
        self.dim = kwargs['e_dim']
        factors = lambda val: [(int(i), int(val / i)) for i in range(1, int(val**0.5)+1) if val % i == 0]
        self.w, self.h = factors(self.dim).pop(-1)
       
        self.conv_filters = conv_filters
        self.conv_size_h = conv_size_h
        self.conv_size_w = conv_size_w
        
        self.ls = [
            Flatten(),
            Activation('relu'),
            Dense(self.dim),
            Dropout(hidden_dp),
            Activation('relu')
            ]
        
    def func(self,s,p,o, training = False):
        
        def forward(x):
            a, b = x[:self.e_dim], x[self.e_dim:]
            a = tf.reshape(a, (1,self.w,self.h,1))
            b = tf.reshape(b,(self.conv_size_w,self.conv_size_h,1,self.conv_filters))
            return tf.nn.conv2d(a, b, strides = [1,1], padding='SAME')
        
        x = tf.map_fn(forward, tf.concat([s,p],axis=-1))
        
        for l in self.ls:
            x = l(x)
     
        return tf.reduce_sum(x * o, axis=-1)
    
class ConvKB(EmbeddingModel):
    def __init__(self,
                 name='ConvKB', 
                 hidden_dp=0.2,
                 conv_filters=3,
                 num_blocks = 1,
                 **kwargs):
        """ConvKB implmentation."""
        super(ConvKB, self).__init__(name=name,**kwargs)
        factors = lambda val: [(int(i), int(val / i)) for i in range(1, int(val**0.5)+1) if val % i == 0]
        
        self.dim = kwargs['e_dim']
        
        self.w, self.h = self.dim, 3
        
        block = [Conv2D(conv_filters,(1,3),strides=(1,1)),
                 BatchNormalization(),
                  Activation('relu'),
                  Dropout(hidden_dp)]
        
        self.ls = []
        for _ in range(num_blocks):
            self.ls.extend(block)
        
        self.ls.extend([Reshape((3,-1)),
                        Lambda(lambda x: tf.reduce_sum(x[:,0]*x[:,1]*x[:,2],axis=-1))])
        
    def func(self, s,p,o, training = False):
        x = tf.concat([s,p,o],axis=-1)
        x = tf.reshape(x,(-1,self.w,self.h,1))
        
        for l in self.ls:
            x = l(x)
        
        return x

class HAKE(EmbeddingModel):
    def __init__(self,
                 epsilon=2, 
                 gamma = 12,  
                 phase_weight = 0.5,
                 mod_weight = 1,
                 name='HAKE',
                 **kwargs):
        """HAKE implmentation."""
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 3*kwargs['r_dim']
        super(HAKE, self).__init__(name=name,**kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.phase_weight = phase_weight
        self.mod_weight = mod_weight
        
        self.pi = np.pi
        self.embedding_range = (self.gamma + self.epsilon) / self.e_dim / 2
        
    def func(self, s,p,o, training = False):
        split2 = lambda x: tf.split(x,num_or_size_splits=2,axis=-1)
        split3 = lambda x: tf.split(x,num_or_size_splits=3,axis=-1)
        
        phase_s, mod_s = split2(s)
        phase_o, mod_o = split2(o)
        phase_p, mod_p, bias_p = split3(p)
        
        phase_s = phase_s / (self.embedding_range / self.pi)
        phase_p = phase_p / (self.embedding_range / self.pi)
        phase_o = phase_o / (self.embedding_range / self.pi)
        
        bias_p = K.clip(bias_p,min_value=-np.Inf,max_value=1.)
        bias_p = tf.where(bias_p < -K.abs(mod_p), -K.abs(mod_p), bias_p)
        
        r_score = self.mod_weight*tf.norm(mod_s * (mod_p + bias_p) - K.abs(mod_o) * (1-bias_p), ord=2)
        p_score = self.phase_weight*tf.norm(tf.math.sin((phase_s+phase_p-phase_o)/2), ord=1,axis=-1)
        return self.gamma - (p_score + r_score)
        
        
class ModE(EmbeddingModel):
    def __init__(self,
                 gamma=12,
                 norm = 2,
                 name='ModE', 
                 **kwargs):
        """ModE implmentation."""
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 3*kwargs['r_dim']
        super(ModE, self).__init__(name=name,**kwargs)
        
        self.norm = norm
        self.gamma
       
    def func(self, s,p,o, training = False):
        return self.gamma - tf.norm(s * p - o, ord=self.norm, axis=-1)
        
    
class RotatE(EmbeddingModel):
    def __init__(self,
                 gamma=12,
                 norm=2,
                 epsilon=2,
                 name='RotatE',
                 **kwargs):
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = kwargs['r_dim']
        super(RotatE, self).__init__(name=name,**kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm = norm
        
        self.pi = np.pi
        self.embedding_range = (self.gamma + self.epsilon) / self.e_dim / 2
        
    def func(self, s,p,o, training=False):
        re_s, im_s = tf.split(s,num_or_size_splits=2,axis=-1)
        re_o, im_o = tf.split(o,num_or_size_splits=2,axis=-1)
        
        phase_r = tf.math.atan2(tf.math.sin(p),tf.math.cos(p))
        
        re_r = tf.math.cos(phase_r)
        im_r = tf.math.sin(phase_r)
        
        re_score = re_s * re_r - im_s * im_r
        im_score = re_s * im_r + im_s * re_r
        re_score = re_score - re_o
        im_score = im_score - im_o
        
        score = tf.concat([re_score,im_score],axis=1)
        score = tf.reduce_sum(score,axis=1)
        
        if self.gamma > 0:
            return self.gamma - score
        else:
            return score
    
class pRotatE(EmbeddingModel):
    def __init__(self,
                 gamma=12,
                 epsilon=2,
                 modulus=0.5,
                 name='pRotatE',
                 **kwargs):
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 2*kwargs['r_dim']
        super(pRotatE, self).__init__(name=name,**kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.pi = np.pi
        self.embedding_range = (self.gamma + self.epsilon) / self.e_dim / 2
        self.modulus = modulus*self.embedding_range
        
    def func(self, s,p,o, training=False):
        
        phase_s = tf.math.atan2(tf.math.sin(s),tf.math.cos(s))
        phase_p = tf.math.atan2(tf.math.sin(p),tf.math.cos(p))
        phase_o = tf.math.atan2(tf.math.sin(o),tf.math.cos(o))
        
        score = tf.abs(tf.math.sin((phase_s + phase_p - phase_o)/2))
        if self.gamma > 0:
            return self.gamma - tf.reduce_sum(score,axis=1)*self.modulus
        else:
            return tf.reduce_sum(score,axis=1)*self.modulus
        
        
