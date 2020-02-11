### KG embedding version 3.

from keras.models import Model
from keras.layers import Layer, Embedding, Lambda, Multiply, Reshape, Concatenate, BatchNormalization, Conv2D, Activation, Dense, Dropout, Conv3D
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import numpy as np

from hyperopt import hp

class EmbeddingModel(tf.keras.Model):
    def __init__(self, 
                 e_dim, 
                 r_dim, 
                 num_entities, 
                 num_relations, 
                 negative_samples=2, 
                 loss_function=binary_crossentropy,
                 name='EmbeddingModel',
                 use_bn = True, 
                 dp = 0.2,
                 hp=None,
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
            
        loss_function : keras.losses.Loss
        
        use_bn : bool 
            Batch norm. 
            
        use_dp : bool 
            Use dropout.
        """
        super(EmbeddingModel, self).__init__(name=name,**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_embedding = Embedding(num_entities, e_dim)
        self.relational_embedding = Embedding(num_relations, r_dim)
        self.loss_function = loss_function
        self.negative_samples = negative_samples
        
        self.dp = dp
        self.use_bn = use_bn
        
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],1)
    
    def call(self,inputs,training=False):
        """
        Parameters
        ----------
        inputs : tensor, shape = (batch_size, 3)
        """
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        
        fs = tf.random.uniform((self.negative_samples,),
                               minval=0, 
                               maxval=self.num_entities, 
                               dtype=tf.dtypes.int32)
        fp = tf.random.uniform((self.negative_samples,), 
                               minval=0, 
                               maxval=self.num_relations,
                               dtype=tf.dtypes.int32)
        fo = tf.random.uniform((self.negative_samples,), 
                               minval=0, 
                               maxval=self.num_entities,
                               dtype=tf.dtypes.int32)
        
        s,p,o = self.entity_embedding(s), self.relational_embedding(p), self.entity_embedding(o)
        fs,fp,fo = self.entity_embedding(fs), self.relational_embedding(fp), self.entity_embedding(fo)
        
        s,p,o = Dropout(self.dp)(s),Dropout(self.dp)(p),Dropout(self.dp)(o)
        fs,fp,fo = Dropout(self.dp)(fs),Dropout(self.dp)(fp),Dropout(self.dp)(fo)
        
        true_score = self.func(s,p,o,training)
        false_score = self.func(fs,fp,fo,training)
        
        true_score = K.expand_dims(true_score)
        false_score = K.expand_dims(false_score)
        
        true_loss = tf.reduce_mean(self.loss_function(tf.ones(tf.size(true_score)),true_score))
        false_loss = tf.reduce_mean(self.loss_function(tf.zeros(tf.size(false_score)),false_score))
        
        loss = 0.5*(true_loss + false_loss)
        
        self.add_loss(loss)
        
        return true_score

class DistMult(EmbeddingModel):
    def __init__(self,
                 name='DistMult', 
                 **kwargs):
        """DistMult implmentation."""
        super(DistMult, self).__init__(**kwargs)
    
    def func(self, s,p,o, training = False):
        return tf.sigmoid(tf.reduce_sum(s*p*o, axis=-1))
        

class TransE(EmbeddingModel):
    def __init__(self, 
                 name='TransE',
                 norm=2,
                 **kwargs):
        """TransE implmentation."""
        super(TransE, self).__init__(**kwargs)
        
        self.norm = norm
        self.loss_function = lambda y,yhat: tf.nn.relu(gamma + y*yhat + (y-1)*yhat)
    
    def func(self, s,p,o, training = False):
        if training:
            return tf.norm(s+p-o, axis=-1, ord=self.norm)
        return 1 - tf.norm(s+p-o, axis=-1, ord=self.norm)

class HolE(EmbeddingModel):
    def __init__(self,
                 name='HolE', 
                 **kwargs):
        """HolE implmentation."""
        super(HolE, self).__init__(**kwargs)
    
    def func(self, s,p,o, training = False):
        def circular_cross_correlation(x, y):
            return tf.math.real(tf.signal.ifft(
            tf.multiply(tf.math.conj(tf.signal.fft(tf.cast(x, tf.complex64))), tf.signal.fft(tf.cast(y, tf.complex64)))))
        
        return tf.sigmoid(tf.reduce_sum(tf.multiply(p, circular_cross_correlation(s, o)), axis=-1))

class ComplEx(EmbeddingModel):
    def __init__(self,
                 name='ComplEx', 
                 norm=1, 
                 **kwargs):
        """ComplEx implmentation."""
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 2*kwargs['r_dim']
        super(ComplEx, self).__init__(**kwargs)
        
        self.norm = norm
    
    def func(self, s,p,o, training = False):
        split2 = lambda x: tf.split(x,num_or_size_splits=2,axis=-1)
        s_real, s_img = split2(s)
        p_real, p_img = split2(p)
        o_real, o_img = split2(o)
        
        s1 = s_real*p_real*o_real 
        s2 = p_real*s_img*o_img
        s3 = p_img*s_real*o_img
        s4 = p_img*s_img*o_real
        s1 = tf.norm(s1, axis=-1, ord=self.norm)
        s2 = tf.norm(s2, axis=-1, ord=self.norm)
        s3 = tf.norm(s3, axis=-1, ord=self.norm)
        s4 = tf.norm(s4, axis=-1, ord=self.norm)
        return tf.sigmoid(s1+s2+s3-s4)


class ConvE(EmbeddingModel):
    def __init__(self,
                 name='ConvE', 
                 hidden_dp=0.2,
                 conv_filters=8,
                 conv_size_w=3, 
                 conv_size_h=3,
                 **kwargs):
        """ConvE implmentation."""
        super(ConvE, self).__init__(**kwargs)
        
        self.dim = kwargs['e_dim']
        factors = lambda val: [(int(i), int(val / i)) for i in range(1, int(val**0.5)+1) if val % i == 0]
        self.w, self.h = factors(self.dim).pop(-1)
        
        self.hidden_dp = hidden_dp
    
        self.conv_filters = conv_filters
        self.conv_size_h = conv_size_h
        self.conv_size_w = conv_size_w
        
        self.ls = [BatchNormalization(),
                  Conv2D(self.conv_filters,(self.conv_size_w,conv_size_h)),
                  Activation('relu'),
                  Dropout(self.hidden_dp),
                  Activation('relu'),
                  Reshape((-1,)),
                  Dense(self.dim)]
        
    def func(self, s,p,o, training = False):
        s = Reshape((self.w,self.h))(s)
        p = Reshape((self.w,self.h))(p)
        x = Concatenate(axis=1)([s,p])
        x = K.expand_dims(x,axis=-1)
        
        for l in self.ls:
            x = l(x)
        
        return tf.sigmoid(tf.reduce_sum(x * o, axis=-1))
    
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
        super(ConvR, self).__init__(**kwargs)
        
        self.dim = kwargs['e_dim']
        factors = lambda val: [(int(i), int(val / i)) for i in range(1, int(val**0.5)+1) if val % i == 0]
        self.w, self.h = factors(self.dim).pop(-1)
       
        self.hidden_dp = hidden_dp
        self.conv_filters = conv_filters
        self.conv_size_h = conv_size_h
        self.conv_size_w = conv_size_w
        
        self.out = Dense(self.dim)
            
 
    def func(self,s,p,o, training = False):
        
        x = Concatenate()([s,p])
        
        def forward(inputs):
            a,b = inputs[:self.dim], inputs[self.dim:]
            b = tf.reshape(b,(self.conv_size_w,self.conv_size_h,1,self.conv_filters))
            a = tf.reshape(a,(1,self.w,self.h,1))
            
            x = tf.nn.conv2d(a, b, strides=2, padding='SAME')
            x = tf.reshape(x,(-1,))
            x = Activation('relu')(x)
            
            x = tf.expand_dims(x,axis=0)
            x = self.out(x)
            x = Dropout(self.hidden_dp)(x)
            x = Activation('relu')(x)
            
            return x
        
        
        x = tf.map_fn(forward, x)
        x = tf.squeeze(x,1)
        
        return tf.sigmoid(tf.reduce_sum(x * o, axis=-1))
    
    

class HAKE(EmbeddingModel):
    def __init__(self,
                 epsilon=2, 
                 gamma = 12,  
                 phase_weight = 0.5,
                 mod_weight = 1,
                 norm = 2,
                 name='HAKE',
                 **kwargs):
        """HAKE implmentation."""
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 3*kwargs['r_dim']
        super(HAKE, self).__init__(**kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.phase_weight = phase_weight
        self.mod_weight = mod_weight
        self.norm = norm
        
        self.pi = np.pi
        self.embedding_range = (self.gamma + self.epsilon) / dim
        
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
        
        return self.gamma - (self.mod_weight*tf.norm(mod_s * (mod_p + bias_p) - K.abs(mod_o) * (1-bias_p)) + self.phase_weight*tf.norm(tf.math.sin((phase_s+phase_p-phase_o)/2), ord=self.norm,axis=-1))
        
class ModE(EmbeddingModel):
    def __init__(self,
                 gamma = 12, 
                 norm = 2,
                 name='ModE', 
                 **kwargs):
        """ModE implmentation."""
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 3*kwargs['r_dim']
        super(ModE, self).__init__(**kwargs)
        
        self.gamma = gamma
        self.norm = norm
        
    def func(self, s,p,o, training = False):
        return self.gamma - tf.norm(s * p - o, ord=self.norm, axis=-1)
    
    
class DenseModel(EmbeddingModel):
    def __init__(self,
                 name='Dense',
                 hidden_units=[],
                 **kwargs):
        """Dense model."""
        super(DenseModel, self).__init__(**kwargs)
        self.ls = [Dense(i, activation='relu') for i in hidden_units]
        self.output = Dense(1, activation='sigmoid')
    
    def func(self, s,p,o, training=False):
        x = tf.concat([s,p,o],axis=-1)
        for l in self.ls:
            x = l(x)
        x = self.output(x)
        return x
    
class Ensemble(EmbeddingModel):
    def __init__(self,
                 models, 
                 **kwargs):
        """Ensemble model. Outputs the ensemble mean."""
        super(Hybrid, self).__init__(**kwargs)
        
        self.models = models
    
    def call(self,inputs):
        v = [m(inputs) for m in self.models]
        return tf.reduce_mean(v, axis=-1)
        
        
    
        
        
    
    
    
        
        
        
