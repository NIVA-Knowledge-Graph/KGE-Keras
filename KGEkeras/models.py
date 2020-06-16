### KG embedding version 3.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, Lambda, Multiply, Reshape, Concatenate, BatchNormalization, Conv2D, Activation, Dense, Dropout, Conv3D, Flatten
from tensorflow.keras.losses import binary_crossentropy, cosine_similarity
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import RandomUniform, GlorotUniform

from tensorflow.keras.constraints import UnitNorm, MaxNorm

EPSILON = 1e-6

def pointwize_hinge(ytrue,ypred,margin=1):
    return tf.reduce_mean(tf.nn.relu(margin-ytrue*ypred))

def pointwize_logistic(ytrue,ypred):
    return tf.reduce_mean(tf.math.log(EPSILON+1+tf.math.exp(-ytrue*ypred)))

def pointwize_square_loss(ytrue,ypred,margin=1):
    return 0.5 * tf.reduce_mean(tf.square(margin-ytrue*ypred))

def pairwize_hinge(true,false,margin=1):
    return tf.reduce_mean(tf.nn.relu(margin+false-true))

def pairwize_logistic(true,false):
    return tf.reduce_mean(tf.math.log(EPSILON+1+tf.math.exp(false-true)))

def pairwize_square_loss(true,false):
    return - tf.reduce_mean(tf.square(false-true))

def pairwize_cross_entropy(true, false):
    return binary_crossentropy(1,true) + binary_crossentropy(0,false)

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
                 loss_function='hinge',
                 loss_type = 'pairwize',
                 name='EmbeddingModel',
                 use_bn = True, 
                 dp = 0.2,
                 margin = 1,
                 loss_weight=1,
                 regularization = 0.01,
                 entity_embedding_args = None,
                 relational_embedding_args = None,
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
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        init_e = GlorotUniform()
        init_r = GlorotUniform()
        self.entity_embedding = Embedding(num_entities,e_dim,embeddings_initializer=init_e, embeddings_regularizer=reg)
        self.relational_embedding = Embedding(num_relations,r_dim,embeddings_initializer=init_r, embeddings_regularizer=reg)
        
        self.dp = dp
        self.loss_function = loss_function
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.e_dim = e_dim
        self.r_dim = r_dim
        self.margin = margin
        self.loss_weight = loss_weight
        self.regularization = regularization
       
        self.pos_label = 1
        self.neg_label = -1
        
        self.loss_type = loss_type
        
        if loss_type == 'margin':
            def pw_loss(x,y):
                x = tf.repeat(x,self.negative_samples,0)
                x = tf.reshape(x,(-1,1))
                y = tf.reshape(y,(-1,1))
                return tf.reduce_mean(tf.math.maximum(x - y + margin, 0))
            lf = pw_loss
            
        elif loss_type == 'sigmoid':
            lf = lambda x,y: - (tf.reduce_mean(tf.math.sigmoid(x)) + tf.reduce_mean(tf.math.sigmoid(-y))) / 2
        elif loss_type == 'softplus':
            lf = lambda x,y: (tf.reduce_mean(tf.math.softplus(-x)) + tf.reduce_mean(tf.math.softplus(y))) / 2
        else:
            raise NotImplementedError
        
        self.lf = lf
       
        self.__dict__.update(kwargs)
    
    def get_config(self):
        return self.__dict__
    
    def call(self,inputs,training=False):
        """
        Parameters
        ----------
        inputs : tensor, shape = (batch_size, 3)
        """
        
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        fs,fp,fo = s,p,o
        
        s,p,o = self.entity_embedding(s),self.relational_embedding(p),self.entity_embedding(o)
        s,p,o = Dropout(self.dp)(s),Dropout(self.dp)(p),Dropout(self.dp)(o)
        
        true_score = self.func(s,p,o,training)
        
        #currupt object
        fs1 = tf.repeat(fs, self.negative_samples, 0)
        fp1 = tf.repeat(fp, self.negative_samples, 0)
        fo1 = tf.random.uniform((self.negative_samples*self.batch_size,), 
                            minval=0, 
                            maxval=self.num_entities,
                            dtype=tf.dtypes.int32)
        
        #corrupt subject
        fs2 = tf.random.uniform((self.negative_samples*self.batch_size,),
                            minval=0, 
                            maxval=self.num_entities, 
                            dtype=tf.dtypes.int32)
        fp2 = tf.repeat(fp, self.negative_samples, 0)
        fo2 = tf.repeat(fo, self.negative_samples, 0)

        fs1,fp1,fo1 = self.entity_embedding(fs1),self.relational_embedding(fp1),self.entity_embedding(fo1)
        fs1,fp1,fo1 = Dropout(self.dp)(fs1),Dropout(self.dp)(fp1),Dropout(self.dp)(fo1)
        fs2,fp2,fo2 = self.entity_embedding(fs2),self.relational_embedding(fp2),self.entity_embedding(fo2)
        fs2,fp2,fo2 = Dropout(self.dp)(fs2),Dropout(self.dp)(fp2),Dropout(self.dp)(fo2)

        false_score1 = self.func(fs1,fp1,fo1,training)
        false_score2 = self.func(fs2,fp2,fo2,training)
        
        loss = (self.lf(true_score,false_score1)+self.lf(true_score,false_score2))/2
            
        self.add_loss(self.loss_weight*loss)
        
        return true_score, loss

class DistMult(EmbeddingModel):
    def __init__(self,
                 name='DistMult', 
                 **kwargs):
        """DistMult implmentation."""
        super(DistMult, self).__init__(**kwargs)
        
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
        return self.gamma - tf.norm(s+p-o, axis=1, ord=self.norm)
    
class CosinE(EmbeddingModel):
    def __init__(self, 
                 name='CosinE',
                 **kwargs):
        """CosinE implmentation."""
        super(CosinE, self).__init__(name=name,**kwargs)
        
    def func(self, s,p,o, training = False):
        return - (1+2*cosine_similarity(s+p,o,axis=-1))

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
        s1 = tf.reduce_sum(s1, axis=-1)
        s2 = tf.reduce_sum(s2, axis=-1)
        s3 = tf.reduce_sum(s3, axis=-1)
        s4 = tf.reduce_sum(s4, axis=-1)
        return s1+s2+s3-s4

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
        
        self.hidden_dp = hidden_dp
    
        self.conv_filters = conv_filters
        self.conv_size_h = conv_size_h
        self.conv_size_w = conv_size_w
        
        self.ls = [Conv2D(self.conv_filters,(self.conv_size_w,conv_size_h)),
                  Activation('relu'),
                  Dropout(self.hidden_dp),
                  Flatten(),
                  Dense(self.dim),
                  Activation('relu'),
                  Dropout(self.hidden_dp)]
        
    def func(self, s,p,o, training = False):
        s = Reshape((self.w,self.h))(s)
        p = Reshape((self.w,self.h))(p)
        x = Concatenate(axis=1)([s,p])
        x = K.expand_dims(x,axis=-1)
        
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
       
        self.hidden_dp = hidden_dp
        self.conv_filters = conv_filters
        self.conv_size_h = conv_size_h
        self.conv_size_w = conv_size_w
        
        self.ls = [
            Flatten(),
            Activation('relu'),
            Dense(self.dim),
            Dropout(self.hidden_dp),
            Activation('relu')
            ]
        
    def func(self,s,p,o, training = False):
        
        def forward(x):
            a, b = x[:self.e_dim], x[self.e_dim:]
            a = tf.reshape(a, (1,self.w,self.h,1))
            b = tf.reshape(b,(self.conv_size_w,self.conv_size_h,1,self.conv_filters))
            return tf.nn.conv2d(a, b, strides = [1,1], padding='SAME')
        
        x = tf.map_fn(forward, Concatenate(axis=-1)([s,p]))
        
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
                  Activation('relu'),
                  Dropout(hidden_dp)]
        
        self.ls = []
        for _ in range(num_blocks):
            self.ls.extend(block)
        
        self.ls.extend([Reshape((3,-1)),
                        Lambda(lambda x: tf.reduce_sum(x[:,0]*x[:,1]*x[:,2],axis=-1))])
        
    def func(self, s,p,o, training = False):
        s = K.expand_dims(s,axis=-1)
        p = K.expand_dims(p,axis=-1)
        o = K.expand_dims(o,axis=-1)
        x = Concatenate(axis=-1)([s,p,o])
        x = K.expand_dims(x,axis=-1)
        
        x = tf.reshape(x, (-1,self.h,self.w,1))
        
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
        
        return self.gamma + (self.mod_weight*tf.norm(mod_s * (mod_p + bias_p) - K.abs(mod_o) * (1-bias_p), ord=2) - self.phase_weight*tf.norm(tf.math.sin((phase_s+phase_p-phase_o)/2), ord=1,axis=-1))
        
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
        
        phase_r = p/(self.embedding_range/self.pi) # try np.wrap
        re_r = tf.math.cos(phase_r)
        im_r = tf.math.sin(phase_r)
        
        re_score = re_s * re_r - im_s * im_r
        im_score = re_s * im_r + im_s * re_r
        re_score = re_score - re_o
        im_score = im_score - im_o
        
        score = tf.concat([re_score, im_score], axis = 1)
        score = tf.reduce_sum(score, axis=1)
        
        return self.gamma - score
    
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
        
        phase_r = p/(self.embedding_range/self.pi)
        phase_s = s/(self.embedding_range/self.pi)
        phase_o = o/(self.embedding_range/self.pi)
        
        score = tf.abs(tf.math.sin((phase_s + phase_r - phase_o)/2))
        
        return self.gamma - tf.reduce_sum(score,axis=1)*self.modulus
        
