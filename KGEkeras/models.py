### KG embedding version 3.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, Lambda, Multiply, Reshape, Concatenate, BatchNormalization, Conv2D, Activation, Dense, Dropout, Conv3D, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from tensorflow.keras.initializers import glorot_normal

from tensorflow.keras.constraints import UnitNorm, MaxNorm

def pointwize_hinge(ytrue,ypred,margin=1):
    return tf.reduce_sum(tf.nn.relu(margin-ytrue*ypred))

def pointwize_logistic(ytrue,ypred):
    return tf.reduce_sum(tf.math.log(1+tf.math.exp(-ytrue*ypred)))

def pointwize_square_loss(ytrue,ypred):
    return 0.5 * tf.reduce_sum(tf.square(margin-ytrue*ypred))

def pairwize_hinge(true,false,margin=1):
    return tf.reduce_sum(tf.nn.relu(margin+false-true))

def pairwize_logistic(true,false):
    return tf.reduce_sum(tf.math.log(1+tf.math.exp(false-true)))

def pairwize_square_loss(true,false):
    return - tf.reduce_sum(tf.square(false-true))

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
        super(EmbeddingModel, self).__init__(name=name)
        if regularization != 0.0:
            reg = lambda x: l3_reg(x,regularization)
        else:
            reg = None
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_embedding = Embedding(input_dim=num_entities, output_dim=e_dim,embeddings_initializer=glorot_normal(),embeddings_regularizer=reg)
        self.relational_embedding = Embedding(input_dim=num_relations, output_dim=r_dim,embeddings_initializer=glorot_normal(),embeddings_regularizer=reg)
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
        
        if loss_type == 'pointwize':
            
            if self.loss_function == 'hinge':
                lf = lambda x,y: pointwize_hinge(x,y,self.margin)
            elif self.loss_function == 'logistic':
                lf = pointwize_logistic
            elif self.loss_function == 'square':
                lf = pointwize_square_loss
            elif self.loss_function == 'cross entropy':
                lf = binary_crossentropy
                self.pos_label, self.neg_label = max(self.pos_label,0), max(self.neg_label,0)
            else:
                raise NotImplementedError(self.loss_function+' is not implemented.')
            
            self.lf = lambda true,false: lf(self.pos_label,true) + lf(self.neg_label,false)/self.negative_samples
        else:
            if self.loss_function == 'hinge':
                lf = lambda x,y: pairwize_hinge(x,y,self.margin)
            elif self.loss_function == 'logistic':
                lf = pairwize_logistic
            elif self.loss_function == 'square':
                lf = pairwize_square_loss
            elif self.loss_function == 'cross entropy':
                lf = pairwize_cross_entropy
            else:
                raise NotImplementedError(self.loss_function+' is not implemented.')
            
            def pairwize(x,y):
                x = tf.repeat(x,self.negative_samples,0)
                x = tf.reshape(x,(self.batch_size,self.negative_samples,1))
                y = tf.reshape(y,(self.batch_size,self.negative_samples,1))
                return lf(self.pos_label*x,self.neg_label*y)
            
            self.lf = pairwize
        
        self.__dict__.update(kwargs)
    
    def call(self,inputs,training=False):
        """
        Parameters
        ----------
        inputs : tensor, shape = (batch_size, 3)
        """
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        fp = p
        s,p,o = self.entity_embedding(s), self.relational_embedding(p), self.entity_embedding(o)
        
        s,p,o = Dropout(self.dp)(s),Dropout(self.dp)(p),Dropout(self.dp)(o)
        
        true_score = self.func(s,p,o,training)
        true_score = tf.math.log_sigmoid(self.pos_label*K.expand_dims(true_score))
        
        if training:
            fs = tf.random.uniform((self.negative_samples*self.batch_size,),
                                minval=0, 
                                maxval=self.num_entities, 
                                dtype=tf.dtypes.int32)

            fp = tf.repeat(fp, self.negative_samples, 0)

            fo = tf.random.uniform((self.negative_samples*self.batch_size,), 
                                minval=0, 
                                maxval=self.num_entities,
                                dtype=tf.dtypes.int32)

            fs,fp,fo = self.entity_embedding(fs), self.relational_embedding(fp), self.entity_embedding(fo)
            fs,fp,fo = Dropout(self.dp)(fs),Dropout(self.dp)(fp),Dropout(self.dp)(fo)

            false_score = self.func(fs,fp,fo,training)
            false_score = tf.math.log_sigmoid(self.neg_label*K.expand_dims(false_score))
        
            loss = self.lf(true_score,false_score)
        else:
            loss = 0.0
        
        self.add_loss(self.loss_weight*loss)
        
        return true_score
        

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

class ComplEx(EmbeddingModel):
    def __init__(self,
                 name='ComplEx', 
                 **kwargs):
        """ComplEx implmentation."""
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 2*kwargs['r_dim']
        super(ComplEx, self).__init__(**kwargs)
    
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
        super(HolE, self).__init__(**kwargs)
    
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
        super(ConvR, self).__init__(**kwargs)
        
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
        
        s = tf.reshape(s, (-1,1,self.w,self.h,1))
        p = tf.reshape(p,(-1,self.conv_size_w,self.conv_size_h,1,self.conv_filters))
        
        x = tf.nn.conv3d(s, p, strides = [1,1,2,2,1], padding='SAME')
        
        for l in self.ls:
            x = l(x)
     
        return tf.reduce_sum(x * o, axis=-1)
    
class ConvKB(EmbeddingModel):
    def __init__(self,
                 name='ConvKB', 
                 hidden_dp=0.2,
                 conv_filters=8,
                 num_blocks = 1,
                 **kwargs):
        """ConvKB implmentation."""
        super(ConvKB, self).__init__(**kwargs)
        factors = lambda val: [(int(i), int(val / i)) for i in range(1, int(val**0.5)+1) if val % i == 0]
        
        self.dim = kwargs['e_dim']
        
        self.w, self.h = factors(3*self.dim).pop(-1)
        
        block = [BatchNormalization(),
                  Conv2D(conv_filters,(3,3),strides=(1,1)),
                  Activation('relu'),
                  Dropout(hidden_dp)]
        
        self.ls = []
        for _ in range(num_blocks):
            self.ls.extend(block)
        
        self.ls.extend([Reshape((-1,)),
                  Dense(self.dim),
                  Activation('relu'),
                  Dropout(hidden_dp),
                  Dense(1)])
        
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
    
class RotatE(EmbeddingModel):
    def __init__(self,
                 gamma=12,
                 norm=2,
                 epsilon=2,
                 name='RotatE',
                 **kwargs):
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = kwargs['r_dim']
        super(RotatE, self).__init__(**kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm = norm
        
        self.pi = np.pi
        self.embedding_range = (self.gamma + self.epsilon) / self.e_dim / 2
    
    def func(self, s,p,o, training=False):
        re_s, im_s = tf.split(s,num_or_size_splits=2,axis=-1)
        re_o, im_o = tf.split(o,num_or_size_splits=2,axis=-1)
        
        phase_r = p/(self.embedding_range/self.pi)
        re_r = tf.math.cos(phase_r)
        im_r = tf.math.sin(phase_r)
        
        re_score = re_s * re_r - im_s * im_r
        im_score = re_s * im_r + im_s * re_r
        re_score = re_score - re_o
        im_score = im_score - im_o
        
        score = tf.concat([re_score, im_score], axis = 1)
        score = tf.norm(score, axis = 1, ord=self.norm)
        
        return self.gamma - score
    
class pRotatE(EmbeddingModel):
    def __init__(self,
                 gamma=12,
                 norm=2,
                 epsilon=2,
                 modulus=0.5,
                 name='pRotatE',
                 **kwargs):
        kwargs['e_dim'] = 2*kwargs['e_dim']
        kwargs['r_dim'] = 2*kwargs['r_dim']
        super(pRotatE, self).__init__(**kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm = norm
        
        self.pi = np.pi
        self.embedding_range = (self.gamma + self.epsilon) / self.e_dim / 2
        self.modulus = modulus*self.embedding_range
    
    def func(self, s,p,o, training=False):
        
        phase_r = p/(self.embedding_range/self.pi)
        phase_s = s/(self.embedding_range/self.pi)
        phase_o = o/(self.embedding_range/self.pi)
        
        score = tf.abs(tf.math.sin(phase_s + phase_r - phase_o))
        
        return self.gamma - tf.reduce_sum(score,axis=1)*self.modulus

class DenseModel(EmbeddingModel):
    def __init__(self,
                 name='Dense',
                 hidden_units=[32],
                 **kwargs):
        """Dense model."""
        super(DenseModel, self).__init__(**kwargs)
        self.ls = [Dense(i, activation='relu') for i in hidden_units]
        self.output = Dense(1)
    
    def func(self, s,p,o, training=False):
        x = tf.concat([s,p,o],axis=-1)
        for l in self.ls:
            x = l(x)
        x = self.output(x)
        return x
    
class Hybrid(EmbeddingModel):
    def __init__(self,
                 models = {},
                 default_model = DistMult,
                 **kwargs):
        """Hybrid model. Outputs the ensemble mean.
        
        Parameters
        ---------
        models : dict
            {prop_id : model}
        """
        super(Hybrid, self).__init__(**kwargs)
        self.models = models
        
        for k in range(kwargs['num_relations']):
            if not k in self.models:
                self.models[k] = default_model(**kwargs)
        self.models = list([i for k,i in self.models.items()])
        
        self.ensemble_layer = Dense(1,use_bias=False,kernel_constraint=UnitNorm())
    
    def call(self, inputs, training=False):
        
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
        
        ints = p
        f_ints = fp
        
        s,p,o = self.entity_embedding(s), self.relational_embedding(p), self.entity_embedding(o)
        fs,fp,fo = self.entity_embedding(fs), self.relational_embedding(fp), self.entity_embedding(fo)
        
        s,p,o = Dropout(self.dp)(s),Dropout(self.dp)(p),Dropout(self.dp)(o)
        fs,fp,fo = Dropout(self.dp)(fs),Dropout(self.dp)(fp),Dropout(self.dp)(fo)
        
        true_score = [tf.expand_dims(model.func(s,p,o),axis=-1) for model in self.models]
        true_score = tf.gather(true_score, ints)
        true_score = self.ensemble_layer(true_score)
        
        
        false_score = [tf.expand_dims(model.func(fs,fp,fo),axis=-1) for model in self.models]
        false_score = tf.gather(false_score, f_ints)
        false_score = self.ensemble_layer(false_score)
        
        true_loss = tf.reduce_mean(self.loss_function(tf.ones(tf.size(true_score)),true_score))
        false_loss = tf.reduce_mean(self.loss_function(tf.zeros(tf.size(false_score)),false_score))
        
        loss = 0.5*(true_loss + false_loss)
        
        self.add_loss(loss)
        
        return true_score
        
