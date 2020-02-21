### tests.py

from KGEkeras.models import DistMult, HolE, TransE, ComplEx, HAKE, ConvE, ModE, ConvR, DenseModel, Hybrid, ConvKB
import numpy as np
import tensorflow as tf
from random import choice
from collections import defaultdict

from keras.layers import Input 
from tqdm import tqdm
from keras.callbacks import Callback, EarlyStopping
from keras.losses import hinge, binary_crossentropy

import kerastuner as kt
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization

from KGEkeras.utils import load_kg, validate

class MyModel(tf.keras.Model):
    def __init__(self, embedding_model,name='my_model'):
        super(MyModel, self).__init__(name=name)
        self.embedding_model = embedding_model
        
    def call(self, inputs):
        return self.embedding_model(inputs)

class MyHyperModel(HyperModel):
    def __init__(self, N, M, bs, embedding_model=DistMult):
        self.N = N
        self.M = M
        self.bs = bs
        self.embedding_model = embedding_model
        
    def build(self, hp):
        ns = self.bs*hp.Choice('negative_samples',[2,10,100])
        dim = hp.Int('embedding_dim',50,200,step=50)
        dm = self.embedding_model(e_dim=dim,
                                  r_dim=dim,
                                  dp=hp.Float('dropout',0.0,0.5,step=0.1,default=0.2),
                                  #hidden_dp=hp.Float('dropout',0.0,0.5,step=0.1,default=0.2),
                                  #num_blocks=hp.Int('conv_blocks',1,4,default=1),
                                  num_entities=self.N,
                                  num_relations=self.M,
                                  loss_function=binary_crossentropy,
                                  negative_samples=ns)
    
        model = MyModel(dm)
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate',[1e-4,1e-3,1e-2]))
    
        model.compile(optimizer=optimizer, loss=lambda y,yhat:0.0)
        
        return model
        
class myCallBack(Callback):
    def __init__(self, validation_data, train_data=None, *args, **kwargs):
        super(myCallBack, self).__init__(*args, **kwargs)
        self.validation_data = validation_data
        self.train_data = train_data
        
    def on_epoch_end(self, epoch, logs = None):
        if epoch >= 10 and epoch % 10 == 0:
            logs = logs or {}
            tmp = validate(self.model, 
                            self.validation_data,
                            self.model.embedding_model.num_entities,
                            self.train_data)
                
            for k in tmp:
                logs['val_'+k] = tmp[k]
                
    def on_train_end(self, logs=None):
        self.on_epoch_end(100,logs=logs)
        
def main():
    
    train = load_kg('./data/nations/train.txt')
    valid = load_kg('./data/nations/valid.txt')
    test = load_kg('./data/nations/test.txt')
    
    E = set([a for a,b,c in train]) | set([c for a,b,c in train])
    E |= set([a for a,b,c in valid]) | set([c for a,b,c in valid])
    E |= set([a for a,b,c in test]) | set([c for a,b,c in test])
    
    R = set([b for a,b,c in train])
    R |= set([b for a,b,c in valid])
    R |= set([b for a,b,c in test])
    
    entity_mapping = {e:i for i,e in enumerate(E)}
    relation_mapping = {r:i for i,r in enumerate(R)}
    
    train = [(entity_mapping[a],relation_mapping[b],entity_mapping[c]) for a,b,c in train]
    valid = [(entity_mapping[a],relation_mapping[b],entity_mapping[c]) for a,b,c in valid]
    test = [(entity_mapping[a],relation_mapping[b],entity_mapping[c]) for a,b,c in test]
    
    bs = 16
    hypermodel = MyHyperModel(len(E),len(R),bs,embedding_model=DistMult)
    
    tuner = RandomSearch(
        hypermodel,
        objective=kt.Objective("val_mrr", direction="max"),
        max_trials=10,
        seed=42,
        project_name=None)
    
    tuner.search(np.asarray(train),np.ones(len(train)),
             validation_data=(np.asarray(valid),np.ones(len(valid))),
             epochs=12,
             batch_size=bs,
             callbacks = [myCallBack(np.asarray(valid),np.asarray(train)),
                          EarlyStopping(monitor='loss',patience=2)])
             
    tuner.results_summary()
    
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    
    print(validate(best_model, 
                   np.asarray(test),
                   best_model.embedding_model.num_entities,
                   np.asarray(train)))
    
if __name__ == '__main__':
    main()
