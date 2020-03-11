### tests.py

from KGEkeras.models import DistMult, HolE, TransE, ComplEx, HAKE, ConvE, ModE, ConvR, DenseModel, Hybrid, ConvKB, RotatE, pRotatE
import numpy as np
import tensorflow as tf
from random import choice, choices
from collections import defaultdict

from keras.layers import Input 
from tqdm import tqdm
from keras.callbacks import Callback, EarlyStopping
from keras.losses import hinge, binary_crossentropy

import kerastuner as kt
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization

from KGEkeras.utils import load_kg, validate

models = {'DistMult':DistMult,
           # 'TransE':TransE,
           # 'HolE':HolE,
           # 'ComplEx':ComplEx,
           # 'ConvE':ConvE,
            #'ConvR':ConvR,
            #  'HAKE':HAKE,
            #  'RotatE':RotatE,
            #  'pRotatE':pRotatE
         }

class MyModel(tf.keras.Model):
    def __init__(self, embedding_model,name='my_model'):
        super(MyModel, self).__init__(name=name)
        self.embedding_model = embedding_model
        
    def call(self, inputs):
        return self.embedding_model(inputs)

class MyHyperModel(HyperModel):
    def __init__(self, N, M, bs):
        self.N = N
        self.M = M
        self.bs = bs
        
    def build(self, hp):
        embedding_model = models[hp.Choice('embedding_model',list(models.keys()))]
        #dim = hp.Int('embedding_dim',50,200,step=50)
        dim = 256
        dm = embedding_model(e_dim=dim,
                                  r_dim=dim,
                                  dp=0.2,
                                  batch_size=self.bs,
                                  num_entities=self.N,
                                  num_relations=self.M,
                                  loss_function='hinge',
                                  loss_type='pairwize',
                                  negative_samples=10)
    
        model = MyModel(dm)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
        model.compile(optimizer=optimizer, loss=lambda x,y:0.0)
        
        return model
        
class myCallBack(Callback):
    def __init__(self, validation_data=[], train_data=[], bs=32, *args, **kwargs):
        super(myCallBack, self).__init__(*args, **kwargs)
        self.train_data = train_data
        self.validation_data = validation_data
        self.bs = bs
        
    def on_epoch_end(self, epoch, logs = None):
        if epoch % 100 == 0 and epoch > 0:
            logs = logs or {}
            
            tmp = validate(self.model, 
                            self.validation_data,
                            self.model.embedding_model.num_entities,
                            self.bs,
                            self.train_data)
                
            for k in tmp:
                logs['val_'+k] = tmp[k]
            print(logs)

def pad(kg,bs):
    while len(kg) % bs != 0:
        kg.append(choice(kg))
    return kg

def main():
    
    train = load_kg('./data/FB15k-237/train.txt')
    valid = load_kg('./data/FB15k-237/valid.txt')
    test = load_kg('./data/FB15k-237/test.txt')
    
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
    
    bs = 1024
    train = pad(train,bs)
    valid = pad(valid,bs)
    test = pad(test,bs)
    
    hypermodel = MyHyperModel(len(E),len(R),bs)
    
    tuner = RandomSearch(
        hypermodel,
        objective=kt.Objective("val_mrr", direction="max"),
        max_trials=1,
        seed=42,
        overwrite=True,
        project_name=None)
    
    tuner.search(np.asarray(train),np.ones(len(train)),
             validation_data=(np.asarray(valid),np.ones(len(valid))),
             epochs=1000,
             batch_size=bs,
             verbose=2,
             callbacks = [myCallBack(np.asarray(valid),bs=bs,train_data=np.asarray(train))
                          ,EarlyStopping(monitor='loss',patience=2)])
             
    tuner.results_summary()
    
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    
    best_model.fit(np.asarray(train),np.ones(len(train)),
             epochs=1,
             batch_size=bs,
             callbacks = [EarlyStopping(monitor='loss',patience=5)])
    
    tmp = validate(best_model, 
                   np.asarray(test),
                   len(E),
                   bs,
                   np.asarray(train))
    print(tmp)
                          
             
if __name__ == '__main__':
    main()
