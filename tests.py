### tests.py

from KGEkeras.models import DistMult, HolE, TransE, ComplEx, HAKE, ConvE, ModE, ConvR, Ensemble, DenseModel
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

from itertools import product

def load_kg(path):
    out = []
    with open(path,'r') as f:
        for l in f:
            l = l.strip().split()
            out.append(l)
    return out

def pad(l, bs):
    num = bs - len(l) % bs
    for _ in range(num):
        l.append(choice(l))
        
def mrr(target, scores):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    labels = [x for x,_ in scores]
    return 1/(1+labels.index(target))

def hits(target, scores, k=10):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    labels = [x for x,_ in scores][:k]
    return int(target in labels)
        
def validate(model, test_data, num_entities, filtering = False, train_triples = set()):
    
    metrics = defaultdict(list)
    
    tp = set([p for _,p,_ in test_data])
    test_triples = list(product(range(num_entities),tp,range(num_entities)))
    if filtering: test_triples = list( set(test_triples) - set(train_triples) )
    
    prediction = dict(zip(test_triples, model.predict(np.asarray(test_triples))))
    prediction = defaultdict(lambda : 0, prediction)
    
    for s,p,o in test_data:
        t = range(num_entities)
        scores_t = [(x,prediction[(s,p,x)]) for x in t]
        
        h = range(num_entities)
        scores_h = [(x,prediction[(x,p,o)]) for x in h]
        
        metrics['tail_mrr'].append(mrr(o,scores_t))
        metrics['tail_h@1'].append(hits(o,scores_t,1))
        metrics['tail_h@3'].append(hits(o,scores_t,3))
        metrics['tail_h@10'].append(hits(o,scores_t,10))
        
        metrics['head_mrr'].append(mrr(s,scores_h))
        metrics['head_h@1'].append(hits(s,scores_h,1))
        metrics['head_h@3'].append(hits(s,scores_h,3))
        metrics['head_h@10'].append(hits(s,scores_h,10))
        
    metrics = {k:np.mean(metrics[k]) for k in metrics}
    tmp = {}
    for k in metrics:
        s = k.split('_')[-1]
        tmp[s] = (metrics['tail_'+s]+metrics['head_'+s])/2
    
    return dict(metrics, **tmp)
        

class MyModel(tf.keras.Model):
    def __init__(self, embedding_model):
        super(MyModel, self).__init__()
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
        dim = hp.Int('embedding_dim',50,200)
        dm = self.embedding_model(e_dim=dim,
                                  r_dim=dim,
                                  dp=hp.Float('droupout',0.0,0.5),
                                  num_entities=self.N, 
                                  num_relations=self.M, loss_function=binary_crossentropy,
                                  hp=hp,
                                  negative_samples=hp.Int('negative_samples', self.bs*2, self.bs*10))
    
        model = MyModel(dm)
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate',1e-4,1e-2,sampling='log'))
    
        model.compile(optimizer=optimizer, loss=lambda y,yhat:0.0, metrics=['acc'])
        
        return model
        
class myCallBack(Callback):
    def __init__(self, validation_data, *args, **kwargs):
        super(myCallBack, self).__init__(*args, **kwargs)
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            tmp = validate(self.model, self.validation_data, self.model.embedding_model.num_entities)
            
            for k in tmp:
                logs['val_'+k] = tmp[k]
            print(logs['val_mrr'])
            
def main():
    
    bs = 512
    
    train = load_kg('./data/kinship/train.txt')
    valid = load_kg('./data/kinship/valid.txt')
    test = load_kg('./data/kinship/test.txt')
    
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
    
    hypermodel = MyHyperModel(len(E),len(R),bs,embedding_model=DistMult)
    
    tuner = RandomSearch(
        hypermodel,
        objective=kt.Objective("val_mrr", direction="max"),
        max_trials=10,
        seed=10,
        project_name=None)
    
    tuner.search(np.asarray(train),np.ones(len(train)),
             validation_data=(np.asarray(valid),np.ones(len(valid))),
             epochs=100,
             batch_size=bs,
             callbacks = [myCallBack(np.asarray(valid)),
                          EarlyStopping('val_loss',patience=2)])
             
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    
    tuner.results_summary()
    print(validate(best_model, test, len(E), filtering=True, train_triples=train))
    
if __name__ == '__main__':
    main()
