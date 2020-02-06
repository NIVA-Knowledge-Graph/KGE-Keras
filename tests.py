### tests.py

from models import DistMult, HolE, TransE, ComplEx, HAKE, ConvE, ModE, ConvR, Hybrid
import numpy as np
import tensorflow as tf
from random import choice
from collections import defaultdict

from keras.layers import Input 
from keras.models import Model
from tqdm import tqdm

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
        
def validate(model, test_data, num_entities, batch_size, filtering = True):
    
    metrics = defaultdict(list)
    
    filtering_tail = defaultdict(set)
    filtering_head = defaultdict(set)
    
    if filtering:
        for s,p,o in test_data:
            filtering_tail[(s,p)].add(o)
            filtering_head[(p,o)].add(s)
        
    for s,p,o in tqdm(test_data):
        #tail
        ft = filtering_tail[(s,p)] - set([o])
        
        inputs = [(s,p,x) for x in range(num_entities) if not x in ft]
        pad(inputs, batch_size)
        predicted_o = list(model.predict(np.asarray(inputs)))[:num_entities-len(ft)]
        predicted = [(x[2],p) for x,p in zip(inputs,predicted_o)]
        
        metrics['tail_mrr'].append(mrr(o,predicted))
        metrics['tail_h@1'].append(hits(o,predicted,1))
        metrics['tail_h@3'].append(hits(o,predicted,3))
        metrics['tail_h@10'].append(hits(o,predicted,10))
        
        #head
        fh = filtering_head[(p,o)] - set([s])
        
        inputs = [(x,p,o) for x in range(num_entities) if not x in fh]
        pad(inputs, batch_size)
        predicted_s = list(model.predict(np.asarray(inputs)))[:num_entities-len(fh)]
        predicted = [(x[0],p) for x,p in zip(inputs,predicted_s)]
        metrics['head_mrr'].append(mrr(s,predicted))
        metrics['head_h@1'].append(hits(s,predicted,1))
        metrics['head_h@3'].append(hits(s,predicted,3))
        metrics['head_h@10'].append(hits(s,predicted,10))
        
    for k in metrics:
        print(k,np.mean(metrics[k]))
        
def main():
    
    bs = 128
    
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
    
    pad(train,bs)
    
    models = []
    for kge_model in [TransE, ComplEx, HolE, ConvE, ConvR, HAKE]:
        dm = kge_model(200, len(entity_mapping), len(relation_mapping), negative_samples=10, use_bn=True, use_dp=True)
    
        input_layer = Input(shape=(3,), name='input_layer', batch_shape=(bs,3))
        x = dm(input_layer)
        my_mod = Model(inputs=input_layer, outputs=x)
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            
        my_mod.compile(optimizer='adam', loss = 'binary_crossentropy')
        my_mod.fit(np.asarray(train),np.ones(len(train)).reshape((-1,1)),batch_size=bs,epochs=100,verbose=1)
            
        validate(my_mod, valid, len(entity_mapping), bs)
    
if __name__ == '__main__':
    main()
