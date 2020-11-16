### tests.py

from KGEkeras.models import DistMult, HolE, TransE, ComplEx, HAKE, ConvE, ModE, ConvR, ConvKB, RotatE, pRotatE
import numpy as np
import tensorflow as tf
from random import choice, choices
from collections import defaultdict

from keras.layers import Input 
from tqdm import tqdm
from keras.callbacks import Callback, EarlyStopping
from keras.losses import hinge, binary_crossentropy

from tensorflow.keras.models import Model

from KGEkeras.utils import load_kg, validate, loss_function_lookup, generate_negative, oversample_data

models = {'DistMult':DistMult,
           'TransE':TransE,
           'HolE':HolE,
           'ComplEx':ComplEx,
           'ConvE':ConvE,
            'ConvR':ConvR,
             'HAKE':HAKE,
             'RotatE':RotatE,
             'pRotatE':pRotatE
         }

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, kg, ns=10, batch_size=32, shuffle=True):
        self.batch_size = min(batch_size,len(kg))
        self.kg = kg
        self.ns = ns
        self.num_e = len(set([s for s,_,_ in kg])|set([o for _,_,o in kg]))
        self.shuffle = shuffle
        self.indices = list(range(len(kg)))
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.kg) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        tmp_kg = np.asarray([self.kg[i] for i in batch])
        
        negative_kg = generate_negative(tmp_kg,N=self.num_e,negative=self.ns)
        X = oversample_data(kgs=[tmp_kg,negative_kg])
    
        return X, None 

def build_model(hp):
    
    params = hp.copy()
    params['e_dim'] = params['dim']
    params['r_dim'] = params['dim']
    params['name'] = 'embedding_model'
    
    embedding_model = models[params['embedding_model']]
    embedding_model = embedding_model(**params)
    triple = Input((3,))
    ftriple = Input((3,))
    
    inputs = [triple, ftriple]
    
    score = embedding_model(triple)
    fscore = embedding_model(ftriple)
    
    loss_function = loss_function_lookup(params['loss_function'])
    loss = loss_function(score,fscore,params['margin'] or 1, 1)
    
    model = Model(inputs=inputs, outputs=loss)
    model.add_loss(loss)
    
    model.compile(optimizer='adam',
                  loss=None)
    
    return model

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
    
    literals = np.random.rand(len(E),5)
    
    train = [(entity_mapping[a],relation_mapping[b],entity_mapping[c]) for a,b,c in train]
    valid = [(entity_mapping[a],relation_mapping[b],entity_mapping[c]) for a,b,c in valid]
    test = [(entity_mapping[a],relation_mapping[b],entity_mapping[c]) for a,b,c in test]
    
    bs = 2048
    
    model = build_model({'num_entities':len(E),
                              'num_relations':len(R),
                              'dim':100,
                              'embedding_model':'DistMult',
                              'literals':literals,
                              'literal_activation':'tanh',
                              'loss_function':'pairwize_hinge',
                              'margin':1})
   
    model.fit(DataGenerator(train,batch_size=bs),
             validation_data=DataGenerator(valid,batch_size=bs),
             epochs=100,
             batch_size=bs,
             verbose=2)
             
                          
             
if __name__ == '__main__':
    main()
