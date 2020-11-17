
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata
from random import choice
from collections import defaultdict

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
from random import choices
EPSILON = 1e-6

from rdflib import Graph, URIRef, Literal, Namespace
import rdflib
from rdflib.namespace import XSD, RDF
UNIT = Namespace('http://qudt.org/vocab/unit#')
from tqdm import tqdm

import spacy
VEC_SIZE = 300

def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

class LiteralConverter:
    def __init__(self,g,padding_value=0):
        self.g = g
        self.non_literal_entities = set(g.subjects()) | set([o for o in g.objects() if isinstance(o,URIRef)])
        self.literal_predicates = set([p for p,o in g.predicate_objects() if isinstance(o,Literal)])
        self.padding_value = padding_value
        self.lang_models = {'xx':spacy.load('xx_ent_wiki_sm'),'en':spacy.load('en_core_web_md')}

    def _process_string_literal(self,x):
        doc = self.lang_models['en'](str(x))
        
        v = doc.vector
        if len(v) < 1:
            v = self.padding_value*np.ones((VEC_SIZE,))
        return v

    def _process_literal(self,x):
        if hasattr(x,'datatype') and (x.datatype == XSD['float'] or x.datatype == XSD['double']):
            return [float(x)]
        
        if hasattr(x,'datatype') and x.datatype == XSD['date']:
            return URIRef('http://examples.org/date/%s' % str(x))
        
        if hasattr(x,'datatype') and x.datatype == XSD['boolean']:
            return [1] if bool(x) else [0]
        
        if len(str(x)) == 4 and isint(x):
            return URIRef('http://examples.org/date/%s' % str(x))
        
        if hasattr(x,'datatype') and (x.datatype is None or x.datatype == XSD['string']):
            return self._process_string_literal(x)
        
        return None

    def fit(self):
        out = defaultdict(dict)
        vec_or_num = {}
        array_ps = set()
        for i,e in tqdm(enumerate(self.non_literal_entities),total=len(self.non_literal_entities),desc='Processing literals'):
            for j,p in enumerate(self.literal_predicates):
                tmp = set(self.g.objects(subject = e, predicate = p / RDF.value)) | set(self.g.objects(subject = e, predicate = p))
                unit = set(self.g.objects(subject = e, predicate = p / UNIT.units))
                
                for t in tmp:
                    t = self._process_literal(t)
                    if t is None: 
                        continue
                    elif isinstance(t,URIRef):
                        self.g.add((e,p,t))
                    else:
                        out[p][e] = t
                        if p not in vec_or_num: vec_or_num[p] = len(t)
        
        s=sum(i for k,i in vec_or_num.items())
        self.literals = {}
        for e in self.non_literal_entities:
            tmp = []
            for p in self.literal_predicates:
                if not p in vec_or_num: continue
            
                if e in out[p]:
                    tmp.append(np.asarray(out[p][e]).reshape((1,-1)))
                else:
                    tmp.append(self.padding_value*np.ones((1,vec_or_num[p])))
            tmp = np.concatenate(tmp,axis=1).reshape((-1,))
            assert len(tmp) == s
            self.literals[e] = tmp
            
    def transform(self,entities):
        return np.asarray([self.literals[e] for e in entities])
    
    def fit_transform(self,entities):
        if not hasattr(self,'literals'):
            self.fit()
        return self.transform(entities)

def load_kg(path):
    out = []
    with open(path,'r') as f:
        for l in f:
            l = l.strip().split()
            out.append(l)
    return out

def generate_negative(kg, N, negative=2, check_kg=False, corrupt_head=True, corrupt_tail=True):
    # false triples:
    assert corrupt_head or corrupt_tail
    R = np.repeat(np.asarray([p for _,p,_ in kg]).reshape((-1,1)),negative,axis=0)
    fs = np.random.randint(0,N,size=(negative*len(kg),1))  
    fo = np.random.randint(0,N,size=(negative*len(kg),1))  
    negative_kg = np.stack([fs,R,fo],axis=1)
    return negative_kg

def oversample_data(kgs,x=None,y=None,testing=False):
    if testing:
        kgs = [list(kg)[:len(y)] for kg in kgs]
    else:
        kgs = [list(kg) for kg in kgs]
        
    if y is not None:
        m = max(max(map(len,kgs)),len(y))
    else:
        m = max(map(len,kgs))
    
    out = []
    for kg in kgs:
        out.append(choices(kg, k=m))
    
    if x is not None and y is not None:
        k = np.ceil(m/len(y))
        y = np.repeat(y,k,axis=0)[:m]
        x = np.repeat(x,k,axis=0)[:m,:]
        for s in np.split(x,3,axis=1):
            out.append(s.reshape((-1,)))
        return [np.squeeze(np.asarray(o)) for o in out], np.asarray(y)
    
    else:
        return [np.squeeze(np.asarray(o)) for o in out]

def pad(kg,bs):
    kg = list(kg)
    while len(kg) % bs != 0:
        kg.append(choice(kg))
    return np.asarray(kg)
        
def mrr(target, scores):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    labels = [x for x,_ in scores]
    return 1/(1+labels.index(target))

def hits(target, scores, k=10):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    labels = [x for x,_ in scores][:k]
    return int(target in labels)

def gen_tail_data(test_data,num_entities,bs,filter_t):
    
    for s,p,o in test_data:
        candiate_objects = list(range(num_entities))
        candiate_objects.remove(o)
        for oi in filter_t[(s,p)]:
            candiate_objects.remove(oi)
                    
        subjects = np.asarray([[int(s)]]*(len(candiate_objects)+1))
        predicates = np.asarray([[int(p)]]*(len(candiate_objects)+1))
        objects = np.asarray([[int(o)]] + [[ent_id] for ent_id in candiate_objects])
        
        triples = np.concatenate((subjects,predicates,objects),axis=-1)
        
        yield triples.reshape((-1,3))
        
def gen_head_data(test_data,num_entities,bs,filter_h):
    
    for s,p,o in test_data:
        candiate_subjects = list(range(num_entities))
        candiate_subjects.remove(s)
        
        for si in filter_h[(p,o)]:
            candiate_subjects.remove(si)
                    
        objects = np.asarray([[int(o)]]*(len(candiate_subjects)+1))
        predicates = np.asarray([[int(p)]]*(len(candiate_subjects)+1))
        subjects = np.asarray([[int(s)]] + [[ent_id] for ent_id in candiate_subjects])
        
        triples = np.concatenate((subjects,predicates,objects),axis=-1)
        
        yield triples.reshape((-1,3))
        
        
def validate(model, test_data, num_entities, bs, filtering_triples = None):
    
    filter_h = defaultdict(set)
    filter_t = defaultdict(set)
    for s,p,o in filtering_triples:
        filter_h[(p,o)].add(s)
        filter_t[(s,p)].add(o)
    
    c_1, c_3, c_10 = 0,0,0
    mean_ranks = []

    for t in tqdm(gen_tail_data(test_data,num_entities,bs,filter_t),total=len(test_data)):
        res = np.asarray(model.predict(t)).reshape((-1,))
        r = rankdata(res,'max')
        target_rank = r[0]
        num_candidate = len(res)
        real_rank = num_candidate - target_rank + 1
        c_1 += 1 if target_rank == num_candidate else 0
        c_3 += 1 if target_rank + 3 > num_candidate else 0
        c_10 += 1 if target_rank + 10 > num_candidate else 0
        mean_ranks.append(real_rank)
        
    tail_hit_at_1 = c_1 / float(len(test_data))
    tail_hit_at_3 = c_3 / float(len(test_data))
    tail_hit_at_10 = c_10 / float(len(test_data))
    tail_avg_rank = np.mean(mean_ranks)
    tail_mrr = np.mean([1/m for m in mean_ranks])
    
    c_1, c_3, c_10 = 0,0,0
    mean_ranks = []
    
    for t in tqdm(gen_head_data(test_data,num_entities,bs,filter_h),total=len(test_data)):
        res = np.asarray(model.predict(t)).reshape((-1,))
        r = rankdata(res,'max')
        target_rank = r[0]
        num_candidate = len(res)
        real_rank = num_candidate - target_rank + 1
        c_1 += 1 if target_rank == num_candidate else 0
        c_3 += 1 if target_rank + 3 > num_candidate else 0
        c_10 += 1 if target_rank + 10 > num_candidate else 0
        mean_ranks.append(real_rank)
        
    head_hit_at_1 = c_1 / float(len(test_data))
    head_hit_at_3 = c_3 / float(len(test_data))
    head_hit_at_10 = c_10 / float(len(test_data))
    head_avg_rank = np.mean(mean_ranks)
    head_mrr = np.mean([1/m for m in mean_ranks])
        
    metrics = {'tail_hits@1':tail_hit_at_1,
               'tail_hits@3':tail_hit_at_3,
               'tail_hits@10':tail_hit_at_10,
               'tail_mr':tail_avg_rank,
               'tail_mrr':tail_mrr,
               'head_hits@1':head_hit_at_1,
               'head_hits@3':head_hit_at_3,
               'head_hits@10':head_hit_at_10,
               'head_mr':head_avg_rank,
               'head_mrr':head_mrr,
               'hits@1':(tail_hit_at_1+head_hit_at_1)/2,
               'hits@3':(tail_hit_at_3+head_hit_at_3)/2,
               'hits@10':(tail_hit_at_10+head_hit_at_10)/2,
               'mr':(tail_avg_rank+head_avg_rank)/2,
               'mrr':(tail_mrr+head_mrr)/2,
               }
    
    return metrics

        
class KGEValidateCallback(Callback):
    def __init__(self, validation_data, train_data=None, *args, **kwargs):
        super(Callback, self).__init__(*args, **kwargs)
        self.validation_data = validation_data
        self.train_data = train_data
        
    def on_epoch_end(self, epoch, logs = None):
        if epoch % 5 == 0:
            logs = logs or {}
            tmp = validate(self.model, 
                            self.validation_data,
                            self.model.num_entities,
                            self.train_data)
                
            for k in tmp:
                logs['val_'+k] = tmp[k]
                
    def on_train_end(self, logs=None):
        self.on_epoch_end(100,logs=logs)
        

def pointwize_hinge(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return tf.reduce_mean(tf.nn.relu(margin-true))+tf.reduce_mean(tf.nn.relu(margin+false))

def pointwize_logistic(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return tf.reduce_mean(tf.math.log(EPSILON+1+tf.math.exp(-true)))+tf.reduce_mean(tf.math.log(EPSILON+1+tf.math.exp(false)))

def pointwize_square_loss(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return tf.reduce_mean(tf.square(margin-true))+tf.reduce_mean(tf.square(margin+false))

def pointwize_cross_entropy(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return binary_crossentropy(1,true)+binary_crossentropy(0,false)

def pairwize_hinge(true,false,margin=1, negative_samples=1, reduce_mean = True):
    false = tf.reshape(false,(-1,negative_samples))
    tmp = tf.nn.relu(margin+false-true)
    if reduce_mean:
        return tf.reduce_mean(tmp)
    return tmp

def pairwize_logistic(true,false,margin=0, negative_samples=1, reduce_mean = True):
    false = tf.reshape(false,(-1,negative_samples))
    tmp = tf.math.log(EPSILON+1+tf.math.exp(false-true))
    if reduce_mean:
        return tf.reduce_mean(tmp) 
    return tmp

def pairwize_square_loss(true,false,margin=0, negative_samples=1, reduce_mean = True):
    false = tf.reshape(false,(-1,negative_samples))
    tmp = - tf.square(false-true)
    if reduce_mean:
        return tf.reduce_mean(tmp)
    return tmp

def loss_function_lookup(name):
    return {
    'pointwize_hinge':pointwize_hinge,
    'pointwize_logistic':pointwize_logistic,
    'pointwize_cross_entropy':pointwize_cross_entropy,
    'pointwize_square_loss':pointwize_square_loss,
    'pairwize_hinge':pairwize_hinge,
    'pairwize_logistic':pairwize_logistic,
    'pairwize_square_loss':pairwize_square_loss
    }[name]

