
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata
from random import choice
from collections import defaultdict

from tensorflow.keras.callbacks import Callback

def load_kg(path):
    out = []
    with open(path,'r') as f:
        for l in f:
            l = l.strip().split()
            out.append(l)
    return out

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
        

from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
EPSILON = 1e-6

def pointwize_hinge(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return tf.reduce_mean(tf.nn.relu(margin-true))+tf.reduce_mean(tf.nn.relu(margin+false))

def pointwize_logistic(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return tf.reduce_mean(tf.math.log(EPSILON+1+tf.math.exp(-true)))+tf.reduce_mean(tf.math.log(EPSILON+1+tf.math.exp(false)))

def pointwize_square_loss(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return tf.reduce_mean(tf.square(margin-true))+tf.reduce_mean(tf.square(margin+false))

def pointwize_cross_entropy(true,false,margin=1,negative_samples=1, reduce_mean = True):
    return binary_crossentropy(1,true)+binary_crossentropy(0,false)

def pairwize_hinge(true,false,margin=1, negative_samples=1, reduce_mean = True):
    tmp = tf.nn.relu(margin+false-tf.tile(true,[negative_samples,1]))
    if reduce_mean:
        return tf.reduce_mean(tmp)
    return tmp

def pairwize_logistic(true,false,margin=0, negative_samples=1, reduce_mean = True):
    tmp = tf.math.log(EPSILON+1+tf.math.exp(false-tf.tile(true,[negative_samples,1])))
    if reduce_mean:
        return tf.reduce_mean(tmp)
    return tmp

def pairwize_square_loss(true,false,margin=0, negative_samples=1, reduce_mean = True):
    tmp = - tf.square(false-tf.tile(true,[negative_samples,1]))
    if reduce_mean:
        return tf.reduce_mean(tmp)
    return tmp

def loss_function_lookup(name):
    return {
    'pointwize_hinge':pointwize_hinge,
    'pointwize_logistic':pointwize_logistic,
    'pointwize_cross_entropy':pointwize_cross_entropy,
    'pointwize_square_loss':,
    'pairwize_hinge':pairwize_hinge,
    'pairwize_logistic':pairwize_logistic,
    'pairwize_square_loss':pairwize_square_loss
    }[name]

