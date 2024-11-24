# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:16:46 2023

@author: rud83
"""

import copy
import pickle
import os
import itertools
import random
import tqdm

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import data

def gen_rn_training_data(settings):
    print("Calculate unlabeled data similarity to positive data.")
    calc_unlabeled_similarity_with_positive(settings)
    print("Select reliable negative data.")
    knn_selection(settings)

def knn_selection(settings):
    folder = settings["save_folder"]
    for knn_k in [int(knn_k) for knn_k in settings["knn_k"]]:
        positively_scores = calc_positively_scores(knn_k, settings)
        for rn_lower_ratio in [float(r) for r in settings["rn_lower_ratio"]]:
            n = int(len(positively_scores)*rn_lower_ratio)
            idxs = random.sample(list(range(n)), int(settings["rn_train_data_num"]))
            train_data = [positively_scores[i][0] for i in idxs]
            with open(folder+r"\data\rn_knn-k={}_rn-lower-ratio={}_train.pkl".format(knn_k, rn_lower_ratio), "wb") as f:
                pickle.dump(train_data, f)

def calc_positively_scores(knn_k, settings):
    folder = settings["save_folder"]
    with open(folder+r"\data\unlabeled_similarity_with_positive_sorted.pkl", "rb") as f:
        similarity_data = pickle.load(f)
    ret = []
    for k, vs in similarity_data.items():
        ret.append((k, sum(vs[:knn_k])/knn_k))
    ret = sorted(ret, key=lambda x:x[1])
    return ret

def calc_unlabeled_similarity_with_positive(settings):
    folder = settings["save_folder"]
    if not "positive_train_data_num" in settings:
        with open(folder+"\\data\\positive_train.pkl", "rb") as f:
            positive_train = pickle.load(f)
    else:
        with open(folder+"\\data\\positive_test.pkl", "rb") as f:
            pote = set(pickle.load(f))
        with open(folder+"\\data\\positive_all.pkl", "rb") as f:
            poal = pickle.load(f)
        positive_train = [pa for pa in poal if not pa in pote]
        random.shuffle(positive_train)
    with open(folder+"\\data\\unlabeled_train.pkl", "rb") as f:
        unlabeled_train = pickle.load(f)
    
    max_k = max([int(k) for k in settings["knn_k"]])
    
    elem_parameter = data.load_element_parameter_from_settings(settings)
    sim_data, elem_rank_dict = calc_element_similarity(elem_parameter)

    elem_list = list(elem_parameter.keys())
    elem_num = len(elem_list)
    sim_array = [0.]*(elem_num*elem_num)
    elem_dict = {}
    for i in range(elem_num):
        elem_dict[elem_list[i]] = i
        for j in range(elem_num):
            sim_array[elem_num*i+j] = sim_data[(elem_list[i],elem_list[j])]
    
    p_train_idx = []
    for tt in positive_train:
        p_train_idx.append(tuple([elem_dict[t] for t in tt]))
    
    unlabeled_similarity_scores = calc_similarity_scores_sorted(unlabeled_train, p_train_idx, sim_array, elem_dict, max_k, elem_num)
    with open(folder+r"\data\unlabeled_similarity_with_positive_sorted.pkl", "wb") as f:
        pickle.dump(unlabeled_similarity_scores, f)

def calc_unlabeld_test_similarity_with_positive_train(settings):
    folder = settings["save_folder"]
    if not "positive_train_data_num" in settings:
        with open(folder+"\\data\\positive_train.pkl", "rb") as f:
            positive_train = pickle.load(f)
    else:
        with open(folder+"\\data\\positive_test.pkl", "rb") as f:
            pote = set(pickle.load(f))
        with open(folder+"\\data\\positive_all.pkl", "rb") as f:
            poal = pickle.load(f)
        positive_train = [pa for pa in poal if not pa in pote]
        random.shuffle(positive_train)
    with open(folder+"\\data\\unlabeled_test.pkl", "rb") as f:
        unlabeled_train = pickle.load(f)
    
    max_k = max([int(k) for k in settings["knn_k"]])
    
    elem_parameter = data.load_element_parameter_from_settings(settings)
    sim_data, elem_rank_dict = calc_element_similarity(elem_parameter)

    elem_list = list(elem_parameter.keys())
    elem_num = len(elem_list)
    sim_array = [0.]*(elem_num*elem_num)
    elem_dict = {}
    for i in range(elem_num):
        elem_dict[elem_list[i]] = i
        for j in range(elem_num):
            sim_array[elem_num*i+j] = sim_data[(elem_list[i],elem_list[j])]
    
    p_train_idx = []
    for tt in positive_train:
        p_train_idx.append(tuple([elem_dict[t] for t in tt]))
    
    unlabeled_similarity_scores = calc_similarity_scores_sorted(unlabeled_train, p_train_idx, sim_array, elem_dict, max_k, elem_num)
    with open(folder+r"\data\unlabeled_test_similarity_with_positive_train_sorted.pkl", "wb") as f:
        pickle.dump(unlabeled_similarity_scores, f)

def calc_element_similarity(element_parameter):
    es = list(element_parameter.keys())
    d = {}
    ret_sim = {}
    ret_rank = {}
    for e1 in es:
        d[e1] = []
        for e2 in es:
            d[e1].append((e2,cosine_similarity(element_parameter[e1],element_parameter[e2])))
        d[e1] = sorted(d[e1],key=lambda x:x[1])
        for i in range(len(d[e1])):
            ret_sim[(e1,d[e1][i][0])] = d[e1][i][1]
            ret_rank[(e1,d[e1][i][0])] = float(i + 1)
    for k in ret_sim.keys():
        if ret_sim[k] > ret_sim[(k[1],k[0])]:
            ret_sim[(k[1],k[0])] = ret_sim[k]
    for k in ret_rank.keys():
        if ret_rank[k] > ret_rank[(k[1],k[0])]:
            ret_rank[(k[1],k[0])] = ret_rank[k]
    return ret_sim, ret_rank

def cosine_similarity(u,v):
    return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def calc_similarity_scores_sorted(unlabeled_data,train_data,sim_array,elem_dict,max_k,elem_num):
    ret = {}
    for uld in tqdm.tqdm(unlabeled_data):
        ul = tuple([elem_dict[ccc] for ccc in uld])
        sims = [calc_dist(ul,td,sim_array,elem_num) for td in train_data]
        sims = sorted(sims,reverse=True)
        ret[uld] = copy.deepcopy(sims[:min(max_k,len(sims))])
    return ret

def calc_dist(aa, bb, sim_array, elem_num):
    a = tuple(set(aa))
    b = tuple(set(bb))
    if len(a) == 3 and len(b) == 3:
        l = [(b[0],b[1],b[2]),(b[0],b[2],b[1]),(b[1],b[0],b[2]),(b[1],b[2],b[0]),(b[2],b[0],b[1]),(b[2],b[1],b[0])]
        x = max([sim_array[calc_idx(m[0],a[0],elem_num)]+sim_array[calc_idx(m[1],a[1],elem_num)]+sim_array[calc_idx(m[2],a[2],elem_num)] for m in l])
        #y = max([sim_data[(a[0],m[0])]+sim_data[(a[1],m[1])]+sim_data[(a[2],m[2])] for m in l])
        return x
    elif len(a) == 2 and len(b) == 2:
        l = [(b[0],b[1]),(b[1],b[0])]
        x = max([sim_array[calc_idx(m[0],a[0],elem_num)]+sim_array[calc_idx(m[1],a[1],elem_num)] for m in l])
        #y = max([sim_data[(a[0],m[0])]+sim_data[(a[1],m[1])] for m in l])
        return x*1.5
    elif len(a) == 1 and len(b) == 1:
        return sim_array[calc_idx(a[0],b[0],elem_num)]*3.
    else:
        return 0.
        
def calc_idx(a, b, elem_num):
    return a*elem_num+b

