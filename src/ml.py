# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:24:08 2023

@author: rud83
"""

import csv
import os
import pickle
import random
import warnings
import collections
import copy
import itertools
import glob
import tqdm

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
import pymatgen.core.composition as pcmp

import data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

def build_model(node_nums,vec_len,l_rate):
    input_data = tf.keras.layers.Input(shape=(3,vec_len),name="input")
    input_data_reshape = tf.reshape(input_data,[-1,vec_len])
    layer0 = tf.keras.layers.Dense(node_nums[0],activation="relu",name="layer0")(input_data_reshape)
    layer1 = tf.keras.layers.Dense(node_nums[1],activation="relu",name="layer1")(layer0)
    layer1 = tf.reshape(layer1,[-1,3,node_nums[1]])
    layer1_maxpool = tf.reshape(tf.reduce_max(layer1,1),(-1,node_nums[1]))
    layer2 = tf.keras.layers.Dense(node_nums[2],activation="relu",name="layer2")(layer1_maxpool)
    layer3 = tf.keras.layers.Dense(node_nums[3],activation="sigmoid",name="layer3")(layer2)
    model = tf.keras.Model(inputs=input_data,outputs=layer3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
                   loss=tf.keras.losses.BinaryCrossentropy())
    return model

def train_all_config(config):
    print("Train models.")
    iters = itertools.product([int(k) for k in  config["data"]["parameters"]["knn_k"]],
                              [float(rn_lower_ratio) for rn_lower_ratio in  config["data"]["parameters"]["rn_lower_ratio"]],
                              [float(l_rate) for l_rate in config["learning"]["learning_rate"]],
                              [int(batch_size) for batch_size in config["learning"]["batch_size"]],
                              [int(epoch) for epoch in config["learning"]["epoch"]],
                              [node_nums for node_nums in config["learning"]["node_nums"]])
    iters = list(iters)
    for knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums in tqdm.tqdm(iters):
        train_and_evaluate(config, knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums)

def train_and_evaluate(config, knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums):
    model_folder = config["files_and_directories"]["save_dir"]+"/ml_model/knn-k={}_rnlr={}_lr={}_bs={}_ep={}_nn={}".format(knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums)
    print(model_folder)
    os.mkdirs(model_folder, exist_ok=True)
    input_data = make_input_data(config, knn_k, rn_lower_ratio)
    train_models(config, model_folder, l_rate, batch_size, epoch, node_nums, input_data)
    evaluate_models(config, model_folder, input_data)

def train_models(config, model_folder, l_rate, batch_size, epoch, node_nums, input_data):
    vec_len = sum([int(x) for x in config["data"]["parameters"]["element_parameter"]["dim"]])
    for i in range(int(config["learning"]["model_num"])):
        model = build_model([int(nns) for nns in node_nums.split("-")], vec_len, l_rate)
        history = model.fit(input_data["train_x"],input_data["train_y"],
                            validation_data=[input_data["test_x"],input_data["test_y"]],
                            batch_size=batch_size, epochs=epoch, verbose=0)
        model.save(model_folder+"/model_{}.tf".format(i))
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(model_folder+'/history_{}.csv'.format(i))
        tf.keras.backend.clear_session()
        del model

def element_set_to_input(ess, config):
    elem_parameter = data.load_element_parameter_from_config(config)
    ret = []
    for es in ess:
        e = list(es)
        while len(e) < 3:
            e.append(e[-1])
        ret.append([elem_parameter[x] for x in e])
    ret = np.array(ret, dtype=np.float32)
    return ret

def make_input_data(config, knn_k, rn_lower_ratio):
    folder = config["files_and_directories"]["save_dir"]
    with open(folder+"/train_and_test_data/positive_train.pkl", "rb") as f:
        positive_train = pickle.load(f)
    with open(folder+"/train_and_test_data/rn_knn-k={}_rn-lower-ratio={}_train.pkl".format(knn_k,rn_lower_ratio), "rb") as f:
        rn_train = pickle.load(f)
    with open(folder+"/train_and_test_data/positive_test.pkl", "rb") as f:
        positive_test = pickle.load(f)
    with open(folder+"/train_and_test_data/unlabeled_test.pkl", "rb") as f:
        unlabeled_test = pickle.load(f)
    
    if config["data"]["data_num"]["positive_train_data_dup"] == "auto":
        random.shuffle(positive_train)
        idx = 0
        orig_poslen = len(positive_train)
        while len(positive_train) < len(rn_train):
            positive_train.append(positive_train[idx])
            idx = (idx+1)%orig_poslen
    else:
        positive_train = positive_train*int(config["data"]["data_num"]["positive_train_data_dup"])

    input_data = {}
    input_data["train_x"] = np.vstack([element_set_to_input(positive_train, config),
                                       element_set_to_input(rn_train, config)])
    input_data["train_y"] = np.vstack([np.ones(shape=(len(positive_train),1)),
                                       np.zeros(shape=(len(rn_train),1))])
    input_data["test_x"] = np.vstack([element_set_to_input(positive_test, config),
                                      element_set_to_input(unlabeled_test, config)])
    input_data["test_y"] = np.vstack([np.ones(shape=(len(positive_test),1)),
                                      np.zeros(shape=(len(unlabeled_test),1))])
    return input_data

def predict(model_folder, xs):
    model_paths = glob.glob(model_folder+r"\\*.tf")
    preds = np.zeros(shape=(len(xs),))
    for path in model_paths:
        model = tf.keras.models.load_model(path)
        ps = np.squeeze(model(xs).numpy())
        preds += ps
    preds /= len(model_paths)
    preds = np.clip(a=preds, a_min=0., a_max=1.)
    return preds

def evaluate_models(model_folder, input_data):
    ths = [t/10 for t in range(1,10)]
    positive_overthreshold = {}
    unlabeled_overthreshold = {}
    ps = predict(model_folder, input_data["test_x"])
    ys = np.squeeze(input_data["test_y"])
    
    positive_ps = sorted([ps[i] for i in range(len(ps)) if ys[i] == 1.])
    unlabeled_ps = sorted([ps[i] for i in range(len(ps)) if ys[i] == 0.])
    
    for th in ths:
        positive_overthreshold[th] = sum([p >= th for p in positive_ps])/len(positive_ps)
        unlabeled_overthreshold[th] = sum([p >= th for p in unlabeled_ps])/len(unlabeled_ps)
    
    with open(model_folder+"/sorted_predicted_value_for_positive_test.pkl", "wb") as f:
        pickle.dump(positive_ps, f)
    with open(model_folder+"/sorted_predicted_value_for_unlabeled_test.pkl", "wb") as f:
        pickle.dump(unlabeled_ps, f)
    with open(model_folder+"/over_threshold_nums_for_positive_test.pkl", "wb") as f:
        pickle.dump(positive_overthreshold, f)
    with open(model_folder+"/over_threshold_nums_for_unlabeled_test.pkl", "wb") as f:
        pickle.dump(unlabeled_overthreshold, f)

