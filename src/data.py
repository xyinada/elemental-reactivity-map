# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:10:29 2023

@author: rud83
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 01:04:36 2021

@author: rud83
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
import os
import pickle
import random
import copy
import warnings
import collections
import pymatgen.core.composition as pcmp
import glob
import json
import pandas as pd

random.seed(0)

def load_config(path):
    print("Load config.")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def element_set_train_test_split(config):
    print("Make training and test data.")
    elemsets = load_element_set_data(config)
    select_test_data(elemsets, config)

def load_element_set_data(config):
    path = config["files_and_directories"]["data_source_file"]
    os.makedirs(config["files_and_directories"]["save_dir"], exist_ok=True)
    with open(config["files_and_directories"]["save_dir"]+"/about_data.csv", "w") as f:
        df = pd.read_csv(path)
        f.write("source_data_num, {}\n".format(len(df)))
        df = df.dropna(subset=[config["condition"]["property_column_name"], config["condition"]["composition_column_name"]])
        f.write("property_data_num, {}\n".format(len(df)))
        if config["condition"]["property_selection_method"] == "range":
            df = df[df[config["condition"]["property_column"]] >= float(config["condition"]["property_filter"]["value_min"])]
            df = df[df[config["condition"]["property_column"]] <= float(config["condition"]["property_filter"]["value_max"])]
        elif config["condition"]["property_selection_method"] == "high":
            df = df.sort_values(config["condition"]["property_column_name"], ascending=False)
            n = int(len(df)*float(config["condition"]["property_filter"]["ratio"]))
            df = df.head(n)
        elif config["condition"]["property_selection_method"] == "low":
            df = df.sort_values(config["condition"]["property_column_name"], ascending=True)
            n = int(len(df)*float(config["condition"]["property_filter"]["ratio"]))
            df = df.head(n)
        elif config["condition"]["property_selection_method"] == "name":
            df = df[df[config["condition"]["property_column_name"]]==config["condition"]["property_filter"]["name"]]
        f.write("property_data_in_range_num, {}\n".format(len(df)))
        elems = set(load_element_parameter_from_config(config).keys())
        cmps = df[config["condition"]["composition_column_name"]]
        cmps = set(cmps.values)
        f.write("unique_composition_num, {}\n".format(len(cmps)))
        ess = set()
        for cmp in cmps:
            try:
                cmpd = pcmp.Composition(cmp).as_dict()
                k = tuple(sorted(list(set(cmpd.keys()))))
                if len(k) <= 0 or 3 < len(k):
                    continue
                for e in k:
                    if not e in elems:
                        break
                else:
                    ess.add(k)
            except:
                pass
        f.write("positive_element_set_num, {}\n".format(len(ess)))
    return list(ess)

def load_compounds_data(config, source_file, composition_column, property_column, method,
                        property_min=0, property_max=0, property_name="", property_selection_ratio=0):
    path = source_file
    df = pd.read_csv(path)
    df = df[df[composition_column] != ""]
    df = df.dropna(subset=[property_column, composition_column])
    if method == "range":
        df = df[df[property_column] >= property_min]
        df = df[df[property_column] <= property_max]
    elif method == "high":
        df = df.sort_values(property_column, ascending=False)
        n = int(len(df)*property_selection_ratio)
        df = df.head(n)
    elif method == "low":
        df = df.sort_values(property_column, ascending=True)
        n = int(len(df)*property_selection_ratio)
        df = df.head(n)
    elif method == "name":
        df = df[df[property_column]==property_name]
    elems = set(load_element_parameter_from_config(config).keys())
    cmps = df[composition_column]
    cmps = set(cmps.values)
    cmp_dict = collections.defaultdict(list)
    for cmp in cmps:
        try:
            cmpd = pcmp.Composition(cmp).as_dict()
            k = tuple(sorted(list(set(cmpd.keys()))))
            if len(k) <= 0 or 3 < len(k):
                continue
            for e in k:
                if not e in elems:
                    break
            else:
                cmp_dict[k].append(pcmp.Composition(cmp).formula.replace(" ", ""))
        except:
            pass
    return cmp_dict

def select_test_data(elemsets, config):
    path = config["files_and_directories"]["save_dir"]
    elems = list(load_element_parameter_from_config(config).keys())
    
    elemsets_set = set(elemsets)
    fins = set()
    unlabeled_elemsets = []
    for a in elems:
        for b in elems:
            for c in elems:
                k = tuple(sorted(list(set([a,b,c]))))
                if k in elemsets_set or k in fins:
                    continue
                unlabeled_elemsets.append(k)
                fins.add(k)
    positive_elemsets = copy.deepcopy(elemsets)
    random.shuffle(positive_elemsets)
    random.shuffle(unlabeled_elemsets)
    if int(config["data"]["data_num"]["positive_test_data"]) >= 1:
        positive_test_num = int(config["data"]["data_num"]["positive_test_data"])
    else:
        positive_test_num = int(len(positive_elemsets)*float(config["data"]["data_num"]["positive_test_data"]))
    unlabeled_test_num = int(config["data"]["data_num"]["unlabeled_test_data"])
    if not "positive_train_data" in config["data"]["data_num"] or int(config["data"]["data_num"]["positive_train_data"]) < 0:
        positive_train_num = len(positive_elemsets)-positive_test_num
    elif int(config["data"]["data_num"]["positive_train_data"]) >= 1:
        positive_train_num = int(config["data"]["data_num"]["positive_train_data"])
    else:
        positive_train_num = int(len(positive_elemsets)*float(config["data"]["data_num"]["positive_train_data"]))
    positive_test = positive_elemsets[:positive_test_num]
    positive_train = positive_elemsets[positive_test_num:positive_test_num+positive_train_num]
    unlabeled_test = unlabeled_elemsets[:unlabeled_test_num]
    unlabeled_train = unlabeled_elemsets[unlabeled_test_num:]

    print("All element set data num:",len(positive_elemsets)+len(unlabeled_test)+len(unlabeled_train))
    print("Positive data num: All-{}, Train-{}, Test{}".format(len(positive_elemsets), len(positive_train), len(positive_test)))
    print("Unlabeled data num: All-{}, Test {}".format(len(unlabeled_train)+len(unlabeled_test), len(unlabeled_test)))
    os.makedirs(config["files_and_directories"]["save_dir"]+"/train_and_test_data", exist_ok=True)
    
    with open(config["files_and_directories"]["save_dir"]+"/train_and_test_data/positive_all.pkl", "wb") as f:
        pickle.dump(positive_elemsets, f)
    with open(config["files_and_directories"]["save_dir"]+"/train_and_test_data/positive_test.pkl", "wb") as f:
        pickle.dump(positive_test, f)
    with open(config["files_and_directories"]["save_dir"]+"/train_and_test_data/positive_train.pkl", "wb") as f:
        pickle.dump(positive_train, f)
    with open(config["files_and_directories"]["save_dir"]+"/train_and_test_data/unlabeled_all.pkl", "wb") as f:
        pickle.dump(unlabeled_elemsets, f)
    with open(config["files_and_directories"]["save_dir"]+"/train_and_test_data/unlabeled_test.pkl", "wb") as f:
        pickle.dump(unlabeled_test, f)
    with open(config["files_and_directories"]["save_dir"]+"/train_and_test_data/unlabeled_train.pkl", "wb") as f:
        pickle.dump(unlabeled_train, f)
    with open(config["files_and_directories"]["save_dir"]+"/about_data.csv", "a") as f:
        f.write("positive_test_data_num,"+str(len(positive_test)))

def load_element_parameter_from_config(config):
    element_parameter_paths = [d["source_file"] for d in config["data"]["parameters"]["element_parameter"]]
    element_parameter_nums = [int(d["dim"]) for d in config["data"]["parameters"]["element_parameter"]]
    element_parameter = load_element_parameters(element_parameter_paths, element_parameter_nums)
    return element_parameter

def load_element_parameter(path,parameter_nums):
    data_dict = {}
    f = open(path,"r")
    reader = csv.reader(f)
    _ = next(reader)
    i = 0
    for row in reader:
        data_dict[row[0]] = list(map(float, row[1:parameter_nums[i]+1]))
    f.close()
    return data_dict

def load_element_parameters(paths,parameter_num):
    ret = collections.defaultdict(list)
    for path in paths:
        dd = load_element_parameter(path,parameter_num)
        for k in dd.keys():
            ret[k].extend(dd[k])
    for k in ret.keys():
        ret[k] = np.array(ret[k])
    return ret
