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
#from numba.decorators import jit
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

def load_settings(path):
    print("Load settings.")
    with open(path, "r") as f:
        lines = f.readlines()
    ret = {}
    list_data = ["knn_k","rn_lower_ratio",
                 "learning_rate", "batch_size", "epoch", "node_nums",
                 "heatmap_name", "heatmap_source", "heatmap_composition_column",
                 "heatmap_property_column", "heatmap_property_selection_method",
                 "heatmap_property_selection_ratio", "heatmap_property_min",
                 "heatmap_property_max", "heatmap_property_name",
                 "heatmap_mark", "heatmap_mark_labels", "heatmap_color"]
    for line in lines:
        l = line.strip().split(",")
        while l[-1] == "":
            l = l[:-1]
        if len(l) == 1:
            ret[l[0]] = False
        elif l[0] in list_data:
            ret[l[0]] = l[1:]
        else:
            ret[l[0]] = l[1]
    return ret

def element_set_train_test_split(settings):
    print("Make training and test data.")
    elemsets = load_element_set_data(settings)
    select_test_data(elemsets, settings)

def load_element_set_data(settings):
    path = settings["source_folder"]+"\\"+settings["source_file"]+".csv"
    with open(settings["save_folder"]+r"\about_data.csv", "w") as f:
        df = pd.read_csv(path)
        f.write("source_data_num, {}\n".format(len(df)))
        df = df.dropna(subset=[settings["property_column"], settings["composition_column"]])
        f.write("property_data_num, {}\n".format(len(df)))
        if settings["property_selection_method"] == "range":
            df = df[df[settings["property_column"]] >= float(settings["property_min"])]
            df = df[df[settings["property_column"]] <= float(settings["property_max"])]
        elif settings["property_selection_method"] == "high":
            df = df.sort_values(settings["property_column"], ascending=False)
            n = int(len(df)*float(settings["property_selection_ratio"]))
            df = df.head(n)
        elif settings["property_selection_method"] == "low":
            df = df.sort_values(settings["property_column"], ascending=True)
            n = int(len(df)*float(settings["property_selection_ratio"]))
            df = df.head(n)
        elif settings["property_selection_method"] == "name":
            df = df[df[settings["property_column"]]==settings["property_name"]]
        f.write("property_data_in_range_num, {}\n".format(len(df)))
        elems = set(load_element_parameter_from_settings(settings).keys())
        cmps = df[settings["composition_column"]]
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

def load_compounds_data(settings, source_file, composition_column, property_column, method,
                        property_min=0, property_max=0, property_name="", property_selection_ratio=0):
    path = settings["source_folder"]+"\\"+source_file+".csv"
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
    elems = set(load_element_parameter_from_settings(settings).keys())
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

def select_test_data(elemsets, settings):
    folder = settings["save_folder"]
    elems = list(load_element_parameter_from_settings(settings).keys())
    
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
    if settings["positive_test_data"] == "num":
        positive_test_num = int(settings["positive_test_data_num"])
    elif settings["positive_test_data"] == "ratio":
        positive_test_num = int(len(positive_elemsets)*float(settings["positive_test_data_ratio"]))
    unlabeled_test_num = int(settings["unlabeled_test_data_num"])
    if not "positive_train_data_num" in settings:
        positive_train_num = len(positive_elemsets)-positive_test_num
    else:
        positive_train_num = int(settings["positive_train_data_num"])
    positive_test = positive_elemsets[:positive_test_num]
    positive_train = positive_elemsets[positive_test_num:positive_test_num+positive_train_num]
    unlabeled_test = unlabeled_elemsets[:unlabeled_test_num]
    unlabeled_train = unlabeled_elemsets[unlabeled_test_num:]
    
    print(len(positive_train))
    print(len(unlabeled_train)+len(unlabeled_test))
    print(len(positive_elemsets)+len(unlabeled_test)+len(unlabeled_train))
    try:
        os.mkdir(folder+"\\data")
    except:
        pass
    
    with open(folder+"\\data\\positive_all.pkl", "wb") as f:
        pickle.dump(positive_elemsets, f)
    with open(folder+"\\data\\positive_test.pkl", "wb") as f:
        pickle.dump(positive_test, f)
    with open(folder+"\\data\\positive_train.pkl", "wb") as f:
        pickle.dump(positive_train, f)
    with open(folder+"\\data\\unlabeled_all.pkl", "wb") as f:
        pickle.dump(unlabeled_elemsets, f)
    with open(folder+"\\data\\unlabeled_test.pkl", "wb") as f:
        pickle.dump(unlabeled_test, f)
    with open(folder+"\\data\\unlabeled_train.pkl", "wb") as f:
        pickle.dump(unlabeled_train, f)
    with open(settings["save_folder"]+r"\about_data.csv", "a") as f:
        f.write("positive_test_data_num,"+str(len(positive_test)))

def load_element_parameter_from_settings(settings):
    element_parameter_paths = [settings["element_parameter_folder"]+"\\"+epn+".csv" for epn in settings["element_parameter"].split("-")]
    element_parameter = load_element_parameters(element_parameter_paths, [int(pn) for pn in settings["parameter_num"].split("-")])
    return element_parameter

def load_element_parameter(path,parameter_nums):
    data_dict = {}
    f = open(path,"r")
    reader = csv.reader(f)
    _ = next(reader)
    i = 0
    for row in reader:
        data_dict[row[0]] = list(map(float,row[1:parameter_nums[i]+1]))
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
