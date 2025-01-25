# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:39:21 2023

@author: rud83
"""

import sys

import data
import knn
import ml
import vis_and_save

args = sys.argv
if len(args) < 2:
    print("Please set config file.")
    sys.exit(1)
elif len(args) > 2:
    print("Too many arguments. Not first arguments will be ignored.")

config_file = args[1]
# setting file
config = data.load_config(config_file)

# gen dataset
data.element_set_train_test_split(config)
knn.gen_rn_training_data(config)

# train
ml.train_all(config)

# visualize
vis_and_save.visualize_test_prediction(config)
vis_and_save.gen_heatmap_all(config)

# save result
vis_and_save.save_vals_all(config)