# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:39:21 2023

@author: rud83
"""

import data
import knn
import ml
import vis

# setting file
config = data.load_config("./example/config/mp_example.json")

# gen dataset
data.element_set_train_test_split(config)
knn.gen_rn_training_data(config)

# train
ml.train_all_config(config)

# visualize
vis.visualize_test_prediction(config)
vis.gen_heatmap_all(config)

# save result
vis.save_vals_all(config)