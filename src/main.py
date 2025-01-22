# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:39:21 2023

@author: rud83
"""

import data
import knn
import ml
import vis_and_save

# setting file
config = data.load_config("./example/config/mp_example.json")

# gen dataset
data.element_set_train_test_split(config)
knn.gen_rn_training_data(config)

# train
ml.train_all_config(config)

# visualize
vis_and_save.visualize_test_prediction(config)
vis_and_save.gen_heatmap_all(config)

# save result
vis_and_save.save_vals_all(config)