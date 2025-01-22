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
settings = data.load_settings("./example/config/mp_example.json")

# gen dataset
data.element_set_train_test_split(settings)
knn.gen_rn_training_data(settings)

# train
ml.train_all_settings(settings)

# visualize
vis.visualize_test_prediction(settings)
vis.gen_heatmap_all(settings)

# save result
vis.save_vals_all(settings)