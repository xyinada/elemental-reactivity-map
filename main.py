# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:39:21 2023

@author: rud83
"""

import data
import knn
import ml
import vis

#name = "supercond_ht"
for name in ["qct\\bergman",
             "qct\\mackay", "qct\\tsai"]:
    print("Target : {}".format(name))
    settings_path = r"E:\dresearch\material_search_map_property\{}\settings.csv".format(name)
    settings = data.load_settings(settings_path)
    
    data.element_set_train_test_split(settings)
    knn.gen_rn_training_data(settings)
    ml.train_all_settings(settings)
    vis.visualize_test_prediction(settings)
    vis.gen_heatmap_all(settings)
    vis.save_vals_all(settings)