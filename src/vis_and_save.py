# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:02:04 2023

@author: rud83
"""

import csv
import os
import random
import glob
import json
import collections
import copy
import pickle
import itertools
from collections import defaultdict
import tqdm

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from bokeh.models import BasicTicker, PrintfTickFormatter
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from bokeh.transform import dodge, factor_cmap
from bokeh.io import save, export_png
import pymatgen.core.composition as pcmp
#from mpld3 import show_d3, fig_to_d3, plugins

import warnings

import data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

matplotlib.interactive(False)
def visualize_test_prediction(config):
    print("Visualize prediction result.")
    folder = config["files_and_directories"]["save_dir"]
    iters = itertools.product([float(l_rate) for l_rate in config["learning"]["learning_rate"]],
                              [int(batch_size) for batch_size in config["learning"]["batch_size"]],
                              [int(epoch) for epoch in config["learning"]["epoch"]],
                              [node_nums for node_nums in config["learning"]["node_nums"]])
    ths = [i/10 for i in range(1,10)]
    xs = [float(x) for x in config["data"]["parameters"]["rn_lower_ratio_list"]]
    os.mkdirs(folder+"/fig", exist_ok=True)
    with open(folder+"/fig/result.csv", "w") as f:
        f.write("knn_k,rn_lower_ratio,learning_rate,batch_size,epoch,node_nums,")
        f.write(",".join(["positive_test_over_threshold={}".format(th) for th in ths])+",")
        f.write(",".join(["unlabeled_test_over_threshold={}".format(th) for th in ths])+",")
        f.write("positive_area_under_cumulative_curve, unlabeled_area_under_cumulative_curve\n")
    for l_rate, batch_size, epoch, node_nums in iters:
        for knn_k in [int(k) for k in config["data"]["parameters"]["knn_k_list"]]:
            pospreds_all = []
            unlpreds_all = []
            posposr_all = defaultdict(list)
            unlposr_all = defaultdict(list)
            for rn_lower_ratio in [float(rn_lower_ratio) for rn_lower_ratio in config["data"]["parameters"]["rn_lower_ratio_list"]]:
                model_folder = folder+"/ml_model/knn-k={}_rn-lower-ratio={}_lrate={}_bsize={}_epoch={}_nodenums={}".format(knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums)
                with open(model_folder+"/sorted_predicted_value_for_positive_test.pkl", "rb") as f:
                    pospreds_all.append(pickle.load(f))
                with open(model_folder+"/sorted_predicted_value_for_unlabeled_test.pkl", "rb") as f:
                    unlpreds_all.append(pickle.load(f))
                with open(model_folder+"/over_threshold_nums_for_positive_test.pkl", "rb") as f:
                    posposr = pickle.load(f)
                with open(model_folder+"/over_threshold_nums_for_unlabeled_test.pkl", "rb") as f:
                    unlposr = pickle.load(f)
                for th in ths:
                    posposr_all[th].append(posposr[th])
                    unlposr_all[th].append(unlposr[th])
                with open(folder+"/fig/result.csv", "a") as f:
                    f.write("{},{},{},{},{},{},".format(knn_k,rn_lower_ratio,l_rate,batch_size,epoch,node_nums))
                    f.write(",".join([str(posposr[th]) for th in ths])+",")
                    f.write(",".join([str(unlposr[th]) for th in ths])+",")
                    f.write(str(sum([(pospreds_all[-1][i+1]-pospreds_all[-1][i])*i/len(pospreds_all[-1]) for i in range(len(pospreds_all[-1])-1)]))+",")
                    f.write(str(sum([(unlpreds_all[-1][i+1]-unlpreds_all[-1][i])*i/len(unlpreds_all[-1]) for i in range(len(unlpreds_all[-1])-1)]))+"\n")
            name = "knn-k={}_lrate={}_bsize={}_epoch={}_nodenums={}".format(knn_k, l_rate, batch_size, epoch, node_nums)
            plot_positive_pred_ratio(xs, posposr_all, "positive_test"+name, folder+r"\fig")
            plot_positive_pred_ratio(xs, unlposr_all, "unlabeled_test"+name, folder+r"\fig")
            plot_cumulative(pospreds_all, unlpreds_all,
                            [float(rn_lower_ratio) for rn_lower_ratio in config["data"]["parameters"]["rn_lower_ratio_list"]],
                            folder+"/fig/predicted_value_cumulative_{}.png".format(name))

def plot_positive_pred_ratio(xs, ys, dataname, save_folder):
    ths = [i/10 for i in range(1,10)]
    fig = plt.figure(figsize=(10,10))
    plt.rcParams['font.size'] = 20
    cmap=cm.get_cmap("plasma")
    colors = [cmap(i) for i in np.linspace(0, 1, len(ths))]
    for i, th in enumerate(ths):
        plt.plot([round(x*100) for x in xs], [y*100 for y in ys[th]],
                 marker="o", label="threshold = {}".format(th), c=colors[i])
    plt.grid()
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xticks(list(range(0,101,10)))
    plt.yticks(list(range(0,101,10)))
    plt.xlabel("x (%)")
    plt.ylabel("Percentage of Data Predicted as Positive (%)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(save_folder+r"\predicted_as_positive_ratio_{}.png".format(dataname),bbox_inches='tight')
    plt.show()

def plot_cumulative(pos, unl, rn_lower_ratio, save_file):
    fig = plt.figure(figsize=(10,10))
    plt.plot([0,1],[0,2000],c="r",alpha=0.5)
    pcmap=cm.get_cmap("viridis")
    ucmap=cm.get_cmap("plasma")
    pcolors = [pcmap(i) for i in np.linspace(0, 1, len(rn_lower_ratio))]
    ucolors = [ucmap(i) for i in np.linspace(0, 1, len(rn_lower_ratio))]
    for i, p in enumerate(pos):
        plt.plot(p, [(j+1)/len(p) for j in range(len(p))],
                 label="Positive-Test_x={}%".format(rn_lower_ratio[i]*100),
                 c=pcolors[i])
    for i, u in enumerate(unl):
        plt.plot(u, [(j+1)/len(u) for j in range(len(u))],
                 label="Unlabeled-Test_x={}%".format(rn_lower_ratio[i]*100),
                 c=ucolors[i])
    plt.xlabel('Synthesizability',fontsize=20)
    plt.ylabel('Cumulative Distribution of Element Sets',fontsize=20)
    #plt.title(t,fontsize=20)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.grid()
    #plt.legend(loc='upper left')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()

def gen_heatmap_all(config):
    folder = config["files_and_directories"]["save_dir"]
    iters = itertools.product([float(l_rate) for l_rate in config["learning"]["learning_rate"]],
                              [int(batch_size) for batch_size in config["learning"]["batch_size"]],
                              [int(epoch) for epoch in config["learning"]["epoch"]],
                              [node_nums for node_nums in config["learning"]["node_nums"]])
    for l_rate, batch_size, epoch, node_nums in iters:
        for knn_k in [int(k) for k in config["data"]["parameters"]["knn_k_list"]]:
            for rn_lower_ratio in [float(k) for k in config["data"]["rn_lower_ratio_list"]]:
                print("")
                print("Generate heatmap : knn_k = {}, rn_lower_ratio = {}, l_rate = {}, batch_size = {}, epoch = {}, node_nums = {}".format(knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums))
                model_folder = folder+"/ml_model/knn-k={}_rn-lower-ratio={}_lrate={}_bsize={}_epoch={}_nodenums={}".format(knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums)
                for i in range(len(config["heatmaps"])):
                    generate_heatmap(config, model_folder, knn_k, rn_lower_ratio, i)

def generate_heatmap(config, model_folder, knn_k, rn_lower_ratio, idx):
    folder = config["files_and_directories"]["save_dir"]
    color = "RdYlBu"
    zmin, zmax = -1, 1
    default_val = 0.15
    threshold = 0.5
    buf = 0.05
    elem_parameter = data.load_element_parameter_from_config(config)
    elements, _ = get_elements()
    use_elements = list(elem_parameter.keys())
    save_folder = model_folder+"/"+config["heatmaps"][idx]["name"]
    os.mkdirs(save_folder+"/heatmap_html", exist_ok=True)
    os.mkdir(save_folder+"/heatmap_png", exist_ok=True)
    
    models = []
    for i in range(int(config["learning"]["model_num"])):
        models.append(tf.keras.models.load_model(model_folder+"/model_{}.tf".format(i)))
    for elem1 in tqdm.tqdm(use_elements):
        #print(i)
        ternary = []
        ternary_param = []
        for elem2 in use_elements:
            for elem3 in use_elements:
                ternary.append([elem1,elem2,elem3])
                ternary_param.append([elem_parameter[elem1],elem_parameter[elem2],elem_parameter[elem3]])
        ternary_param = tf.convert_to_tensor(ternary_param)
        
        prediction = np.zeros(shape=(len(ternary),), dtype=np.float64)
        for i in range(int(config["learning"]["model_num"])):
            pred = tf.reshape(models[i](ternary_param), (-1,)).numpy()
            prediction += pred/float(config["learning"]["model_num"])
        prediction = np.clip(a=prediction,a_min=0,a_max=1.)
        
        ternary_data = []
        columns = ["element1", #0
                   "element2", #1
                   "element3", #2
                   "color_score", #3
                   "hov_element", #4
                   "hov_score", #5
                   ]
        data_dicts = []
        heatmap_settings = {"source": [d["file"] for d in config["heatmaps"][idx]["mark_soruces"]],
                            "composition": [d["composition_column"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "property": [d["property_column"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "method": [d["property_selection_method"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "ratio": [d["property_filter"]["ratio"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "min": [d["property_filter"]["value_min"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "max": [d["property_filter"]["value_max"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "name": [d["property_filter"]["name"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "marks": [d["mark"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "labels": [d["label"] for d in config["heatmaps"][idx]["mark_sources"]],
                            "colors": [d["square_color"] for d in config["heatmaps"][idx]["mark_sources"]]
                            }
        if len(config["heatmaps"][idx]["mark_sources"]) != 0:
            for i in range(len(heatmap_settings["source"])):
                data_dicts.append(data.load_compounds_data(config,
                                                           heatmap_settings["source"][i],
                                                           heatmap_settings["composition"][i],
                                                           heatmap_settings["property"][i],
                                                           heatmap_settings["method"][i],
                                                           property_min=float(heatmap_settings["min"][i]),
                                                           property_max=float(heatmap_settings["max"][i]),
                                                           property_name=heatmap_settings["name"][i],
                                                           property_selection_ratio=float(heatmap_settings["ratio"][i]),
                                                           ))
                columns.append(str(i))
            columns.append("mark")
            columns.append("mark_color")
        
        for (elem1,elem2,elem3),k in zip(ternary,prediction):
            sq_data = {}
            sq_data["element1"] = elem1
            sq_data["element2"] = elem2
            sq_data["element3"] = elem3
            
            if elem1 == elem2 and elem2 == elem3:
                sq_data["hov_element"] = elem1
            elif elem1 == elem2:
                sq_data["hov_element"] = ", ".join([elem1,elem3])
            elif elem1 == elem3 or elem2 == elem3:
                sq_data["hov_element"] = ", ".join([elem1,elem2])
            else:
                sq_data["hov_element"] = ", ".join([elem1,elem2,elem3])
            
            sq_data["hov_score"] = "{:.4f}".format(k)
            
            elemset = tuple(sorted(list(set([elem1,elem2,elem3]))))
            for i, hd in enumerate(columns[6:len(columns)-2]):
                sq_data[hd] = ", ".join(data_dicts[i][elemset])
            
            if len(config["heatmaps"][idx]["mark_sources"]) != 0:
                marks = heatmap_settings["marks"]
                for i, hd in enumerate(columns[6:len(columns)-2]):
                    if len(sq_data[hd]) != 0:
                        sq_data["mark"] = marks[i]
                        if k >= threshold:
                            sq_data["mark_color"] = "white"
                        else:
                            sq_data["mark_color"] = "black"
                        break
                else:
                    sq_data["mark"] = ""
                    sq_data["mark_color"] = "black"
                
                crs = heatmap_settings["colors"]
                for i, hd in enumerate(columns[6:len(columns)-2]):
                    if crs[i] == "r" and len(sq_data[hd]) != 0:
                        sq_data["color_score"] = -(k*(1+default_val-buf)-default_val+buf)
                        break
                else:
                    sq_data["color_score"] = k*(1-default_val-buf)+default_val+buf
            
            ternary_data.append([sq_data[cn] for cn in columns])
        
        tooltips = [("Element:","@hov_element"),
                    ("Score", "@hov_score"),
                    ]
        tooltips.extend([(l, "@"+str(i)) for i, l in enumerate(heatmap_settings["labels"])])
        tools = "hover,save,pan,box_zoom,reset,wheel_zoom"
        
        df = pd.DataFrame(ternary_data, columns=columns)
        cm_ = plt.get_cmap(color)
        cm = [matplotlib.colors.rgb2hex((cm_(xxx)[0],cm_(xxx)[1],cm_(xxx)[2])) for xxx in range(256)]
        
        p = figure(title=elem1,
                   width=2000, height=2000,
                   x_range=elements, y_range=list(reversed(elements)),x_axis_location="above",
                   tools=tools, toolbar_location="above", tooltips=tooltips)
        
        p.title.text_font_size = "24pt"
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "12px"
        p.axis.major_label_standoff = 0
        p.axis.axis_label_text_font_style = 'bold'
        
        if len(config["heatmaps"][idx]["mark_sources"]) != 0:
            legend_list = []
            marks = heatmap_settings["marks"]
            for i in range(len(marks)):
                legend_list.append("{}: {}".format(marks[i], heatmap_settings["labels"][i]))
            p.xaxis.axis_label = "        ".join(legend_list)
        
        p.background_fill_color = None
        p.border_fill_color = None
        
        r = p.rect("element2", "element3", 1., 1., source=df, fill_alpha=1.,
                   fill_color=linear_cmap("color_score", cm, low=zmin, high=zmax),
                   line_color=None)
        
        x = dodge("element2", 0., range=p.x_range)
        text_props = dict(source=df, text_align="center", text_baseline="middle")
        
        p.text(x=x, y=dodge("element3", 0., range=p.y_range),
               text="mark", text_font_style="normal", text_color="mark_color", **text_props)
        
        p.hover.renderers = [r]
        
        save(p, filename=save_folder+r"\heatmap_html\{}.html".format(elem1), title=elem1)
        
        p.toolbar_location = None
        export_png(p, filename=save_folder+r"\heatmap_png\{}.png".format(elem1))

        #show(p)

def save_vals_all(config):
    folder = config["files_and_directories"]["save_fir"]
    iters = itertools.product([float(l_rate) for l_rate in config["learning"]["learning_rate"]],
                              [int(batch_size) for batch_size in config["learning"]["batch_size"]],
                              [int(epoch) for epoch in config["learning"]["epoch"]],
                              [node_nums for node_nums in config["learning"]["node_nums"]])
    for l_rate, batch_size, epoch, node_nums in iters:
        for knn_k in [int(k) for k in config["data"]["parameters"]["knn_k_list"]]:
            for rn_lower_ratio in [float(k) for k in config["data"]["parameters"]["rn_lower_ratio"]]:
                print("")
                print("Save predicted values : knn_k = {}, rn_lower_ratio = {}, l_rate = {}, batch_size = {}, epoch = {}, node_nums = {}".format(knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums))
                model_folder = folder+"/ml_model/knn-k={}_rn-lower-ratio={}_lrate={}_bsize={}_epoch={}_nodenums={}".format(knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums)
                for i in range(len(config["heatmaps"])):
                    save_vals(config, model_folder, knn_k, rn_lower_ratio,i)

def save_vals(config, model_folder, knn_k, rn_lower_ratio, idx):
    elem_parameter = data.load_element_parameter_from_config(config)
    elements, _ = get_elements()
    use_elements = list(elem_parameter.keys())
    save_folder = model_folder+"/"+config["heatmaps"][idx]["name"]    
    
    data_dicts = []
    heatmap_settings = {"source": [d["file"] for d in config["heatmaps"][idx]["mark_soruces"]],
                        "composition": [d["composition_column"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "property": [d["property_column"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "method": [d["property_selection_method"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "ratio": [d["property_filter"]["ratio"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "min": [d["property_filter"]["value_min"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "max": [d["property_filter"]["value_max"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "name": [d["property_filter"]["name"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "marks": [d["mark"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "labels": [d["label"] for d in config["heatmaps"][idx]["mark_sources"]],
                        "colors": [d["square_color"] for d in config["heatmaps"][idx]["mark_sources"]]
                        }
    columns = ["element_set", "predicted_value"]
    if len(config["heatmaps"][idx]["mark_sources"]) != 0:
        for i in range(len(heatmap_settings["source"])):
            data_dicts.append(data.load_compounds_data(config,
                                                       heatmap_settings["source"][i],
                                                       heatmap_settings["composition"][i],
                                                       heatmap_settings["property"][i],
                                                       heatmap_settings["method"][i],
                                                       property_min=float(heatmap_settings["min"][i]),
                                                       property_max=float(heatmap_settings["max"][i]),
                                                       property_name=heatmap_settings["name"][i],
                                                       property_selection_ratio=float(heatmap_settings["ratio"][i]),
                                                       ))
            columns.append(heatmap_settings["labels"][i])
    
    models = []
    for i in range(int(config["learning"]["model_num"])):
        models.append(tf.keras.models.load_model(model_folder+r"\model_{}.tf".format(i)))
    
    fins = set()
    ternary = []
    ternary_param = []
    pred_results = []
    for elem1 in tqdm.tqdm(use_elements):
        for elem2 in use_elements:
            for elem3 in use_elements:
                es = tuple(sorted(list(set([elem1, elem2, elem3]))))
                if es in fins:
                    continue
                ternary.append(es)
                ternary_param.append([elem_parameter[elem1],elem_parameter[elem2],elem_parameter[elem3]])
                fins.add(es)

    for i in range(len(ternary)//10000+1):
        tp = tf.convert_to_tensor(ternary_param[i*10000:min((i+1)*10000, len(ternary))])
        prediction = np.zeros(shape=(tp.shape[0],), dtype=np.float64)
        for i in range(int(config["learning"]["model_num"])):
            pred = tf.reshape(models[i](tp),(-1,)).numpy()
            prediction += pred/float(config["learning"]["model_num"])
        prediction = np.clip(a=prediction,a_min=0,a_max=1.)
        pred_results.extend(prediction)

    save_data = []
    for i in range(len(pred_results)):
        sd = ["-".join(ternary[i]), str(round(pred_results[i], 4))]
        ul = 1
        for j in range(2, len(columns)):
            if ternary[i] in data_dicts[j-2]:
                sd.append("1")
                ul = 0
            else:
                sd.append("0")
        sd.append(str(ul))
        save_data.append(sd)
    
    save_data.sort(key=lambda x:x[0])
    os.mkdirs(save_folder+"/{}".format(heatmap_settings["labels"][0]), exist_ok=True)
    with open(save_folder+"/{}/knn_k={}_rn-lower_ratio={}.csv".format(heatmap_settings["labels"][0], knn_k, rn_lower_ratio), "w") as f:
        f.write(",".join(columns)+",Unlabeled"+"\n")
        for sd in save_data:
            f.write(",".join(sd)+"\n")

def get_elements():
    elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn","Fe", "Co","Ni", "Cu", "Zn",
                "Ga", "Ge", "As", "Se", "Br", "Kr",
                "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                "In", "Sn", "Sb", "Te", "I", "Xe",
                "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm","Sm","Eu", "Gd", "Tb",
                "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                "Tl", "Pb", "Bi", "Po", "At", "Rn",
                "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"]
    elements_dict = {}
    for i in range(len(elements)):
        elements_dict[elements[i]] = i
    return elements, elements_dict

def unlabeled_predvals_vs_knnscore_score_plot(config):
    print("Plot unlabeled predvals vs knnscore score.")
    folder = config["files_and_directories"]["save_dir"]
    iters = itertools.product([float(l_rate) for l_rate in config["learning"]["learning_rate"]],
                              [int(batch_size) for batch_size in config["learning"]["batch_size"]],
                              [int(epoch) for epoch in config["learning"]["epoch"]],
                              [node_nums for node_nums in config["learning"]["node_nums"]])
    xs = [float(x) for x in config["data"]["parameters"]["rn_lower_ratio_list"]]
    
    with open(folder+"/train_and_test_data/unlabeled_similarity_with_positive_sorted.pkl", "r") as f:
        train_knn_results = pickle.load(f)
    with open(folder+"/train_and_test_data/unlabeled_test_similarity_with_positive_train_sorted.pkl", "r") as f:
        test_knn_results = pickle.load(f)
    
    with open(folder+"/train_and_test_data/unlabeled_train.pkl", "r") as f:
        unlabeled_train = pickle.load(f)
    with open(folder+"/train_and_test_data/unlabeled_test.pkl", "r") as f:
        unlabeled_test = pickle.load(f)
    
    elem_parameter = data.load_element_parameter_from_config(config)
    
    os.mkdirs(folder+r"/fig", exist_ok=True)
    
    for l_rate, batch_size, epoch, node_nums in iters:
        for knn_k in [int(k) for k in config["data"]["parameters"]["knn_k_list"]]:
            for rn_lower_ratio in [float(rn_lower_ratio) for rn_lower_ratio in config["data"]["parameters"]["rn_lower_ratio_list"]]:
                model_folder = folder+"/ml_model/knn-k={}_rn-lower-ratio={}_lrate={}_bsize={}_epoch={}_nodenums={}".format(knn_k, rn_lower_ratio, l_rate, batch_size, epoch, node_nums)
                models = []
                for i in range(int(config["learning"]["model_num"])):
                    models.append(tf.keras.models.load_model(model_folder+"/model_{}.tf".format(i)))
                #モデルの読み込み
                #予測
                train_ternary_param = []
                test_ternary_param = []
                train_knnscore = []
                test_knnscore = []
                for ultr in unlabeled_train:
                    u = list(ultr)
                    if len(u) > 3:
                        continue
                    while len(u) < 3:
                        u.append(u[-1])
                    train_ternary_param.append([elem_parameter[u[0]],elem_parameter[u[1]],elem_parameter[u[2]]])
                    train_knnscore.append(sum(train_knn_results[ultr][:knn_k]))
                for ulte in unlabeled_test:
                    u = list(ulte)
                    if len(u) > 3:
                        continue
                    while len(u) < 3:
                        u.append(u[-1])
                    test_ternary_param.append([elem_parameter[u[0]],elem_parameter[u[1]],elem_parameter[u[2]]])
                    test_knnscore.append(sum(test_knn_results[ulte][:knn_k]))
                
                train_ternary_param = tf.convert_to_tensor(train_ternary_param)
                test_ternary_param = tf.convert_to_tensor(test_ternary_param)

                train_prediction = np.zeros(shape=(len(unlabeled_train),), dtype=np.float64)
                test_prediction = np.zeros(shape=(len(unlabeled_test),), dtype=np.float64)
                for i in range(int(config["learning"]["model_num"])):
                    train_pred = tf.reshape(models[i](train_ternary_param),(-1,)).numpy()
                    train_prediction += train_pred/float(config["learning"]["model_num"])
                    test_pred = tf.reshape(models[i](test_ternary_param),(-1,)).numpy()
                    test_prediction += test_pred/float(config["learning"]["model_num"])
                train_prediction = np.clip(a=train_prediction,a_min=0,a_max=1.)
                test_prediction = np.clip(a=test_prediction,a_min=0,a_max=1.)
                
                #プロット
                fig = plt.figure(figsize=(10,10))
                plt.plot(train_knnscore, train_prediction)
                plt.grid()
                plt.xlabel("Average top-k similarity score")
                plt.ylabel("Predicted Value")
                plt.title("Training data")
                plt.savefig(folder+"/fig/train_knn_vs_prediction/{}.png")
                
                fig = plt.figure(figsize=(10,10))
                plt.plot(test_knnscore, test_prediction)
                plt.grid()
                plt.xlabel("Average top-k similarity score")
                plt.ylabel("Predicted Value")
                plt.title("Test data")
                plt.savefig(folder+"/fig/test_knn_vs_prediction/{}.png")