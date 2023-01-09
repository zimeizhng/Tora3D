from utils_others import open_summery
import os.path as osp
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams[ 'figure.dpi' ] = 300 # Make figures have reasonable resolution when exporting


def get_num_confs_list(summ):    
    values = summ.values()
    num_confs_list = []
    for value in values:
        try:
            v = value['totalconfs']
            num_confs_list.append(v)
        except KeyError as e:
            continue
            
    return num_confs_list

def bar(list_1, list_2=None, label="UNknow", xlabel="x",ylabel="y", size=64, fs=(20,20), color = "r"):
    fig, ax1 = plt.subplots(figsize=fs)
    plt.xlabel(xlabel, fontdict={'size': size})
    plt.ylabel(ylabel, fontdict={'size': size})
    if list_2 == None:
        ax1.bar(len(list_1),list_1,color = color,label=label)
    else:
        ax1.bar(list(list_1),list(list_2), color = color,label=label)

def plot(list_1, list_2=None, label="UNknow", xlabel="x",ylabel="y", size=64, fs=(20,20), color = "r"):
    fig, ax1 = plt.subplots(figsize=fs)
    plt.xlabel(xlabel, fontdict={'size': size})
    plt.ylabel(ylabel, fontdict={'size': size})
    if list_2 == None:
        ax1.plot(len(list_1),list_1,color,label=label)
        # seaborn.barplot(len(list_1),list_1,color,label=label)
        
    else:    
        ax1.plot(list(list_1),list(list_2),color,label=label)
        # seaborn.barplot(list(list_1),list(list_2),color,label=label)

def sort_dic(dict_data):
    sorted_dic=sorted(dict_data.items(),key=lambda x:x[0])
    return sorted_dic

def statistic_num_confs(root, label="UNknow", xlabel="x",ylabel="y", size=64, fs=(20,20), color = "r"):
    """
    pram:
        root:  eg: "/home/admin/work/geomol/rdkit_folder/drugs"
        
    """
    summ = open_summery(root)
    print("len(summ)",len(summ))
    
    num_confs_list = get_num_confs_list(summ)
    print("len(num_confs_list)",len(num_confs_list))    
    nc_dic = Counter(num_confs_list)
    
    display(bar(nc_dic.keys(), nc_dic.values(), 
                label=label, xlabel=xlabel,ylabel=ylabel, size=size, fs=fs, color = color))
    
    
    
    
# ------------------pandas、matplotlib作图之设置坐标轴标签、标题和图例标签的字体大小-----------------------
# import matplotlib.pyplot as plt
 
# fig,ax=plt.subplots(1,1,figsize=(9,6))

# ax.set_ylabel(fontsize=20) #设置y轴标签字体大小
# ax.set_xlabel(fontsize=20) #设置x轴标签字体大小
# ax.set_title(fontsize=30)  #设置标题字体大小
# ax.legend(fontsize=15)     #设置图例字体大小
# =========================================================================================================
    