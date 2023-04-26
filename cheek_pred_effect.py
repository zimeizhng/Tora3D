from pre_data_f import pre_data, pre_test_data

from utils_base import pickle_
from utils_base import info
from utils_base import group_by_num

from utils_others import extract_mol_from_chemdir
from utils_others import rotate_a_mol
from utils_others import writosdf

from utils_others import *
from utils_base import *

from utils import count_parameters

from loops import batch_inference, inference

import torch
from torch import nn
from torch.optim import Adam
import copy
import os.path as osp
from tqdm.notebook import tqdm

# from dgllife.utils import Meter
# from functools import partial

import numpy as np
import pandas as pd
import time

from find_isomer.get_iso_by_FF import  drawit, get_ff_ms

# torch.set_printoptions(profile="full")   #
# torch.set_printoptions(sci_mode=False)  #


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



def construct_test_loader_by_dic(test_200_dic_path, 
                                    smilelist,
                                    torsion_angle,
                                    torsion_list,
                                    relativeenergy_list,
                                    batch_size,
                                    max_num_atom, 
                                    wl_max_iter, 
                                    num_choosed_confs, 
                                    device = device):
    # get idx
    test_200_dic = pickle_.load(test_200_dic_path)
    idxs = get_idxs_bylist(test_200_dic.keys(), smilelist)    

    # get dataset
    smilelist_200 = list(np.array(smilelist)[idxs])
    torsion_angle_200 = list(np.array([t.cpu() for t in torsion_angle])[idxs])
    torsion_list_200 = list(np.array([t.cpu() for t in torsion_list])[idxs])
    relativeenergy_list_200 = list(np.array(relativeenergy_list)[idxs])

    # get loadr
    test_loader = pre_test_data(smilelist_200, torsion_list_200, torsion_angle_200, relativeenergy_list_200, batch_size,
                  max_num_atom, wl_max_iter, num_choosed_confs, device = device)
    return test_loader



def pre_1batch(batch_id, batch_data, kn, model, loss_criterion, loss_criterion_, optimizer=None, debug=False, max_re=6):

    prediction_list_batchge_list = []
    # labels_list_batchge_list = []
    
    for i in range(kn):
        loss_value, acc_sig, prediction_list, labels_list,\
            masks, relativeenergys, num_atom_list, angle_group_num, num_confs_list, smiles, _ = \
            batch_inference(model, batch_data, loss_criterion, loss_criterion_, optimizer=None, debug = debug, max_re = max_re)
          
        prediction_list_batchge = group_by_num(prediction_list, num_confs_list)
        prediction_list_batchge_ = [torch.stack(i) for i in prediction_list_batchge]
        labels_list_batchge = group_by_num(labels_list, num_confs_list)
        labels_list_batchge_ = [torch.stack(i) for i in labels_list_batchge]
        
        prediction_list_batchge_list.append(prediction_list_batchge_)     
            
    prediction_list_batchge_ = [torch.cat(i,0) for i in zip(*prediction_list_batchge_list)]
    return masks, relativeenergys, num_atom_list, angle_group_num, num_confs_list,\
                        smiles,prediction_list_batchge_,labels_list_batchge_, _

def noise_angles_allconfs(angles_allconfs, f_percent):
    num_points = int(torch.prod(torch.tensor(angles_allconfs.shape))*f_percent)
    for i in range(num_points):
        rand_point = np.random.choice(range(angles_allconfs.shape[0])), np.random.choice(range(angles_allconfs.shape[1]))

        a1 = np.random.uniform(-angles_allconfs.abs().min().cpu(),angles_allconfs.abs().min().cpu())
        a2 = np.random.uniform( angles_allconfs.abs().max().cpu(), 180)
        a3 = np.random.uniform( -180,-angles_allconfs.abs().max().cpu())
        angles_allconfs[rand_point] = np.random.choice([a1,a2,a3])
    return angles_allconfs

def twist(angles_allconfs, mol_ff_list, mask): 
    angles_allconfs_tuple = torch.chunk(angles_allconfs, len(mol_ff_list), dim = 0)
    mol_list_pred = []
    for i, angles_allconfs_ in enumerate(angles_allconfs_tuple):
        mol_list_p = rotate_a_mol_normangle(mol_ff_list[i], mask, angles_allconfs_)
        mol_list_pred.extend(mol_list_p)
    return mol_list_pred

def sig_value2angle(x):
    y = (x[:,:,0]+0.5)*180*(x[:,:,1])
    return y

def generate_pred_confs(model, data_loader, 
                        loss_criterion, loss_criterion_, 
                        save_conf_root=None, # "./confs_save"
                        chem_dataset_dir = "/home/admin/work/geomol/rdkit_folder/drugs",
                        ori_confs_path = "./data/drugs/test_loader_smol.pkl",
                        optimizer=None, 
                        debug = False,
                        angle_range_list = None,
                        k = "",
                        max_re = 5,
                        kn = 2,
                        f_percent = 0.1,
                        part_batch = None
                       ):

    if part_batch == None:
        part_batch = (0, len(data_loader))
    
    
    model.eval()
    
    matr_list_list = []
    matp_list_list = []
    covr_list_list = []
    covp_list_list = []
    
#     covr_tl_list_list = []
#     covr_tr_list_list = []

    angles_allconfs_list_list = []
    lab_angles_allconfs_list_list = []

    df = pd.DataFrame(columns=["angle_group_num","num_confs_list","matr","matp","covr","covp"])
    dfidx = 0
    ori_confs = pickle_.load(ori_confs_path)
    
    _list = []
    smiles_list = []
    num_confs_list_list = [] 
    num_atom_list_list =[]
    angle_group_num_list =[]
    
    with torch.no_grad():
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            
            if batch_id < part_batch[0] :
                continue            
            
            if batch_id == part_batch[1]:
                break
            masks, relativeenergys, num_atom_list, angle_group_num, \
               num_confs_list, smiles,prediction_list_batchge_,labels_list_batchge_, _ = pre_1batch(
                                                                  batch_id, batch_data, kn, model, 
                                                                  loss_criterion, loss_criterion_, 
                                                                  optimizer=optimizer, debug=debug, max_re=6)
            _list.append(_)
            smiles_list.append(smiles) 
            num_confs_list_list.append(num_confs_list)
            num_atom_list_list.append(num_atom_list)
            angle_group_num_list.append(angle_group_num)
            
            
            angles_allconfs_list = []
            lab_angles_allconfs_list = []
            matr_list = []
            matp_list = []
            covr_list = []
            covp_list = []

            for i in tqdm(range(len(prediction_list_batchge_))):

                smile = smiles[i]
                try:
                    mol3D = ori_confs[smile]
                except KeyError as e:
                    print(e)
                    continue
                if angle_range_list == None:
                    mol_ff_list = [mol3D]
                else:
                    mol_ff_list = get_ff_ms(mol3D, angle_range_list)
        
                angles_allconfs = sig_value2angle(prediction_list_batchge_[i])
                lab_angles_allconfs = sig_value2angle(labels_list_batchge_[i])
                angles_allconfs = noise_angles_allconfs(angles_allconfs, f_percent)
                mol_list_pred = twist(angles_allconfs, mol_ff_list, masks[i])
                mol_list_true, relativeenergy_list_true = extract_mol_from_chemdir(chem_dataset_dir, smile)
                
                try:
                    matr,matp,covr,covp             = get_onemol_COV_MAT_o(mol_list_true, mol_list_pred, threshold = 1.25)
                except RuntimeError as e:
                    print(e)                    
                    continue
                if matp>1e+10:
                    continue             
                           
                # 写入文件
                # -------------------------------------------------------------
                if save_conf_root != None:  
                    to_root = osp.join(save_conf_root,smile.replace("/", "_"))

                    sdf_name_p =  f"p_k{k}_kn{kn}_f_p{f_percent}_scale{model.noise_scale}_re{max_re}_len{len(mol_list_pred)}_covr{round(covr, 3)}_pred_confs_aligned.sdf"
                    writosdf(mol_list_pred,to_root, sdf_name_p, Align=True)
                    sdf_name_t =  f"t_k{k}_len{len(mol_list_true)}_true_confs_aligned.sdf"
                    writosdf(mol_list_true,to_root, sdf_name_t, Align=True)
                # ===============================================================

                one_mole_statis = (angle_group_num[i],num_confs_list[i],matr,matp,covr,covp)
                df.loc[dfidx] = one_mole_statis

                dfidx += 1
                angles_allconfs_list.append(angles_allconfs)
                lab_angles_allconfs_list.append(lab_angles_allconfs)
                matr_list.append(matr)
                matp_list.append(matp)
                covr_list.append(covr)
                covp_list.append(covp)
       
            matr_list_list.append(np.mean(matr_list))
            matp_list_list.append(np.mean(matp_list))
            covr_list_list.append(np.mean(covr_list))
            covp_list_list.append(np.mean(covp_list))
            
            print( np.mean(covr_list_list))
            
            angles_allconfs_list_list.append(angles_allconfs_list)
            lab_angles_allconfs_list_list.append(lab_angles_allconfs_list)
            
    matr_ = np.mean(matr_list_list)
    matp_ = np.mean(matp_list_list)
    covr_ = np.mean(covr_list_list)
    covp_ = np.mean(covp_list_list)
    
    
    return matr_,matp_,covr_,covp_,df,\
                  lab_angles_allconfs_list_list,angles_allconfs_list_list,_list, smiles_list,num_confs_list_list,num_atom_list_list,angle_group_num_list
if __name__ == "__main":

    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    model.noise_scale = 100

    matr_,matp_,covr_,covp_,\
        lab_angles_allconfs_list_list,angles_allconfs_list_list= generate_pred_confs(model, test_loader, 
                                                                loss_criterion, loss_criterion_, 
                                                                save_conf_root="./confs_save",
                                                                chem_dataset_dir = "/home/admin/work/geomol/rdkit_folder/drugs",
                                                                ori_confs_path  = test_dic_path,                                                                 
                                                                optimizer=None, 
                                                                debug = False,
                                                                angle_range_list = [(-20,20),(40,100)],
                                                                k = "2",
                                                                max_re = 6,
                                                                kn = 2,
                                                                f_percent=0.25)
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



