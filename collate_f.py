import torch
from generate_feat import generate_feat
from utils import Data_enhancement
from utils_base import pad_list, get_pad_tensor
import random
import numpy as np

def collate(samples, max_num_atom, wl_max_iter,num_choosed_confs = 300, device = "cuda"):

    smiles, labels, mask, relativeenergy = map(list, zip(*samples))
    
    angle_group_num = [l.shape[-2] for l in labels] 
    max_agn = max(angle_group_num) 

    num_mol = len(smiles)
    labels_ = []
    relativeenergy_ = []
    for i in range(num_mol):
        label = labels[i].to(torch.float32).to(device)
        r = torch.tensor(relativeenergy[i], device =device)

        num_confs = label.shape[0]
        if num_confs >= num_choosed_confs:
            idx_choosed_confs = random.sample(range(num_confs), num_choosed_confs)
            label_ = label[idx_choosed_confs]
            r_ = r[idx_choosed_confs]
            labels_.append(label_)
            relativeenergy_.append(r_)        
        else:
            labels_.append(label)
            relativeenergy_.append(r) 

    num_confs_list = [l.shape[0] for l in labels_] 

    labels = torch.cat([get_pad_tensor(labels_[i], [labels_[i].shape[0], max_agn, labels_[i].shape[-1]]) for i in range(num_mol)],dim=0)
    relativeenergy = torch.cat(relativeenergy_, dim =0)

    try:
        atom_feat, AM_bond_feat, node_color_list, d_list, num_atom_list = zip(*[generate_feat(smile, max_num_atom, wl_max_iter,device=device) for smile in smiles])
    except TypeError as e:
        print("这个里面有generate_feat 错误")
    
    atom_feat = torch.stack(atom_feat)
    AM_bond_feat = torch.stack(AM_bond_feat)
    node_color_feat = torch.tensor(node_color_list, device = device)
    d = torch.tensor(d_list, device=device)
    return atom_feat, AM_bond_feat, node_color_feat, d, labels, mask, relativeenergy, num_atom_list, angle_group_num, num_confs_list, smiles
