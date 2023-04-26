

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 超参数

# generate feat
max_num_atom=90
max_Emb_wl=40
max_Emb_d=10

wl_max_iter=3
# data_enhancement = False

dataset_name = "drugs"



# data_path = f"user_data/input/{dataset_name}/smiles_list.smi"
data_path = f"user_data/input/{dataset_name}/smiles_same.smi"
write_root = f'user_data/input/{dataset_name}/same_test_output'
sdf_input_path = osp.join(write_root,"smiles_same_one.sdf")
max_relativeenergy = 20
smiles_len_max = 64

batch_size = 64


# from load_data import load_data
from pre_data_from_users import geom_confs
from utils import choose_dihedral_pairs_atoms4
from utils import get_allbonds_rotatiable
from utils import load_smiles
from utils_others import genrate_con_from_roedangles


from collate import collate_new_user

import torch
from torch import nn
from torch.optim import Adam
import copy
from torch.utils.data import DataLoader
from functools import partial

from dgllife.utils import Meter

import numpy as np
from itertools import product

from rdkit import Chem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
import os
import time
import random
import os.path as osp

def predict_con_from_user(device, cheekpoint_path):
    model, Train_loss, Val_loss = load_model(device=device, cheekpoint_path=cheekpoint_path )
            






































