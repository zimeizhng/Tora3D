import torch
import numpy as np
import pandas as pd
from numpy import *
import pickle
from rdkit import Chem
import torch_geometric as tg
from torch_geometric.utils import degree
import networkx as nx
from itertools import product
from math import cos, sin, pi, atan2
from utils_base import angle_vector,getArotateM
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
from rdkit.Chem.rdMolTransforms import GetDihedralDeg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def Data_enhancement(labels,mask):
    for i in range(len(labels)):
        label_i = labels[i]
        mask_i = mask[i]
        mask_i = [list(i) for i in mask_i] 
        len_ = len(mask_i)

        rand = np.random.choice(len_+1)
        rand_ = np.random.choice(len_, rand, replace=False)

        for j in rand_:
            mask_i[j].reverse()

#         rand_xuhao=np.random.choice(len(label_i), len(label_i), replace=False)

#         label_i = np.array(label_i)
#         mask_i = np.array(mask_i)
#         label_i = label_i[rand_xuhao]    
#         mask_i = mask_i[rand_xuhao]

        labels[i] = list(label_i)
        mask[i] = mask_i
        
    return labels, mask


def open_pickle(mol_path):
    with open(mol_path, "rb") as f:
        dic = pickle.load(f)
    return dic

def norm_inchi_index(mol):
    inchi, aux_info = Chem.MolToInchiAndAuxInfo(mol)
    for i in aux_info.split('/'):
        if i[0]=='N':
            pos=i[2:].split(',')
    mm_map = [int(j)-1 for i,j in enumerate(pos)]
    return mm_map


# def get_cycle_values(cycle_list, start_at=None):
#     start_at = 0 if start_at is None else cycle_list.index(start_at)
#     while True:
#         yield cycle_list[start_at]
#         start_at = (start_at + 1) % len(cycle_list)
        

# def get_cycle_indices(cycle, start_idx):
#     cycle_it = get_cycle_values(cycle, start_idx)
#     indices = []

#     end = 9e99
#     start = next(cycle_it)
#     a = start
#     while start != end:
#         b = next(cycle_it)
#         indices.append(torch.tensor([a, b],device = device))
#         a = b
#         end = b

#     return indices

# def get_current_cycle_indices(cycles, cycle_check, idx):
#     c_idx = [i for i, c in enumerate(cycle_check) if c][0]
#     current_cycle = cycles.pop(c_idx)
#     current_idx = current_cycle[(np.array(current_cycle) == idx.item()).nonzero()[0][0]]
#     return get_cycle_indices(current_cycle, current_idx)


# def get_dihedral_pairs(edge_index, data):
#     """
#     Given edge indices, return pairs of indices that we must calculate dihedrals for
#     """
#     start, end = edge_index
#     degrees = degree(end)
#     dihedral_pairs_true = torch.nonzero(torch.logical_and(degrees[start] > 1, degrees[end] > 1))
#     dihedral_pairs = edge_index[:, dihedral_pairs_true].squeeze(-1)

#     # # first method which removes one (pseudo) random edge from a cycle
#     dihedral_idxs = torch.nonzero(dihedral_pairs.sort(dim=0).indices[0, :] == 0).squeeze().detach().cpu().numpy()

#     # prioritize rings for assigning dihedrals
#     dihedral_pairs = dihedral_pairs.t()[dihedral_idxs]
#     G = nx.to_undirected(tg.utils.to_networkx(data))
#     cycles = nx.cycle_basis(G)
#     keep, sorted_keep = [], []

#     if len(dihedral_pairs.shape) == 1:
#         dihedral_pairs = dihedral_pairs.unsqueeze(0)

#     for pair in dihedral_pairs:
#         x, y = pair

#         if sorted(pair) in sorted_keep:
#             continue

#         y_cycle_check = [y in cycle for cycle in cycles]
#         x_cycle_check = [x in cycle for cycle in cycles]

#         if any(x_cycle_check) and any(y_cycle_check):  # both in new cycle
#             cycle_indices = get_current_cycle_indices(cycles, x_cycle_check, x)
#             keep.extend(cycle_indices)
#             sorted_keep.extend([sorted(c) for c in cycle_indices])
            
#             if sorted(pair) not in sorted_keep:
#                 keep.append(pair)
#                 sorted_keep.extend([sorted(pair)])
            
#             continue

#         if any(y_cycle_check):
#             cycle_indices = get_current_cycle_indices(cycles, y_cycle_check, y)
#             keep.append(pair)
#             keep.extend(cycle_indices)

#             sorted_keep.append(sorted(pair))
#             sorted_keep.extend([sorted(c) for c in cycle_indices])
#             continue

#         keep.append(pair)
#     keep = [t.to(device) for t in keep]
#     return torch.stack(keep).t()

# def get_precisemany_dihedral_pairs_atoms4(smiles):
#     """
#     获得分子的二面角，只得到可旋转的二面角，提取每个可旋转键的全部二面角
#     """
#     bonds_rotatiable = get_allbonds_rotatiable(smiles)
#     if len(bonds_rotatiable) == 0:
#         return torch.tensor([], device=device)
#     mol = Chem.MolFromSmiles(smiles)
#     dihedral_pairs_atoms4_list = []
#     for bond in bonds_rotatiable:
#         atom1_idx, atom2_idx = bond
#         atom1 = mol.GetAtomWithIdx(atom1_idx)
#         atom2 = mol.GetAtomWithIdx(atom2_idx)

#         nei_atom1s = []
#         for nei_atom1 in atom1.GetNeighbors():
#             if nei_atom1.GetIdx() != atom2_idx:
#                 nei_atom1s.append(nei_atom1.GetIdx())

#         nei_atom2s = []
#         for nei_atom2 in atom2.GetNeighbors():
#             if nei_atom2.GetIdx() != atom1_idx:
#                 nei_atom2s.append(nei_atom2.GetIdx())

#         loop_val = [nei_atom1s,nei_atom2s]
#         for i in product(*loop_val):
#             dihedral_pairs_atoms4 = [i[0],atom1_idx,atom2_idx,i[1]]
#             dihedral_pairs_atoms4_list.append(dihedral_pairs_atoms4)
        
#     return torch.tensor(dihedral_pairs_atoms4_list, device=device).t()
    
    

# def get_preciseone_dihedral_pairs_atoms4(smiles):
#     """
#     获得分子的二面角，只得到可旋转的二面角，每个键只选择一个二面角
#     """
#     bonds_rotatiable = get_allbonds_rotatiable(smiles)
#     if len(bonds_rotatiable) == 0:
#         return torch.tensor([], device=device)
#     precisemany_dihedral_pairs_atoms4 = get_precisemany_dihedral_pairs_atoms4(smiles)
#     precisemany_dihedral_pairs_atoms4_t = precisemany_dihedral_pairs_atoms4.t()
#     bonds_rotatiable = torch.tensor(bonds_rotatiable, device=device)
#     dihedral_pairs_atoms4_list = []
#     for bond in bonds_rotatiable:
#         for atoms4 in precisemany_dihedral_pairs_atoms4_t:
#             if not (atoms4[1:3] != bond).sum() :
#                 dihedral_pairs_atoms4_list.append(atoms4)
#                 break
                
#     return torch.stack(dihedral_pairs_atoms4_list).t()

# def get_tense_dihedral_pairs_atoms4(data):
    
#     """
#     这个是根据edge_index_dihedral_pairs来返回的四原子组，键的顺序和edge_index_dihedral_pairs一致，而且是tense的，是全部的二面角，可能有重复（由于edge_index_dihedral_pairs的重复）
#     """
    
#     edge_index_dihedral_pairs = data.edge_index_dihedral_pairs   
#     smiles = data.name
#     mol = Chem.MolFromSmiles(smiles)
#     edge_index_dihedral_pairs_t = edge_index_dihedral_pairs.t()
    
#     dihedral_pairs_atoms4_list = []
#     for bond in edge_index_dihedral_pairs_t:
#         atom1_idx, atom2_idx = bond
#         atom1 = mol.GetAtomWithIdx(atom1_idx.item())
#         atom2 = mol.GetAtomWithIdx(atom2_idx.item())

#         nei_atom1s = []
#         for nei_atom1 in atom1.GetNeighbors():
#             if nei_atom1.GetIdx() != atom2_idx:
#                 nei_atom1s.append(nei_atom1.GetIdx())

#         nei_atom2s = []
#         for nei_atom2 in atom2.GetNeighbors():
#             if nei_atom2.GetIdx() != atom1_idx:
#                 nei_atom2s.append(nei_atom2.GetIdx())

#         loop_val = [nei_atom1s,nei_atom2s]
#         for i in product(*loop_val):
#             dihedral_pairs_atoms4 = [i[0],atom1_idx.item(),atom2_idx.item(),i[1]]
#             dihedral_pairs_atoms4_list.append(dihedral_pairs_atoms4)
        
#     return torch.tensor(dihedral_pairs_atoms4_list, device=device).t()  

    
# def get_dihedral_pairs_atoms4(data):
    
#     # data需要先整一下, edge_index_dihedral_pairs好像有重复的
#     bo = data.edge_index_dihedral_pairs
#     bot = bo.t()

#     list_bot = []
#     for each in bot:
#         each_ = each.cpu().numpy().tolist()
#         each_reverse = [each_[1],each_[0]]

#         if each_ not in list_bot and each_reverse not in list_bot:
#             list_bot.append(each_) 
    
#     bot = torch.tensor(list_bot, device=device)
#     bo = bot.t()
#     num = bot.shape[0]
#     mol = Chem.MolFromSmiles(data.name)
    

#     left_list = []
#     right_list = []
#     for i in range(num):
#         bond = bot[i]

#         if i-1 not in range(num):
#             if (bond[0] == bo[1]).sum() == 0:
#                 atom = mol.GetAtomWithIdx(bond[0].item())
#                 for nei_atom in atom.GetNeighbors():
#                     if nei_atom.GetIdx() != bond[1]:
#                         left = nei_atom.GetIdx()
                        
#                         break                       
#             else:
#                 l_i = nonzero(bond[0] == bo[1])[0].squeeze().squeeze()
#                 left = bo[0][l_i]

#         else:
#             if bot[i-1][1] != bond[0]:
                
#                 if (bond[0] == bo[1]).sum() == 0:
#                     atom = mol.GetAtomWithIdx(bond[0].item())
#                     for nei_atom in atom.GetNeighbors():
#                         if nei_atom.GetIdx() != bond[1]:
#                             left = nei_atom.GetIdx()
                            
#                             break

#                 else:
#                     l_i = nonzero(bond[0] == bo[1])[0].squeeze().squeeze()
#                     left = bo[0][l_i]
                 
#             else:
#                 left = bot[i-1][0]

#         left_list.append(left)


#         if i+1 not in range(num):
#             if (bond[1] == bo[0]).sum() == 0:
#                 atom = mol.GetAtomWithIdx(bond[1].item())
#                 for nei_atom in atom.GetNeighbors():
#                     if nei_atom.GetIdx() != bond[0]:
#                         right = nei_atom.GetIdx()
#                         break
                        
#             else:
#                 r_i = nonzero(bond[1] == bo[0])[0].squeeze().squeeze()
#                 right = bo[1][r_i]

#         else:
#             if bot[i+1][0] != bond[1]:
                
#                 if (bond[1] == bo[0]).sum() == 0:
#                     atom = mol.GetAtomWithIdx(bond[1].item())
#                     for nei_atom in atom.GetNeighbors():
#                         if nei_atom.GetIdx() != bond[0]:
#                             right = nei_atom.GetIdx()
#                             break
                            
#                 else:
#                     r_i = nonzero(bond[1] == bo[0])[0].squeeze().squeeze()
#                     right = bo[1][r_i]            
#             else:
#                 right = bot[i+1][1]

#         # 如果二面角第一个和第四个原子相同，则需要修改一下
#         if right == left:
#             right_atom = mol.GetAtomWithIdx(bond[1].item())
#             for nei_atom in right_atom.GetNeighbors():
#                 if nei_atom.GetIdx() != right and nei_atom.GetIdx() != bond[0]:
#                     right = nei_atom.GetIdx()

#                     break

#             #如果实在没有的话 ，就设为-1       
#             #if right == left:
#                 #right = -1

#         right_list.append(right)

# #     print(left_list)
# #     print(right_list)
# #     print(bo)
#     dihedral_pairs_atoms4 = torch.cat((torch.tensor(left_list,device = device).unsqueeze(0), bo, torch.tensor(right_list,device = device).unsqueeze(0)))
    
#     return dihedral_pairs_atoms4
    
    
# def getDihedralAngle(p0,p1,p2,p3):
#     """
#     正负定义在 文件  /home/admin/work/大四下学期药物所服务器文件/大四下学期药物所服务器文件/v100/predict_Dihedral_angle/草稿箱/二面角正负的定义.png  中
#     """
#     p0,p1,p2,p3 = p0.cpu(),p1.cpu(),p2.cpu(),p3.cpu()
    
#     e01 = p1-p0
#     e12 = p2-p1
#     e23 = p3-p2
    
#     X = np.cross(e01,e12)
#     Y = np.cross(e12,e23)

#     cos = np.dot(X,Y)/(np.linalg.norm(X)*np.linalg.norm(Y))
#     if cos>1:
#         cos=1
#     elif cos<-1:
#         cos=-1
#     if str(cos)=="nan":
#         cos = 1
#     cos_theta = np.arccos(cos)
#     theta = np.rad2deg(cos_theta)
    
#     # 计算正负号
#     XXY = np.cross(X,Y)
#     o_o = np.dot(XXY,e12)/(np.linalg.norm(XXY)*np.linalg.norm(e12))
#     #print(o_o)
#     if o_o <= 0:
#         theta = -theta
        
#     if str(theta)=="nan" or str(theta)=="-nan" :
#         print("p0,p1,p2,p3:", p0,p1,p2,p3)
#         print(theta)
#     if theta >180 or theta<-180:
#         print("p0,p1,p2,p3:", p0,p1,p2,p3)
#         print(theta)
        
#     if theta == 0:
#         sig = 0
#     elif theta > 0:
#         sig = 1
#     elif theta < 0:
#         sig = -1
#     return (abs(theta)/180)-0.5 , sig




# # 内部调用
# def calculate_dihedrals_from_df(series, pos):
#     idx4 = series.values

#     p0 = pos[idx4][0]
#     p1 = pos[idx4][1]
#     p2 = pos[idx4][2]
#     p3 = pos[idx4][3]
    
#     DihedralAngle = getDihedralAngle(p0,p1,p2,p3)
#     return DihedralAngle

# def calculate_dihedrals(data):
#     """
#     返回构象数*二面角数的一个矩阵，每个值都是一个二面角
#     parm:

#     """
#     num = len(data.relativeenergy_list)
    
#     _3Dmols_DihedralAngles = []
    
#     for i in range(num):
#         pos = data.pos[i]
#         mapp = data.mapp
#         convertd_dihedral_pairs_atoms4 = convert_dihedral_index(data.dihedral_pairs_atoms4,mapp)
#         each3Dmol_DihedralAngles = convertd_dihedral_pairs_atoms4.apply(calculate_dihedrals_from_df,pos=pos).values
        
#         _3Dmols_DihedralAngles.append(each3Dmol_DihedralAngles)
        
#     _3Dmols_DihedralAngles_ = torch.tensor(_3Dmols_DihedralAngles, device=device)
    
#     return _3Dmols_DihedralAngles_


# 内部调用
def convert_dihedral_index(dihedral_pairs_atoms4, mapp):
    
    """
    para:
    dihedral_pairs_atoms4:  四原子二面角索引（norm）
    mapp:  对应关系
    
    将dihedral_pairs_atoms4转化为构象来的index，为了下一步的计算二面角
    """
    dihedral_pairs_atoms4_ = pd.DataFrame(dihedral_pairs_atoms4.cpu().numpy())
    mapp_ = np.array(mapp)
    norm = mapp_.transpose()[::-1][0].tolist()
    stru = mapp_.transpose()[::-1][1].tolist()
    dihedral_pairs_atoms4_ = dihedral_pairs_atoms4_.replace(norm, stru)
    return dihedral_pairs_atoms4_


def get_one_bond_nei(bond, mol):
    atom1_idx, atom2_idx = bond
    atom1 = mol.GetAtomWithIdx(atom1_idx)
    atom1_nei = atom1.GetNeighbors()
    atom1_ns = [n.GetIdx() for n in atom1_nei if n.GetIdx()!=atom2_idx]
    atom2 = mol.GetAtomWithIdx(atom2_idx)
    atom2_nei = atom2.GetNeighbors()
    atom2_ns = [n.GetIdx() for n in atom2_nei if n.GetIdx()!=atom1_idx]    
    return atom1_ns, atom2_ns


def theta2sigvalue(theta):
    """convert Angular theta to sig(positivate, minus and zero) and value
    return 
        value: (-0.5,0.5)
        sig: (positivate, minus and zero)
    """
    if theta == 0:
        sig = 0
    elif theta > 0:
        sig = 1
    elif theta < 0:
        sig = -1
    return (abs(theta)/180)-0.5 , sig

def cal_alpha(bond, mol, angle_sys=False):
    atom1_idx, atom2_idx = bond
    atom1_n, atom2_n = get_one_bond_nei(bond, mol)
    tor_list = []
    for f_atom in atom1_n:
        for b_atom in atom2_n:
            each_tor_angle = GetDihedralDeg(mol.GetConformer(), 
                                            f_atom,atom1_idx,atom2_idx,b_atom)
            each_tor_angle_v = angle_vector(each_tor_angle)
            
            tor_list.append(each_tor_angle_v)
    s = 1000000*np.array(tor_list).sum(0)
    
    alpha = -atan2(*s)*180/pi
    
    if angle_sys==False:
        if str(alpha) == "nan":
            return None
        else:
            value , sig = theta2sigvalue(alpha)
        return value , sig   
    else:
        return alpha
            


def get_alphas_of_1conf(convertd_dihedral_pairs_atoms2,mol, angle_sys=False):
    alphas = []
    for i in range(convertd_dihedral_pairs_atoms2.shape[0]):
        bond = convertd_dihedral_pairs_atoms2.iloc[i]
        alpha = cal_alpha(bond, mol, angle_sys=angle_sys)
        alphas.append(alpha)
    return alphas


def get_alphas_of_1mole(convertd_dihedral_pairs_atoms2,mol_list):
    """返回一个分子的全部构象的标准二面角序列
    return： 【num_confs, num_tor】
    """
    return torch.tensor([get_alphas_of_1conf(convertd_dihedral_pairs_atoms2,mol) for mol in mol_list])


# def calculate_dihedrals_norm(mol_list, dihedral_pairs_atoms2, mapp):
#     """
#     返回构象数*二面角数的一个矩阵，每个值都是一个标准化的二面角
#     利用RDKIT函数进行计算二面角
#     parm: 
#         mol_list： 是提取来的构象（序号非smiles的序号），无H，mapp两种序号对应关系
#     """
#     num = len(mol_list)
    
#     _3Dmols_DihedralAngles = []
    
#     for i in range(num):
#         mol = mol_list[i]
#         convertd_dihedral_pairs_atoms2 = convert_dihedral_index(dihedral_pairs_atoms2,mapp)
#         each3Dmol_DihedralAngles = convertd_dihedral_pairs_atoms4.apply(calculate_dihedrals_from_df,pos=pos).values
        
#         _3Dmols_DihedralAngles.append(each3Dmol_DihedralAngles)
        
#     _3Dmols_DihedralAngles_ = torch.tensor(_3Dmols_DihedralAngles, device=device)
    
#     return _3Dmols_DihedralAngles_

def calculate_dihedrals2(data):
    """
    返回构象数*二面角数的一个矩阵，每个值都是一个二面角
    parm:

    """
    mapp = data.mapp
    convertd_dihedral_pairs_atoms2 = convert_dihedral_index(data.dihedral_pairs_atoms2,mapp)
    dihedral_degree = get_alphas_of_1mole(convertd_dihedral_pairs_atoms2,data.mol_list)
    
    return dihedral_degree





def load_smiles(filename):
    with open(filename) as f:
        a = [line.strip('\n') for line in f]
    return a

def count_console(file_path):
    smiles_list_true_sdf_console = load_smiles(file_path)
    valid_idx = []
    # other_idx = []
    for i in smiles_list_true_sdf_console:
        if i.split(" ")[2]=="1":
            valid_idx.append(int(i.split(" ")[0])-1)
            
    return valid_idx



def get_allbonds_rotatiable(smiles):
    mol = Chem.MolFromSmiles(smiles)
#     masks = masks_list_[i]
    
    bonds_rotatiable = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                atom_s, atom_e = bond.GetBeginAtom(), bond.GetEndAtom()
                if len(atom_s.GetNeighbors())>1 and len(atom_e.GetNeighbors())>1:
                    bonds_rotatiable.append([start, end])

                
                
    return bonds_rotatiable

def choose_dihedral_pairs_atoms4(bonds_rotatiable, dihedral_pairs_atoms4):
    bonds_rotatiable_ = torch.tensor(bonds_rotatiable,device = device)
    dihedral_pairs_atoms4=torch.tensor(dihedral_pairs_atoms4,device = device)
    
    choosed_dihedral_pairs_atoms4 = []
    choosed_dih_idx = []
    for i in bonds_rotatiable_:
        i_r = i.index_select(0,torch.tensor([1,0],device = device))
        o_o = dihedral_pairs_atoms4[dihedral_pairs_atoms4[:,[1,2]].eq(i).sum(-1) == 2]
        index = (dihedral_pairs_atoms4[:,[1,2]].eq(i).sum(-1) == 2).nonzero()
        
        if o_o.shape[0] == 0:
            o_o = dihedral_pairs_atoms4[dihedral_pairs_atoms4[:,[1,2]].eq(i_r).sum(-1) == 2]
        
            index = (dihedral_pairs_atoms4[:,[1,2]].eq(i_r).sum(-1) == 2).nonzero()
            
        choosed_dihedral_pairs_atoms4.append(o_o[0])
        choosed_dih_idx.append(index[0][0])
        
    return torch.stack(choosed_dihedral_pairs_atoms4), torch.stack(choosed_dih_idx)

def pad_list(x,lengh,value):
    return  x + [value for i in range(lengh - len(x))]

if __name__ == '__name__':
    
    labels = [
        [13,757,848],
        [883,90,7,2,3,7],
        [7.1,0.3,9,0.4]
    ]
    mask = [
        [[1,2,3,4],[0,2,3,4],[1,3,2,4]],
        [[0,2,3,4],[1,2,3,4],[0,2,3,4],[0,2,3,5],[1,2,4,5],[0,2,3,5]],
        [[1,2,3,5],[1,2,3,4],[0,2,3,4],[0,2,3,5]]
    ]
   
    labels,mask = Data_enhancement(labels,mask)
    print('labels:',labels)
    print('mask:',mask)
    