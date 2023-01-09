import os
import os.path as osp
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
from rdkit.Chem.rdMolAlign import AlignMolConformers, GetBestRMS
import torch
import pickle
import glob
import json
import copy
import time

from utils import get_allbonds_rotatiable
from utils import choose_dihedral_pairs_atoms4
from utils import get_one_bond_nei
from utils import get_alphas_of_1conf

from utils_base import pickle_

from rdkit.Chem.rdMolAlign import AlignMol
import numpy as np
# maxIters = 1000

from rdkit.Chem.rdMolAlign import AlignMol
from rdkit import Chem


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams[ 'figure.dpi' ] = 300 # Make figures have reasonable resolution when exporting


def get_angle(mol_list, mask_):
    """
    给一个分子的多个构象，返回mask_对应的二面角序列
    
    """
    if type(mask_) == torch.Tensor:
        mask_ = mask_.cpu().numpy().tolist()
        
    angles = []
    for mol in mol_list:
        angle = []
        for idx in mask_:

            a = GetDihedralDeg(mol.GetConformer(), idx[0],idx[1],idx[2],idx[3])
            angle.append(a)
        angles.append(angle)
    angles_ = torch.tensor(angles)
    return angles_


# 新的旋转构象函数
# ---------------------------------------------------------------------------------
def rotate_a_mol(mol3D, dihedral_pairs_atoms4, angles_allconfs):
    mol_list = []
    for angles in angles_allconfs:
        mol3D_r = rotate_a_conf(mol3D, dihedral_pairs_atoms4, angles)
        mol_list.append(mol3D_r)
    return mol_list    
        

def rotate_a_conf(mol3D, dihedral_pairs_atoms4, angles):
    """接受一个分子需要旋转的二面角，逐个旋转，返回最终的分子mol，自己写的这个函数不会改变原来的分子"""
    
    mol3D_dcopy = copy.deepcopy(mol3D)
    Chem.SanitizeMol(mol3D_dcopy)
    
    
    dihedral_pairs_atoms4 = dihedral_pairs_atoms4.cpu().numpy().tolist()
    angles = angles.cpu().numpy().tolist()
    for dih_idx, angle in zip(dihedral_pairs_atoms4, angles):

        if str(GetDihedralDeg(mol3D_dcopy.GetConformer(), dih_idx[0],dih_idx[1],dih_idx[2],dih_idx[3])) != "nan" :
            SetDihedralDeg(mol3D_dcopy.GetConformer(), dih_idx[0],dih_idx[1],dih_idx[2],dih_idx[3], angle)
            
        else:
            print(dih_idx, "rdkit计算二面角失败，这个二面角不旋转了")
            
    return mol3D_dcopy
# =======================================================================================

# 新的旋转构象函数
# ---------------------------------------------------------------------------------
def rotate_a_mol_normangle(mol3D, dihedral_pairs_atoms2, angles_allconfs):
    mol3D, dihedral_pairs_atoms4, angles_allconfs_now = convertbeforerotate(mol3D, dihedral_pairs_atoms2, angles_allconfs)
    mol_list = rotate_a_mol(mol3D, torch.tensor(dihedral_pairs_atoms4), angles_allconfs_now)
    return mol_list

def convertbeforerotate(mol3D, dihedral_pairs_atoms2, angles_allconfs):
    dihedral_pairs_atoms4, torsion_oris = dihedral_pairs_atoms2_2_4(dihedral_pairs_atoms2, mol3D)
    gama_allconfs = calculate_gama_allconfs(mol3D, dihedral_pairs_atoms2, angles_allconfs)
    angles_allconfs_now = gama_allconfs2angles_allconfs_now(gama_allconfs, torsion_oris)
    return mol3D, dihedral_pairs_atoms4, angles_allconfs_now

def gama_allconfs2angles_allconfs_now(gama_allconfs, torsion_oris):
    angles_allconfs_now = gama_allconfs + torch.tensor(torsion_oris).to(gama_allconfs.device)
    return angles_allconfs_now
    
def calculate_gama_allconfs(mol3D, dihedral_pairs_atoms2, angles_allconfs):
    dihedral_pairs_atoms2_df = pd.DataFrame(dihedral_pairs_atoms2.cpu()).astype("int")
    alpha_allconfs_ori = get_alphas_of_1conf(dihedral_pairs_atoms2_df,mol3D,angle_sys=True)
    gama_allconfs = angles_allconfs-torch.tensor(alpha_allconfs_ori).to(angles_allconfs.device)   
    return gama_allconfs

def bond2_2_4(bond, mol3D):
    atom1_ns, atom2_ns = get_one_bond_nei(bond.cpu().numpy().tolist(), mol3D)
    atom4idx = int(atom1_ns[0]),int(bond[0]),int(bond[1]),int(atom2_ns[0])  
    torsion = GetDihedralDeg(mol3D.GetConformer(), *atom4idx)
    return atom4idx, torsion
    
    
def dihedral_pairs_atoms2_2_4(dihedral_pairs_atoms2, mol3D):
    """choose the first one torsion as the one ratated, convert atom2idx to atom4idx
    param:   
           dihedral_pairs_atoms2,
           mol3D
    return :
         dihedral_pairs_atoms4
    """
    dihedral_pairs_atoms4=[]
    torsion_oris =[]
    for bond in dihedral_pairs_atoms2:
        bond4idx, torsion_ori = bond2_2_4(bond, mol3D)
        dihedral_pairs_atoms4.append(bond4idx)
        torsion_oris.append(torsion_ori)
        
    return dihedral_pairs_atoms4, torsion_oris
# =======================================================================================



def plot_tor_seq(tensor, cp="g", cl="salmon"):
    fig, ax1 = plt.subplots(figsize=(20,10))
    
    for i in range(len(tensor)):
        y = tensor[i].cpu()
        x = range(len(y))
        ax1.scatter(x,y,color=cp)
        ax1.plot(x,y,color=cl)

    plt.legend(prop={'family':'SimHei','size':30})

def cheek_roedbond_pre_con(smiles, masks,tensor):
    """取出来预测或labels中的可旋转的那部分"""
    bonds_rotatiable = get_allbonds_rotatiable(smiles)
    choosed_dihedral_pairs_atoms4,choosed_dih_idx = choose_dihedral_pairs_atoms4(bonds_rotatiable, masks)
    tensor_ = tensor[:,choosed_dih_idx,:]
    return tensor_


def RMSD_matrix(mols, maxIters =1000):
    if type(mols) == Chem.rdchem.Mol:
        num = len(mols.GetConformers())
        RMSD_m = np.zeros((num,num))
        for i in range(num):
            for j in range(num):
                RMSD_m[i,j] = AlignMol(mols, mols, prbCid=i, refCid=j, maxIters = maxIters)
                
    else:
        num = len(mols)
        RMSD_m = np.zeros((num,num))
        for i in range(num):
            for j in range(num):
                RMSD_m[i,j] = AlignMol(mols[i], mols[j], maxIters = maxIters)  
                
    return RMSD_m



def looklook():
    """can't run!!!!!!
    调试用的，查看一下预测值，真实值的相差等等"""
    prediction_list_tensor = torch.tensor(prediction_list)
    prediction_list_tensor.shape
    prediction_list_tensor[:,:,1]
    
    count_di(prediction_list_tensor[:,:,1])
    data.dihedral_degree.shape
    count_di(data.dihedral_degree[:,1,:])
    data.dihedral_degree[:,0,:][:,52]
    prediction_list_tensor[:,:,0][:,52]


def count_di(tensor_):
    """计算有多少个不同的值"""
    li = []
    for i in tensor_:
        li.append(str(i.cpu().detach().numpy().tolist()))
    return len(li),len(set(li))




def sort_confs_by_RMSD(mol_list, maxIters=1000):
    """这个函数接受一个mol的列表，返回按照与第一个分子的rmsd值排序的列表"""
#     print(mol_list)
    RMSDlist = []
    for i in range(len(mol_list)):
        RMSD = AlignMol(mol_list[0], mol_list[i], maxIters = maxIters)
        RMSDlist.append(RMSD)

    idx_sort = np.argsort(RMSDlist)
#     print(idx_sort)
    mol_list = np.array(mol_list)[idx_sort].tolist()
#     print(mol_list)
    
    return mol_list

def genrate_con_from_roedangles_n(prediction_list, masks_list, relativeenergys_list, smiles_list,
                               sdf_input_path,
                               write_root, device, AlignmaxIters = 1000):
    """这里的write_root 其实是一个根目录"""
            
    suppl = get_suppl_from_sdf(sdf_input_path)
    mol_ref = Chem.AddHs(Chem.MolFromSmiles(smiles_list[0]))
    mol_list = []
    
    for i in range(len(smiles_list)):
        prediction = prediction_list[i]
        prediction = torch.tensor(prediction)
        prediction = (prediction[:,0]+0.5)*180*(prediction[:,1]-1)

        smiles = smiles_list[i]
        masks = torch.tensor(masks_list[i]).to(device)
        mol3D = suppl[0]
        Chem.SanitizeMol(mol3D)
        relativeenergy = relativeenergys_list[i]
        genrate_con_from_roedangles(smiles, masks, prediction, mol3D, relativeenergy, write_root)
        
        mol_ref.AddConformer(mol3D.GetConformer(),assignId=True) 
        
#         mol_list.append(mol3D)
    
    # 生成的构象保存到sdf文件中
    writosdf(mol_ref,write_root,f"{smiles_list[0]}_roted_confs.sdf".replace("/", "_"))
    print(f"将mol_ref: {mol_ref} 写入文件{write_root}/",f"{smiles_list[0]}_roted_confs.sdf".replace("/", "_"),"中了")
#     # 对齐构象并保存到sdf文件中
#     AlignMolConformers(mol_ref,maxIters=AlignmaxIters)
#     writosdf(mol_ref,write_root,f"{smiles_list[0]}_roted_confs_aligned.sdf".replace("/", "_"))

#     # 对齐构象，并按照与第一个构象的相似程度对构象集进行排序，并保存到文件中
#     mol_list_ = sort_confs_by_RMSD(mol_list, maxIters=1000)
#     mol_ref = aggr_mol(mol_list_)  # 整合为一个mol
#     AlignMolConformers(mol_ref,maxIters=AlignmaxIters)
#     writosdf(mol_ref,write_root,f"{smiles_list[0]}_roted_confs_aligned_sorted.sdf".replace("/", "_"))
#     return mol_list, mol_list_

def get_suppl_from_sdf(sdf_input_path, removeHs=False):
    suppl = Chem.SDMolSupplier(sdf_input_path, removeHs)
    print("len(suppl)",len(suppl))
    return suppl

def visualization_mole(mole):
    """可视化分子，接收一个smiles 或者 mol  ，显示分子的2D图形
    
        return： None
    """
    try:
        mol = Chem.MolFromSmiles(mole)
    except TypeError as e:
        mol = mole
    for i,atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)
    display(mol)


def genrate_con_from_roedangles(smiles, masks, prediction, mol3D, relativeenergy, write_root): 
    """
    从给定初始构象通过设定可旋转单键二面角角度来生成构象（使用rdkit函数：SetDihedralDeg）
    pram: 
        smiles:
        masks:          模型预测出来的二面角对
        prediction:     所对应的角度值
        mol3D:          sdf文件读取来的3d结构mol
        write_path:     扭转后的小分子写入的文件路径：*.sdf
    """
    print(f"开始扭转{smiles}")
    try:
        os.makedirs(osp.join(write_root, '{}'.format(smiles.replace("/","_")) ))
    except FileExistsError as e:
        print(smiles,"文件夹已经存在")
    write_path = osp.join(write_root, '{}/rotated_con_re{}.sdf'.format(smiles.replace("/","_"), relativeenergy) )
    writer = Chem.SDWriter(write_path)
    
#     mol = Chem.MolFromSmiles(smiles)
#     for i,atom in enumerate(mol.GetAtoms()):
#         atom.SetAtomMapNum(i)
#     display(mol)
    
    bonds_rotatiable = get_allbonds_rotatiable(smiles)
    print("bonds_rotatiable:", bonds_rotatiable)
    if len(bonds_rotatiable) == 0:
        choosed_dihedral_pairs_atoms4 = torch.tensor([])
        choosed_dih_idx = torch.tensor([])
    else:
        try:
            choosed_dihedral_pairs_atoms4,choosed_dih_idx = choose_dihedral_pairs_atoms4(bonds_rotatiable, masks)
        except IndexError as error:
            choosed_dihedral_pairs_atoms4 = torch.tensor([])
            choosed_dih_idx = torch.tensor([])
            print("IndexError")
    print("choosed_dihedral_pairs_atoms4:",choosed_dihedral_pairs_atoms4)
    print("choosed_dih_idx:", choosed_dih_idx)
    if len(choosed_dih_idx) == 0:
        angles = torch.tensor([])
    else:
        angles = torch.tensor(prediction)[choosed_dih_idx]
    # print("angles:", angles)

    angles_l = angles.cpu().numpy().tolist()

    print("angles_l:", angles_l)
    # 旋转单键
    for j, dih_idx in enumerate(choosed_dihedral_pairs_atoms4):

        dih_idx = dih_idx.cpu().numpy().tolist()


        if str(GetDihedralDeg(mol3D.GetConformer(), dih_idx[0],dih_idx[1],dih_idx[2],dih_idx[3])) != "nan" :
            SetDihedralDeg(mol3D.GetConformer(), dih_idx[0],dih_idx[1],dih_idx[2],dih_idx[3], angles_l[j])

    # 写入文件
    writer.write(mol3D)
    print("写入文件{}".format(smiles))

    writer.close()
    
    
    
    
def open_pickle(mol_path):
    with open(mol_path, "rb") as f:
        dic = pickle.load(f)
    return dic

#==============================================================================================
# 将mol_list  写入文件中，格式为一个smiles一个文件夹，里面包括若干个不同相对能量的构象
def write_mollisttosdfdir(root, smiles, mol_list, relativeenergy_list):
    
    if not os.path.exists(root):
        os.makedirs(root)
        
    smiles = smiles.replace("/","_")
    smiles_dir_name = osp.join(root,smiles)
    
    mol_re_list = list(zip(mol_list,relativeenergy_list))
    for mol_re in mol_re_list:
        write_moltosdf(smiles_dir_name, mol_re[0], mol_re[1])

# # # CanonicalizeMol 规范化后的分子3d坐标写入sdf文件
# # mol_list_can = [CanonicalizeMol(each["rd_mol"]) for each in mol_dic["conformers"]]
# write_mollisttosdfdir(save_root, smiles, mol_list, relativeenergy_list)

import os
def write_moltosdf(root, mol, relativeenergy):
    # 将mol(一个构象)导出为sdf，保存在root目录下，如果root不存在则递归创建
    """
    root: 要写入 的文件夹
    mol： 有一个构象的3D分子
    relativeenergy：相对能量
    """
    if not os.path.exists(root):
        os.makedirs(root)
    with open(osp.join(root, f'true_con_re_{relativeenergy}.sdf'),'w+') as file:
        print(Chem.MolToMolBlock(mol),file=file)
        
# ==========================================================================================================



def extract_align_save(chem_dataset_dir, smiles, to_root, sdf_name, maxIters=2000):
    """
    提取chem数据文件夹（chem_dataset_dir）中提取感兴趣分子（smiles）的所有构象, 对齐之后保存到to_root文件中
    
    return ： None
    """
    mol_list, relativeenergy_list = extract_mol_from_chemdir(chem_dataset_dir, smiles)
    # 整合为一个mol
    mol_ref = aggr_mol(mol_list)
   
    # 对齐所有的构象
    AlignMolConformers(mol_ref,maxIters=maxIters)
    
    # 写入一个sdf文件
    writosdf(mol_ref,to_root,sdf_name)
    
    
    
def open_summery(chem_dataset_dir):
    # 打开summery文件
    summary_file = glob.glob(os.path.join(chem_dataset_dir, "summary*.json"))[0]
    with open(summary_file, "r") as f:
        summ = json.load(f)
        
    return summ
    
    
def extract_mol_dic_from_chemdir(chem_dataset_dir, smiles):
    """
     从提取chem数据文件夹中提取感兴趣分子（smiles）的构象:mol_dic
     chem_dataset_dir:  "*/drugs or qm9"
     smiles: 感兴趣的分子的smiles
     
     return: mol_dic
    """
    # 打开summery文件
    summary_file = glob.glob(os.path.join(chem_dataset_dir, "summary*.json"))[0]
    with open(summary_file, "r") as f:
        summ = json.load(f)
#     print(f"{chem_dataset_dir}有{len(summ)}个smiles")
    # 获取到感兴趣分子对应的pickle文件
    smiles_con_path_ = summ[smiles]['pickle_path'].split("/")[-1]
    smiles_con_path = osp.join(chem_dataset_dir,smiles_con_path_)
    # 打开pickle文件
    mol_dic = open_pickle(smiles_con_path)
    mol_list = [each["rd_mol"] for each in mol_dic["conformers"]]
    relativeenergy_list = [each["relativeenergy"] for each in mol_dic["conformers"]]
        
    return mol_dic 
def extract_mol_from_chemdir(chem_dataset_dir, smiles):
    """
     从提取chem数据文件夹中提取感兴趣分子（smiles）的所有构象及其对应的相对能量
     chem_dataset_dir:  "*/drugs or qm9"
     smiles: 感兴趣的分子的smiles
     
     return: mol_list,relativeenergy_list
    """
    # 打开summery文件
    summary_file = glob.glob(os.path.join(chem_dataset_dir, "summary*.json"))[0]
    with open(summary_file, "r") as f:
        summ = json.load(f)
#     print(f"{chem_dataset_dir}有{len(summ)}个smiles")
    # 获取到感兴趣分子对应的pickle文件
    smiles_con_path_ = summ[smiles]['pickle_path'].split("/")[-1]
    smiles_con_path = osp.join(chem_dataset_dir,smiles_con_path_)
    # 打开pickle文件
    mol_dic = open_pickle(smiles_con_path)
    mol_list = [each["rd_mol"] for each in mol_dic["conformers"]]
    relativeenergy_list = [each["relativeenergy"] for each in mol_dic["conformers"]]
    
    return mol_list,relativeenergy_list

def extract_mol_from_chemdir_(chem_dataset_dir, smiles):
    """
     从提取chem数据文件夹中提取感兴趣分子（smiles）的所有构象及其对应的相对能量
     chem_dataset_dir:  "*/drugs or qm9"
     smiles: 感兴趣的分子的smiles
     
     return: mol_list,relativeenergy_list
    """
    # 打开summery文件
    summary_file = glob.glob(os.path.join(chem_dataset_dir, "summary*.json"))[0]
    with open(summary_file, "r") as f:
        summ = json.load(f)
#     print(f"{chem_dataset_dir}有{len(summ)}个smiles")
    # 获取到感兴趣分子对应的pickle文件
    smiles_con_path_ = summ[smiles]['pickle_path'].split("/")[-1]
    smiles_con_path = osp.join(chem_dataset_dir,smiles_con_path_)
    # 打开pickle文件
    mol_dic = open_pickle(smiles_con_path)
    
    
    conformers = mol_dic['conformers']
    rdmols = [conf["rd_mol"] for conf in conformers]
    # [len(rdmol.GetConformers()) for rdmol in rdmols]  每个rdmol都是一个构象
    positions = [rdmol.GetConformer().GetPositions() for rdmol in rdmols]
    
    relativeenergy_list = [conf["relativeenergy"] for conf in conformers]
    
    return mol_dic, conformers, rdmols, positions, relativeenergy_list

def aggr_mol(mol_list):
    mol_ref = copy.deepcopy(mol_list[0])
    for mol in mol_list[1:]:
        mol_ref.AddConformer(mol.GetConformer(),assignId=True)
    return mol_ref 

def writosdf(mol,to_root, sdf_name, Align=False, maxIters = 1000):
    """
    将mol_list或者一个mol中的多个构象保存到一个sdf文件（osp.join(to_root, sdf_name)）中
    mol: 可以是mol或者mol_list
    to_root: 是要保存到的文件夹位置"""
    
    try:
        os.makedirs(to_root)
    except FileExistsError as e:
        print(to_root,"文件夹已经存在")
    write_root = osp.join(to_root, sdf_name)    
    
    writer = Chem.SDWriter(write_root)
    if type(mol)==list or type(mol)==tuple:
        mol = aggr_mol(mol)
    if Align == True:
        AlignMolConformers(mol,maxIters=maxIters)
    for cid in range(mol.GetNumConformers()):
        writer.write(mol, confId=cid)

    writer.close()
    
# 自己定义的  COV  和  AMR
# ————————————————————————————————————————
def COV(true_mol_list, predict_mol_list, threshold = 1.25, maxIters = 1000):
    true = 0
    for true_mol in true_mol_list:
        for predict_mol in predict_mol_list:
            Chem.SanitizeMol(true_mol)
            Chem.SanitizeMol(predict_mol)
            
#             print(AlignMol(true_mol, predict_mol))
            try:
                if AlignMol(Chem.RemoveAllHs(true_mol), Chem.RemoveAllHs(predict_mol), maxIters = maxIters) < threshold:
                    true+=1
                    break
            except RuntimeError as e:
                print("COV计算出现问题：RuntimeError: No sub-structure match found between the probe and query mol")
                return None
    return true/len(true_mol_list)



def AMR(true_mol_list, predict_mol_list):
    RMSD=0
    for predict_mol in predict_mol_list:
        RMSD_min = min([AlignMol(true_mol, predict_mol) for true_mol in true_mol_list])
#         print(RMSD_min)
        RMSD+=RMSD_min
#     print(len(predict_mol_list))
    return RMSD/len(predict_mol_list) 
# ===============================================================================


# geodiff 的COV MAT方法
# ----------------------------------------------------------------------------
def get_best_rmsd(mol1, mol2):
    probe = Chem.RemoveHs(mol1)
    ref = Chem.RemoveHs(mol2)
    rmsd = GetBestRMS(probe, ref)
    return rmsd
def get_rmsd_confusion_matrix(mol_list1, mol_list2):

    num_mol1 = len(mol_list1)
    num_mol2 = len(mol_list2)

    rmsd_confusion_mat = -1 * np.ones([num_mol1, num_mol2],dtype=np.float)
    
    for i in range(num_mol1):
        for j in range(num_mol2):
            rmsd_confusion_mat[i,j] = get_best_rmsd(mol_list1[i], mol_list2[j])    
    
    return rmsd_confusion_mat

def get_onemol_COV_MAT(rmsd_confusion_mat, threshold = 1.25):
    rmsd_list1_min = rmsd_confusion_mat.min(-1)     # # np (num_mol1, )
    rmsd_list2_min = rmsd_confusion_mat.min(0)     # # np (num_mol2, )
    
    mat1 = rmsd_list1_min.mean()
    mat2 = rmsd_list2_min.mean()
    
    cov1 = (rmsd_list1_min <= threshold).mean()
    cov2 = (rmsd_list2_min <= threshold).mean()
    
    return mat1,mat2,cov1,cov2

def get_onemol_COV_MAT_o(mol_list1, mol_list2, threshold = 1.25):
    rmsd_confusion_mat = get_rmsd_confusion_matrix(mol_list1, mol_list2)
    mat1,mat2,cov1,cov2 = get_onemol_COV_MAT(rmsd_confusion_mat, threshold = threshold)
    return mat1,mat2,cov1,cov2

# ===============================================================================

def sorted_mol_list_pred(mol_list_true, mol_list_pred):

    rmsd_confusion_mat = get_rmsd_confusion_matrix(mol_list_true, mol_list_pred)
    idx = rmsd_confusion_mat.argmin(-1)
    mol_list_pred_sorted = list(np.array(mol_list_pred)[idx])
    
    return mol_list_pred_sorted, idx



def average_atoms(atoms_idx, atom_feat, device):
#     list = []
#     for atom in atoms:
    atoms_idx = torch.tensor(atoms_idx).to(device)
    average_feat = atom_feat[atoms_idx].mean(0)
    return average_feat
def average_edges(bonds, bond_feat):
    list_ = []
    for bond in bonds:
        feat = bond_feat[bond]
        list_.append(feat)
        average_feat = torch.stack(list_).mean(0)
    return average_feat


# ------------------------------------------------------------------------------------------
def load_4files(path, ol, nam = " (1)"):
    """load four list files"""
 
    # 文件路径
    smiles_list_path = osp.join(path, ol, f"smiles_list{nam}.pkl" )          
    torsion_idx_list_path = osp.join(path, ol, f"torsion_idx_list{nam}.pkl")

    torsion_list_path = osp.join(path, ol, f"torsion_list{nam}.pkl")
    relativeenergy_list_path = osp.join(path, ol, f"relativeenergy_list{nam}.pkl")

    print("开始读取加载数据")
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    smilelist = pickle_.load(smiles_list_path)
    torsion_angle = pickle_.load(torsion_list_path)
    torsion_list = pickle_.load(torsion_idx_list_path)
    relativeenergy_list = pickle_.load(relativeenergy_list_path)

    # torsion_angle = [list(zip(torsion[0],(np.array(torsion[1])+1).tolist())) for torsion in torsion_angle]

    print("加载完成")
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    print(len(smilelist), len(torsion_angle), len(torsion_list), len(relativeenergy_list))
    return smilelist, torsion_angle, torsion_list, relativeenergy_list 

#===================================================================

def df_groupby(df, by):
    """
    按某列分组并求平均值
    param:
        df : 
                angle_group_num	num_confs_list	matr	matp	covr	covp
                249	1.0	1.0	0.187517	0.381681	1.000000	1.000000
                328	1.0	2.0	0.208589	0.209407	1.000000	1.000000
                222	1.0	2.0	0.279021	0.351168	1.000000	1.000000
                454	1.0	2.0	0.276246	0.250742	1.000000	1.000000
                333	1.0	1.0	0.707969	0.728104	1.000000	1.000000
                ...	...	...	...	...	...	...
                385	13.0	136.0	1.489988	2.138231	0.095588	0.011029
                275	13.0	300.0	1.492999	1.941721	0.139785	0.051667
                184	13.0	300.0	1.538483	1.836178	0.059459	0.025000
                123	14.0	204.0	1.601289	2.174254	0.009804	0.004902
                334	14.0	146.0	2.630047	3.304267	0.000000	0.000000

        by: eg: "angle_group_num"  or   "num_confs_list"
    """
    # df_ = df.sort_values(by = "angle_group_num")
    df_g = df.groupby(by).mean()
    return df_g