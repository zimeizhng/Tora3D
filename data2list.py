from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

import os
import os.path as osp
import numpy as np
import glob
import pickle
import json
import random
import tqdm
import time

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data

from utils import norm_inchi_index

from utils import calculate_dihedrals2
from utils import get_allbonds_rotatiable


from utils_others import aggr_mol,writosdf,sort_confs_by_RMSD
from utils_base import pickle_

from rdkit.Chem.rdMolAlign import AlignMolConformers


dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')

class geom_confs(Dataset):
    def __init__(self, root, get_dih_mothed="preciseone", save_data_path=None, save_every = 10000, transform=None, pre_transform=None, write=True, appoint_mole=None, save_con_path=None, AlignmaxIters=1000, device="cuda"):
<<<<<<< HEAD

=======
        # root：geom分子构象数据文件存放位置
        # appoint_mole： 决定是否仅返回感兴趣分子的四个list， 默认为None, 可以传入smiles list
        # save_con_path： 决定是否将分子的真实构象保存到sdf文件中， 默认为None，可以传入  "*.sdf“
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
        super(geom_confs, self).__init__(root, transform, pre_transform)

        self.root = root
        self.write = write
        self.AlignmaxIters = AlignmaxIters
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        self.save_every = save_every
        self.device = device
        
        # ----------------------------------------------------
        # 准备self.all_files
<<<<<<< HEAD
        if appoint_mole==None:    # 
            self.all_files = glob.glob(osp.join(self.root, '*.pickle'))
        elif appoint_mole!=None:   # 
=======
        if appoint_mole==None:    # 全部
            self.all_files = glob.glob(osp.join(self.root, '*.pickle'))
        elif appoint_mole!=None:   # 指定的几个感兴趣的分子
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
            self.all_files = []
            summary_file = glob.glob(os.path.join(self.root, "summary*.json"))[0]
            with open(summary_file, "r") as f:
                summ = json.load(f)
            for smiles in appoint_mole:
                smiles_con_path_ = summ[smiles]['pickle_path'].split("/")[-1]
                smiles_con_path = osp.join(self.root, smiles_con_path_)
                self.all_files.append(smiles_con_path)
        # ==================================================
        
        # ----------------------------------------------------
<<<<<<< HEAD
        # 
=======
        # 确定是否返回真实构象
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
        self.save_con_path = save_con_path
        if self.save_con_path is not None:
            if not os.path.exists(self.save_con_path):
                os.makedirs(self.save_con_path) 
        # ==================================================
    
        # ----------------------------------------------------
<<<<<<< HEAD
        # 
        if self.write ==True:
            if save_data_path==None:
                raise ValueError(" when write is true, must have save_data_path")   
=======
        # 确定是否将四个list写入文件
        if self.write ==True:
            if save_data_path==None:
                raise ValueError(" write为true时， 必须传入参数：save_data_path")   
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
            if save_data_path!=None:
                if not os.path.exists(save_data_path):
                    os.makedirs(save_data_path)  
                    
            self.smiles_list = []
            self.torsion_idx_list = []
            self.torsion_list = []
            self.relativeenergy_list = []
            
            self.smiles_list_file =         osp.join(save_data_path, "smiles_list.pkl")
            self.torsion_idx_list_file =    osp.join(save_data_path, "torsion_idx_list.pkl")       
            self.torsion_list_file =        osp.join(save_data_path, "torsion_list.pkl")        
            self.relativeenergy_list_file = osp.join(save_data_path, "relativeenergy_list.pkl")       
         # ==================================================
        
    def len(self):
        # return len(self.pickle_files)  # should we change this to an integer for random sampling?
        return len(self.all_files)

    def get(self, idx):
#         print(idx)
        """
<<<<<<< HEAD

=======
        return data
        data.mapp:  list of tupple 表示对应关系，里面前面的是3d构象的原子索引，后面的是标准smiles 原子索引
        data.mol_list:  过滤以后的mol_list, 无H
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
        """

        pickle_file = self.all_files[idx]
        mol_dic = self.open_pickle(pickle_file)
        data = self.featurize_mol(mol_dic)
        
        if data:
            
            # data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)
            # data.dihedral_pairs_atoms4 = self.get_dihedral_pairs_atoms4_(data.name)
            
            data.dihedral_pairs_atoms2 = torch.tensor(get_allbonds_rotatiable(data.name)).to(self.device)
<<<<<<< HEAD
=======
            # data.dihedral_pairs_atoms2  [可旋转键数，2]
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
            data.dihedral_degree = calculate_dihedrals2(data)
            
            if self.write ==True:
                self.smiles_list.append(data.name)
                self.torsion_idx_list.append(data.dihedral_pairs_atoms2)   # data.dihedral_pairs_atoms4: torch.Size([4, num_tor])
                self.torsion_list.append(data.dihedral_degree) # data.dihedral_degree: torch.Size([num_confs, 2, num_tor])
                self.relativeenergy_list.append(data.relativeenergy_list) # data.relativeenergy_list : len = num_confs
                
                if idx % self.save_every == 0:
                    pickle_.save(self.smiles_list_file, self.smiles_list)
                    pickle_.save(self.torsion_idx_list_file, self.torsion_idx_list)
                    pickle_.save(self.torsion_list_file, self.torsion_list)
                    pickle_.save(self.relativeenergy_list_file, self.relativeenergy_list)
                """
                four lists : same len , 
                               each: 
                                  smiles_list: smiles, 
                                  torsion_idx_list:eg: tensor([[1, 2],
                                                     [2, 3],
                                                     [3, 5],
                                                     [6, 7],
                                                     [8, 9]], device='cuda:0')
                                  torsion_list:  eg:  tensor([[[-0.4606, -1.0000],
                                                      [ 0.4924, -1.0000],
                                                      [-0.1182,  1.0000],
                                                      [ 0.4647,  1.0000],
                                                      [-0.0297, -1.0000]],
                                                      .
                                                      .
                                                      .]
                                             first is value (-0.5,0.5), last is sig -1,0,1
                                  relativeenergy_list : 
                                             eg: [4.697, 4.82, 5.141]  there confs
                """
                
            else:
                data.smiles_list_f = []
                data.torsion_idx_list_f = []
                data.torsion_list_f = []
                data.relativeenergy_list_f = []
                
                for i in range(len(data.relativeenergy_list)):
                    data.smiles_list_f.append(data.name)
                    data.torsion_idx_list_f.append(data.dihedral_pairs_atoms4.t().cpu().numpy().tolist())
                    data.torsion_list_f.append(data.dihedral_degree[i].cpu().numpy().tolist())
                    data.relativeenergy_list_f.append(data.relativeenergy_list[i])
            if self.save_con_path:
#                 mol_list_ = sort_confs_by_RMSD(data.mol_list, maxIters=self.AlignmaxIters)

<<<<<<< HEAD
                # 
=======
                # 写入一个sdf文件
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
                writosdf(data.mol_list,self.save_con_path,f"{data.name}_true_confs_aligned.sdf".replace("/", "_"), Align=True, maxIters = self.AlignmaxIters)
                
        return data

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
        
        
<<<<<<< HEAD
        # 
=======
        # 烦死了，mol_dic还得重新排序一下，有的不是按照relativeenergy来排的
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
        relativeenergy_list = [dic["relativeenergy"] for dic in mol_dic["conformers"]]
        idx_sort = np.argsort(relativeenergy_list)
        mol_dic["conformers"] = np.array(mol_dic["conformers"])[idx_sort].tolist()
        
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]
        

        # filter mols rdkit can't intrinsically handle
        mol_ = Chem.MolFromSmiles(name)
        if mol_:
            canonical_smi = Chem.MolToSmiles(mol_)
        else:
            print("mol_:", mol_)
            return None
        # filter mol that don't have rotatiable bonds
        if len(get_allbonds_rotatiable(canonical_smi))==0:
<<<<<<< HEAD
            print("")
=======
            print("没有可旋转键")
>>>>>>> ead8aac572b4bfbf5e25b5638f0da5f049708d5f
            return None
        # skip conformers with fragments
        if '.' in name:
            print("'.' in name")
            return None

        # skip conformers without dihedrals
        num_atom = Chem.MolFromSmiles(canonical_smi).GetNumAtoms()
        if num_atom < 4:
            print("num_atom < 4")
            return None
        if Chem.MolFromSmiles(canonical_smi).GetNumBonds() < 4:
            print("NumBonds() < 4")
            return None
        if not Chem.MolFromSmiles(canonical_smi).HasSubstructMatch(dihedral_pattern):
            print("not Chem.MolFromSmiles(canonical_smi).HasSubstructMatch(dihedral_pattern):")
            return None


#         pos = torch.zeros([len(confs), num_atom, 3], device=self.device)
#         pos_mask = torch.zeros(self.max_confs, dtype=torch.int64)
        relativeenergy_list = []
        mol_list = []
#         k = 0
#         print("len(confs):",len(confs))
        for conf in confs:
            mol = conf['rd_mol']

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception as e:
                print("conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))")
                continue
            
            if conf_canonical_smi != canonical_smi:
#                 print(conf_canonical_smi)
#                 print(canonical_smi)
#                 print()
                continue
            
            
#             pos[k] = torch.tensor(Chem.RemoveAllHs(mol).GetConformer().GetPositions(), dtype=torch.float)
#             pos_mask[k] = 1
            
            relativeenergy_list.append(conf['relativeenergy'])
            mol_list.append(Chem.RemoveAllHs(mol))
            
#             k += 1
#             print(f"  k: {k}")
#             if k == self.max_confs:
#                 break

        # return None if no non-reactive conformers were found
        if len(mol_list) == 0:
            print("no non-reactive conformers were found")
            return None
        
        # 确定映射
        try:            
            mm_map = norm_inchi_index(Chem.RemoveAllHs(mol_list[0]))
            print("-") 
            mm_map_c = norm_inchi_index(Chem.MolFromSmiles(canonical_smi))
            print("-")
        except ValueError as e:
            return None            
        mapp = list(zip(mm_map,mm_map_c))


        row, col = [], []
        for bond in Chem.MolFromSmiles(canonical_smi).GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

        edge_index = torch.tensor([row, col], dtype=torch.long, device = self.device)

        perm = (edge_index[0] * num_atom + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]

        data = Data(
#                     pos=pos,
#                     pos_mask=pos_mask, 
                    mapp=mapp, 
                    relativeenergy_list=relativeenergy_list, 
                    mol_list=mol_list,
                    edge_index=edge_index, name=canonical_smi )
        return data