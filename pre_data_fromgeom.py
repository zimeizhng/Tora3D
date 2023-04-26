from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

import os.path as osp
import numpy as np
import glob
import pickle
import random
import tqdm

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Dataset, Data


from utils import get_dihedral_pairs
from utils import norm_inchi_index
from utils import get_dihedral_pairs_atoms4
from utils import get_tense_dihedral_pairs_atoms4

from utils import calculate_dihedrals

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')

class geom_confs(Dataset):
    def __init__(self, root, split_path, mode, get_dih_mothed, transform=None, pre_transform=None, max_confs=10):
        super(geom_confs, self).__init__(root, transform, pre_transform)

        self.root = root
        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(split_path, allow_pickle=True)[self.split_idx]
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        self.dihedral_pairs = {} # for memoization
        all_files = sorted(glob.glob(osp.join(self.root, '*.pickle')))
        self.pickle_files = [f for i, f in enumerate(all_files) if i in self.split]
        self.max_confs = max_confs
        
        self.data_dic = {}
        
        if get_dih_mothed == "sparse":
            self.get_dihedral_pairs_atoms4_ = get_dihedral_pairs_atoms4
        elif get_dih_mothed == "tense":
            self.get_dihedral_pairs_atoms4_ = get_tense_dihedral_pairs_atoms4

    def len(self):
        # return len(self.pickle_files)  # should we change this to an integer for random sampling?
        return 200000 if self.split_idx == 0 else 20000

    def get(self, idx):
        """
        return data
        data.mapp_list tupple 表示对应关系，里面前面的是3d构象的原子索引，后面的是标准smiles 原子索引
        """
        data = None
        while not data:
            pickle_file = random.choice(self.pickle_files)
            mol_dic = self.open_pickle(pickle_file)
            data = self.featurize_mol(mol_dic)
        if idx in self.dihedral_pairs:
            data.edge_index_dihedral_pairs = self.dihedral_pairs[idx]
        else:
            data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)
            data.dihedral_pairs_atoms4 = self.get_dihedral_pairs_atoms4_(data)
            data.dihedral_degree = calculate_dihedrals(data)
        return data

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):  
        
        relativeenergy_list = [dic["relativeenergy"] for dic in mol_dic["conformers"]]
        idx_sort = np.argsort(relativeenergy_list)
        mol_dic["conformers"] = np.array(mol_dic["conformers"])[idx_sort].tolist()
        
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        if mol_:
            canonical_smi = Chem.MolToSmiles(mol_)
        else:
            return None

        if '.' in name:
            return None

        num_atom = Chem.MolFromSmiles(canonical_smi).GetNumAtoms()
        if num_atom < 4:
            return None
        if Chem.MolFromSmiles(canonical_smi).GetNumBonds() < 4:
            return None
        if not Chem.MolFromSmiles(canonical_smi).HasSubstructMatch(dihedral_pattern):
            return None

        pos = torch.zeros([self.max_confs, num_atom, 3])
        pos_mask = torch.zeros(self.max_confs, dtype=torch.int64)
        mapp_list = []
        relativeenergy_list = []
        mol_list = []
        k = 0
        for conf in confs:
            mol = conf['rd_mol']

            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception as e:
                continue

            if conf_canonical_smi != canonical_smi:
                continue

                        
            mm_map = norm_inchi_index(Chem.RemoveAllHs(mol))
            mm_map_c = norm_inchi_index(Chem.MolFromSmiles(canonical_smi))

            
            mapp = list(zip(mm_map,mm_map_c))
            
            
            pos[k] = torch.tensor(Chem.RemoveAllHs(mol).GetConformer().GetPositions(), dtype=torch.float)
            pos_mask[k] = 1
            mapp_list.append(mapp)
            relativeenergy_list.append(conf['relativeenergy'])
            mol_list.append(Chem.RemoveAllHs(mol))
            
            k += 1
            if k == self.max_confs:
                break

        if k == 0:
            return None

        row, col = [], []
        for bond in Chem.MolFromSmiles(canonical_smi).GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

        edge_index = torch.tensor([row, col], dtype=torch.long)

        perm = (edge_index[0] * num_atom + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]

        data = Data(pos=pos, pos_mask=pos_mask, mapp_list=mapp_list, 
                    relativeenergy_list=relativeenergy_list, 
#                     mol_list=mol_list,
                    edge_index=edge_index, name=canonical_smi )
        return data