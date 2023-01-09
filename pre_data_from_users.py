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
from utils import get_tense_dihedral_pairs_atoms4
from utils import calculate_dihedrals
from utils import load_smiles

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')

class geom_confs(Dataset):
    def __init__(self, root,relativeenergy,transform=None, pre_transform=None, max_confs=10):
        super(geom_confs, self).__init__(root, transform, pre_transform)

        self.root = root
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        self.smiles_list = load_smiles(self.root)
        self.relativeenergy = relativeenergy

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        """
        return data

        """
        data = None
        while True:
            smiles = self.smiles_list[idx]
            relativeenergy_ = self.relativeenergy[idx]
            data = self.featurize_mol(smiles,relativeenergy_)
            if data is None:
                self.smiles_list.pop(idx)
            else:
                break   
     
        if data is not None:
            data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)
            data.dihedral_pairs_atoms4 = get_tense_dihedral_pairs_atoms4(data)
            data.dihedral_degree = torch.zeros(data.dihedral_pairs_atoms4.shape[-1])
        return data

    def featurize_mol(self, smiles,relativeenergy_):
        name = smiles
        
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

        row, col = [], []
        for bond in Chem.MolFromSmiles(canonical_smi).GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

        edge_index = torch.tensor([row, col], dtype=torch.long)

        perm = (edge_index[0] * num_atom + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]

        data = Data(
                    relativeenergy=relativeenergy_, 
                    edge_index=edge_index, name=canonical_smi )
        return data