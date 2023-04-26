import random
import numpy as np
import torch
import torch.nn.functional as F

from rdkit import Chem
import rdkit.Chem as Chem
from rdkit.Chem import MolFromSmiles

import networkx as nx
import hashlib


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]



def generate_feat(smiles, max_num_atom, wl_max_iter,device="cuda"):

    mol = MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)

    # generate_node_feat

    SYMBOL = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other']
    HYBRIDIZATION = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]

    num_atom = Chem.RemoveHs(mol).GetNumAtoms()
    
    
    symbol = torch.zeros((num_atom, 16), device=device)
    hybridization = torch.zeros((num_atom, 6), device=device)
    degree = torch.zeros((num_atom, 6), device=device)
    num_h = torch.zeros((num_atom, 5), device=device)
    chirality = torch.zeros((num_atom, 3), device=device)
    aromatic = torch.zeros((num_atom, 1), device=device)
    formal_charge = torch.zeros((num_atom, 1), device=device)
    radical_electrons = torch.zeros((num_atom, 1), device=device)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i] = torch.tensor(one_of_k_encoding_unk(atom.GetSymbol(), SYMBOL),device = device)
        hybridization[i] = torch.tensor(one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATION),device = device)
        degree[i] = torch.tensor(one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]),device = device)
        num_h[i] = torch.tensor(one_of_k_encoding_unk(atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4]),device = device)
        try:
            chirality[i] = torch.tensor(one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S', 'unknown']),device = device)
        except:
            chirality[i] = torch.tensor(one_of_k_encoding_unk(atom.GetChiralTag(), \
                                                 ['CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_UNSPECIFIED']),device = device)
        aromatic[i] = atom.GetIsAromatic()
        formal_charge[i] = atom.GetFormalCharge()
        radical_electrons[i] = atom.GetNumRadicalElectrons()  
    
    atom_feat = torch.cat(
        [symbol, hybridization, degree, num_h, chirality, aromatic, formal_charge, radical_electrons], -1)
    # torch.size([num_ayom, 39])
    zeros_ = torch.zeros(max_num_atom, atom_feat.shape[-1], device = device) 
    zeros_[:num_atom,:] = atom_feat
    atom_feat = zeros_
       
    # generate_bond_feat

    BOND_TYPE = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    BOND_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]

    AM_bond_feat = torch.zeros([max_num_atom, max_num_atom, 10], device = device)  

    for bond in mol.GetBonds():
        #     print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        bond_type = one_of_k_encoding(bond.GetBondType(), BOND_TYPE)
        bond_ring = [bond.GetIsConjugated(), bond.IsInRing()]
        bond_stereo = one_of_k_encoding(str(bond.GetStereo()), BOND_STEREO)
        bond_feat = bond_type + bond_ring + bond_stereo

        #     print(bond_feat)

        AM_bond_feat[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = torch.tensor(bond_feat, device = device)
        AM_bond_feat[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = torch.tensor(bond_feat, device = device)

    # generate WL    
    node_list = list(range(num_atom))
    link_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    node_color_dict, node_neighbor_dict = setting_init(node_list, link_list)
    node_color_dict = WL_recursion(node_list, node_color_dict, node_neighbor_dict, max_iter=wl_max_iter)   
    node_color_list = []    
    for node in node_list:
        node_color_list.append(node_color_dict[node])
    node_color_list = node_color_list+[0]*(max_num_atom-len(node_color_list))
    
    # generate degree
    d_list = [mol.GetAtomWithIdx(i).GetDegree() for i in range(num_atom)]
    d_list = d_list+[0]*(max_num_atom-len(d_list))
    
    # atom_feat:  torch.size([max_num_atom, 39])
    # AM_bond_feat:  torch.size([max_num_atom, max_num_atom, 10])
    # d_list:    (list) len=max_num_atom
    # node_color_list:  (list) len=max_num_atom
    return atom_feat, AM_bond_feat, node_color_list, d_list, num_atom


def WL_recursion(node_list, node_color_dict, node_neighbor_dict, max_iter):
    iteration_count = 1
    while True:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if max(node_color_dict.values())==max(new_color_dict.values()) or iteration_count==max_iter:

            return node_color_dict
        else:
            node_color_dict = new_color_dict
        iteration_count += 1


def setting_init(node_list, link_list):
    node_color_dict = {}
    node_neighbor_dict = {}
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in link_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1
    return node_color_dict, node_neighbor_dict
