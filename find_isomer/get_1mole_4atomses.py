from rdkit.Chem import Draw
from torch_geometric.data import Data
import networkx as nx
import torch_geometric as tg

import sys 
# sys.path.append("/home/admin/work/大四下学期药物所服务器文件/大四下学期药物所服务器文件/v100/predict_Dihedral_angle/trans_t_e(pos_wl_d) from_geom/noise_all")

from utils_others import *
from utils_mole import *


def get_edge_type(mol, edge_list):
    return [mol.GetBondBetweenAtoms(int(edge[0]), int(edge[1])).GetBondType() for edge in edge_list]

def get_pre_aft_idx_fromring(index, edge_list):
    pre_index = index-1
    aft_index = index+1
    if index == len(edge_list)-1:
        aft_index = 0
    return pre_index,aft_index

def get_4atom(edge_list, index):
    pre_node = edge_list[get_pre_aft_idx_fromring(index, edge_list)[0]][0]
    aft_node = edge_list[get_pre_aft_idx_fromring(index, edge_list)[-1]][-1]
    return [pre_node, *edge_list[index], aft_node]



def get_1mole_4atomses(mol):
    """
    得到一个分子的每个环中的一个单键的四原子（为下一步立场优化做准备）
    """
    cycles = np.array([cycle for cycle in extract_cycles(mol) if len(cycle)>=5])
    cycle_edge_list = np.array([get_edge_list_from_cycle(cycle) for cycle in cycles])
    cycle_edge_list_type = np.array([np.array(get_edge_type(mol, edge_list)) for edge_list in cycle_edge_list])
    # 判断是不是单键
    cycle_edge_list_type_ios = [(np.array(edge_list_type) != Chem.rdchem.BondType.SINGLE).sum()<=2
                                for edge_list_type in cycle_edge_list_type]
    
    
    # 判断哪些环是有小于两个非单键的
    choosed_cycles = cycles[cycle_edge_list_type_ios]
    choosed_cycle_edge_list = cycle_edge_list[cycle_edge_list_type_ios]
    choosed_cycle_edge_list_type = cycle_edge_list_type[cycle_edge_list_type_ios]
    choosed_cycle_list_index = [np.where(type_list == 1)[0][0] for type_list in choosed_cycle_edge_list_type]
    
    choosed_4atom_list = [get_4atom(choosed_cycle_edge_list[j], choosed_cycle_list_index[j])
                                           for j in range(len(choosed_cycle_edge_list))]
    
      
#     visualization_mole(mol)
#     print(cycles)
#     print(choosed_4atom_list)
    return choosed_4atom_list

if __name__ == '__main__':
    for i in range(len(smilelist)):
        smiles = smilelist[i]
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

        choosed_4atom_list = get_1mole_4atomses(mol)

