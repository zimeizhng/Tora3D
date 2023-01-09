from torch_geometric.data import Data
import networkx as nx
import torch_geometric as tg
import torch

from rdkit import Chem

import py3Dmol


from ipywidgets import interact, fixed, IntSlider
import ipywidgets
import py3Dmol

def extract_cycles(mol):
    """
    提取一个分子中的所有环
    """
    row, col = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]    
    edge_index = torch.tensor([row, col], dtype=torch.long)
    
    data = Data(edge_index = edge_index)
    G = nx.to_undirected(tg.utils.to_networkx(data))
    cycles = nx.cycle_basis(G)
    return cycles


def get_edge_list_from_cycle(cycle):
    """
    [2, 3, 4, 9, 10, 1]  ———》   [[2, 3], [3, 4], [4, 9], [9, 10], [10, 1], [1, 2]]
    """
    edge_list = []
    for i,n in enumerate(cycle):
        if i < len(cycle)-1:
            edge = [n, cycle[i+1]]
        else:
            edge = [n, cycle[0]]
        edge_list.append(edge)
    return edge_list


def drawit(m, p=None, confId=-1):
    """ pymol可视化
    param:
        m: mol
    """
    if p == None:
        p = py3Dmol.view(width=400,height=200)
        
    mb = Chem.MolToMolBlock(m, confId=confId)
    p.removeAllModels()
    p.addModel(mb, "sdf")
    p.setStyle({"stick":{}})
    p.setBackgroundColor("0xeeeeee")
    p.zoomTo()
    return p.show()
# drawit(mol, p)

# 另一个可视化三D小分子的方法：来自于geomol
# -----------------------------------------------------------------------------
from rdkit import Chem
import pickle

from ipywidgets import interact, fixed, IntSlider
import ipywidgets
import py3Dmol

def show_mol(mol, view, grid):
    mb = Chem.MolToMolBlock(mol)
    view.removeAllModels(viewer=grid)
    view.addModel(mb,'sdf', viewer=grid)
    view.setStyle({'model':0},{'stick': {}}, viewer=grid)
    view.zoomTo(viewer=grid)
    return view

def view_single(mol):
    view = py3Dmol.view(width=600, height=600, linked=False, viewergrid=(1,1))
    show_mol(mol, view, grid=(0, 0))
    return view

def MolTo3DView(mol, size=(600, 600), style="stick", surface=False, opacity=0.5, confId=0, \
                stick_colorscheme="cyanCarbon"):
    """Draw molecule in 3D
    
    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick':{'colorscheme':'cyanCarbon'}
                                   {'colorscheme':'skyblue'}
                                   但是好像只有"cyanCarbon"有效，其他颜色都无效
                              'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """

    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol[confId])
    
    mol_ = Chem.MolFromMolBlock(mblock)
    smiles = Chem.MolToSmiles(mol_)
    print("smiles:", smiles)
    mol_smiles = Chem.MolFromSmiles(smiles)
    display(mol_smiles)
    
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    viewer.setStyle({'model':0},{'stick':{'colorscheme': stick_colorscheme}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer

def conf_viewer(idx, mol, stick_colorscheme="cyanCarbon"):
    return MolTo3DView(mol, confId=idx, stick_colorscheme=stick_colorscheme).show()

def visualize_confs(mols, conf_viewer = conf_viewer, stick_colorscheme="cyanCarbon"):
    interact(conf_viewer, idx=ipywidgets.IntSlider(min=0, max=len(mols)-1, step=1), mol=fixed(mols), stick_colorscheme=stick_colorscheme);

# ================================================================
