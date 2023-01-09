import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from .get_1mole_4atomses import get_1mole_4atomses
from itertools import product
def drawit(m, p, confId=-1):
    mb = Chem.MolToMolBlock(m, confId=confId)
    p.removeAllModels()
    p.addModel(mb, "sdf")
    p.setStyle({"stick":{}})
    p.setBackgroundColor("0xeeeeee")
    p.zoomTo()
    return p.show()
def get_iso_by_FF(mol, atom4, angle_range, p=None):
    if p == None:
        p = py3Dmol.view(width=400,height=200)   
    m = copy.deepcopy(mol)
    Chem.SanitizeMol(m)
    mp = AllChem.MMFFGetMoleculeProperties(m)
    ff = AllChem.MMFFGetMoleculeForceField(m, mp)    
    ff.MMFFAddTorsionConstraint(*atom4, False, *angle_range, 10000.0)
    ff.Minimize() 
    return m
def get_iso_by_FF_(m, choosed_4atom_list, angle_range_list):
    for i in range(len(choosed_4atom_list)):
        m = get_iso_by_FF(m, list(map(int,choosed_4atom_list[i])), angle_range_list[i]) 
    return m        
def get_all_iso(mol, choosed_4atom_list, angle_range_list):
    all_angle_combine = product(*len(choosed_4atom_list)*[angle_range_list])
    m_list = [get_iso_by_FF_(mol, choosed_4atom_list, angle_range_l) for angle_range_l in all_angle_combine]       
    return m_list
def get_ff_ms(mol, angle_range_list):
    choosed_4atom_list = get_1mole_4atomses(mol)
    m_list = get_all_iso(mol, choosed_4atom_list, angle_range_list)
    m_list.append(mol)
    return m_list