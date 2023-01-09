from torch.utils.data import DataLoader
from functools import partial
from collate_f import collate
import random
import pandas as pd
from torch_geometric.data import Data
import pandas as pd
from functools import partial


def pre_test_data(SMILES, Torsion_list, Torsion, relativeenergy_list, batchsize, 
                     max_num_atom = 9, wl_max_iter = 3, num_choosed_confs = 300, device="cuda"):
    testdata = list(zip(SMILES,Torsion,Torsion_list,relativeenergy_list))
    
    test_loader = DataLoader(testdata, batch_size=batchsize, 
                         collate_fn=partial(collate, max_num_atom=max_num_atom, 
                                            wl_max_iter=wl_max_iter,
                                            num_choosed_confs = num_choosed_confs,
                                           device=device,
                                           ),
                        shuffle=False,pin_memory=False)
    print("len(test_loader)", len(test_loader))
    return test_loader


def pre_data(SMILES, Torsion_list, Torsion, relativeenergy_list, batchsize, rate_trte=0.9, rate_tr_in_trti=0.8, max_num_atom = 9, wl_max_iter = 3, num_choosed_confs = 300, inference=False, device = "cuda"):
    
    if inference==True:
        data=list(zip(SMILES, Torsion, Torsion_list, relativeenergy_list))
        data_loader = DataLoader(data, batch_size=batchsize, 
                              collate_fn=partial(collate, max_num_atom=max_num_atom, wl_max_iter=wl_max_iter,
                                                ),
                             shuffle=False,pin_memory=False)
        return data_loader
    
    # splite dataset
    num = len(SMILES)
    list1 = range(num)
    random.seed(23)
    Train_index = random.sample(list1, int(num * rate_trte))

    test_index = list(set(list1).difference(set(Train_index)))
    train_index = random.sample(Train_index, int(num * rate_tr_in_trti))
    val_index = list(set(Train_index).difference(set(train_index)))

    train_smiles, train_angle, train_mask, train_relativeenergy = data_split(SMILES, Torsion_list, Torsion, relativeenergy_list, train_index)
    val_smiles, val_angle, val_mask, val_relativeenergy = data_split(SMILES, Torsion_list, Torsion, relativeenergy_list, val_index)
    test_smiles, test_angle, test_mask, test_relativeenergy = data_split(SMILES, Torsion_list, Torsion, relativeenergy_list, test_index)
    
    traindata = [(train_smiles[i], train_angle[i], train_mask[i], train_relativeenergy[i]) for i in range(len(train_index))]
    valdata = [(val_smiles[i], val_angle[i], val_mask[i], val_relativeenergy[i]) for i in range(len(val_index))]
    testdata = [(test_smiles[i], test_angle[i], test_mask[i], test_relativeenergy[i]) for i in range(len(test_index))]
    
    # dataloader
    train_loader = DataLoader(traindata, batch_size=batchsize, 
                              collate_fn=partial(collate, max_num_atom=max_num_atom, 
                                                 wl_max_iter=wl_max_iter,
                                                 num_choosed_confs = num_choosed_confs,
                                                device=device),
                             shuffle=True,pin_memory=False)
    val_loader = DataLoader(valdata, batch_size=batchsize, 
                            collate_fn=partial(collate, max_num_atom=max_num_atom, 
                                               wl_max_iter=wl_max_iter,
                                               num_choosed_confs = num_choosed_confs,
                                              device=device),
                           shuffle=True,pin_memory=False)
    test_loader = DataLoader(testdata, batch_size=batchsize, 
                             collate_fn=partial(collate, max_num_atom=max_num_atom, 
                                                wl_max_iter=wl_max_iter,
                                                num_choosed_confs = num_choosed_confs,
                                               device=device),
                            shuffle=False,pin_memory=False)
    return train_loader, val_loader, test_loader

def data_split(SMILES, Torsion_list, Torsion, relativeenergy_list, index):
    Smiles = [SMILES[i] for i in index]
    mask = [Torsion_list[i] for i in index]
    angle = [Torsion[i] for i in index]
    relativeenergy = [relativeenergy_list[i] for i in index]
    return Smiles, angle, mask, relativeenergy



def pre_data_samesmilesinbatch(smilelist,torsion_angle,torsion_list,relativeenergy_list,len_=10):

    df = pd.DataFrame({"smilelist": smilelist,"torsion_angle": torsion_angle, "torsion_list": torsion_list, "relativeenergy_list": relativeenergy_list})
    df=df.groupby(by='smilelist')

    df_cat = pd.DataFrame()
    df_cat["dihedral_degree"] = df["torsion_angle"].apply(list)
    df_cat["dihedral_pairs_atoms4"] = df["torsion_list"].apply(list)
    df_cat["relativeenergy_list"] = df["relativeenergy_list"].apply(list)

    df_cat_=df_cat[df_cat.applymap(l)["dihedral_degree"]>=len_]

    df_cat_ = df_cat_.apply(partial(eliminate,len_=len_))

    smilelist_ = []
    torsion_angle_ = []
    torsion_list_ = []
    relativeenergy_list_ = []

    for i,index in enumerate(df_cat_.index):
        for i in range(len_):
            smilelist_.append(index)
            torsion_angle_.append(df_cat_["dihedral_degree"][index][i])
            torsion_list_.append(df_cat_["dihedral_pairs_atoms4"][index][i])        
            relativeenergy_list_.append(df_cat_["relativeenergy_list"][index][i])       

    traindata = [(smilelist_[i], torsion_angle_[i], torsion_list_[i], relativeenergy_list_[i]) for i in range(len(smilelist_))]


    # dataloader
    train_loader = DataLoader(traindata, batch_size=len_, 
                              collate_fn=partial(collate, max_num_atom=max_num_atom, wl_max_iter=wl_max_iter,
                                                data_enhancement = data_enhancement),
                             shuffle=False,pin_memory=False)

    print(len(smilelist_), len(torsion_angle_), len(torsion_list_), len(relativeenergy_list_))
    print(len(train_loader))
    return  train_loader

def eliminate(list_, len_):
    return list_[:len_]
def l(list_):
    return len(list_)

def create_data(row):
    data = Data()
    data.name = row.name
    data.dihedral_degree = row["dihedral_degree"]
    data.dihedral_pairs_atoms4= row["dihedral_pairs_atoms4"]
    data.relativeenergy_list = row["relativeenergy_list"]
    return data
