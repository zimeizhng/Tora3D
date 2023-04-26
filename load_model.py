from model_ import PosEmb_Seq2eq
from utils import count_parameters

import torch


def load_model(max_num_atom,
            max_Emb_wl,
            max_Emb_d,
            wl_max_iter,
            device, 
            cheekpoint_path=None ):
    model = PosEmb_Seq2eq(          # 4 * (node_feat_dim + outp_dim*3) + 3 * edge_feat_dim 

        pf_dim_pos=256,
        pf_dim1=256,  # I take 256
        pf_dim2=256,  # I take 256
        max_Emb_wl=max_Emb_wl,   
        max_Emb_d=max_Emb_d,

        # pf_dim_readout=256,  # I take 256
        outp_dim=16,  # I take 16
        trg_emb_dim=512,

        ENC_LAYERS=1,
        ENC_HEADS=5,  # 
        ENC_PF_DIM=128,
        DEC_LAYERS=3,
        DEC_HEADS=4,
        DEC_PF_DIM=64,
        device=device,

        t_e_ENC_LAYERS=1,
        t_e_ENC_HEADS=5,      
        t_e_ENC_PF_DIM=64,


        edge_feat_dim=10,
        node_feat_dim=39,
        dropout=0.1,
        max_num_atom=max_num_atom
    ).to(device)
    print("model的参数个数为:", count_parameters(model))
    
    Train_loss = []
    Val_loss = []
    Train_loss_value = []
    Train_loss_sig = []
    Val_loss_value = []
    Val_loss_sig = []

    
    if cheekpoint_path is not None:
        
        cheekpoint = torch.load(cheekpoint_path)
        model.load_state_dict(cheekpoint['model'])
        
        Train_loss = cheekpoint["Train_loss"]
        Val_loss = cheekpoint["Val_loss"]
        Train_loss_value =cheekpoint["Train_loss_value"]
        Train_loss_sig =cheekpoint["Train_loss_sig"]
        Val_loss_value =cheekpoint["Val_loss_value"]
        Val_loss_sig =cheekpoint["Val_loss_sig"]

    return model, Train_loss,Train_loss_value,Train_loss_sig, Val_loss ,Val_loss_value,Val_loss_sig

