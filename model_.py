import torch.nn.functional as F
from torch import nn
import torch
import copy
from utils_others import average_atoms, average_edges
from utils import get_one_bond_nei
from rdkit import Chem


class PosEmb_Seq2eq(nn.Module):
    def __init__(self,
                 pf_dim_pos,  # I take 256
                 pf_dim1,  # I take 256
                 pf_dim2,
                 max_Emb_wl,
                 max_Emb_d,
                 
                 outp_dim,  # I take 16  
                 trg_emb_dim,  # I take 32
                 ENC_LAYERS,
                 ENC_HEADS,
                 ENC_PF_DIM,
                 DEC_LAYERS,
                 DEC_HEADS,
                 DEC_PF_DIM,
                 device,
                 
                 t_e_ENC_LAYERS,
                 t_e_ENC_HEADS,      # node_feat_dim+
                 t_e_ENC_PF_DIM,     # 4 * (node_feat_dim + outp_dim) + 3 * edge_feat_dim  
                 
                 noise_dim = 50,       
                 noise_loc=0, 
                 noise_scale=1,
                 edge_feat_dim=10,
                 node_feat_dim=39,
                 dropout=0.1,
                 max_num_atom=9

                 ):
        super(PosEmb_Seq2eq, self).__init__()

        # 原子层attention
        # self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
        #                           edge_feat_size=edge_feat_size,
        #                           num_layers=num_layers,
        #                           graph_feat_size=graph_feat_size,
        #                           dropout=dropout)
        self.device = device
        self.noise_dim = noise_dim
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale
        self.max_num_atom = max_num_atom
        self.node_feat_dim = node_feat_dim
        self.outp_dim = outp_dim
        self.DIM = 4 * (node_feat_dim + outp_dim) + 3 * edge_feat_dim + noise_dim

        self.make_pos = Make_pos(max_num_atom * max_num_atom * edge_feat_dim, pf_dim_pos, outp_dim * max_num_atom,
                                 dropout)
        self.make_pos_wl = Make_pos_wl(pf_dim1, pf_dim2, outp_dim, dropout, max_Emb_wl)
        self.make_pos_d  = Make_pos_d(pf_dim1, pf_dim2, outp_dim, dropout, max_Emb_d)
        
        self.t_e_encoder = Encoder(
            node_feat_dim+outp_dim,      # 
            t_e_ENC_LAYERS,
            t_e_ENC_HEADS,      # 
            t_e_ENC_PF_DIM,
            dropout,
            device,
            output_dim=node_feat_dim+outp_dim)   
        
        self.encoder = Encoder(
            self.DIM,      
            ENC_LAYERS,
            ENC_HEADS,      
            ENC_PF_DIM,
            dropout,
            device,
            output_dim=4 * outp_dim + trg_emb_dim)   
        self.decoder = Decoder(
            output_dim=4,          
            hid_dim=trg_emb_dim,     
            outp_dim=outp_dim,
            n_layers=DEC_LAYERS,
            n_heads=DEC_HEADS,      
            pf_dim=DEC_PF_DIM,
            dropout=dropout,
            device=device

        )

        self.model = Seq2Seq(self.encoder,
                             self.decoder,
                             device).to(device)

        self.layerNorm = torch.nn.LayerNorm(4)
        
    def get_s_e_feat(self, bond, smiles, atom_feat_pos_i):
        atom1_ns, atom2_ns = get_one_bond_nei(bond.cpu().numpy().tolist(), Chem.MolFromSmiles(smiles))    
        atom_s_feat = average_atoms(atom1_ns, atom_feat_pos_i, self.device)
        atom_e_feat = average_atoms(atom2_ns, atom_feat_pos_i, self.device)

        return atom_s_feat, atom_s_feat        
        
    def get_s_e_edge(self, bond, smiles, AM_bond_feat_i):
        atom1_ns, atom2_ns = get_one_bond_nei(bond.cpu().numpy().tolist(), Chem.MolFromSmiles(smiles))    

        bonds1 = [(a, int(bond[0])) for a in atom1_ns]
        bonds2 = [(int(bond[1]), a) for a in atom2_ns]

        bond_s_feat = average_edges(bonds1, AM_bond_feat_i)
        bond_e_feat = average_edges(bonds2, AM_bond_feat_i)

        return bond_s_feat, bond_e_feat
    
    def add_noise(self, tensor, noise_dim, loc=0, scale=1):
        rand_dist = torch.distributions.normal.Normal(loc=loc, scale=scale)
        shape = list(tensor.size())
        shape[-1] = int(noise_dim)
        rand_x = rand_dist.sample(shape).to(self.device)
        tensor_n = torch.cat([tensor, rand_x], dim=-1)
        return tensor_n        
        
        
    def repeat_tensor_by_numlist(self, tensor, numlist):
        """
        
        """
        shape = tensor.shape

        tensor_list = []
        for i, num_confs in enumerate(numlist):
            wqe = torch.ones([len(shape)])
            wqe[0] = num_confs
            wqe = tuple(wqe.int().cpu().numpy().tolist())
            tensor_i = tensor[i].unsqueeze(0).repeat(wqe)
            tensor_list.append(tensor_i)

        tensor_n = torch.cat(tensor_list, dim=0)

        return tensor_n    
    
    
    def repeat_list_by_numlist(self, list_, numlist):
        list_new = []
        [list_new.extend([list_[i]]*numlist[i]) for i in range(len(list_))]
        return list_new
   
    
    
    
    def make_src_mask_(self, num_atom_list, max_num_atom):
        # num_atom_list: (list)  (len:batch size)
        src_mask = torch.full([len(num_atom_list), max_num_atom], False,device=self.device).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, max_num_atom]
        for i, group in enumerate(num_atom_list):
            src_mask[i][0][0][:group] = True

        # src_mask = [batch size, 1, 1, max_num_atom]

        return src_mask
    def forward(self, device, atom_feat, AM_bond_feat, node_color_feat, d, labels, masks, relativeenergys, num_atom_list, angle_group_num, num_confs_list, smiles_list):
        
        """
        param: 

             (atom_feat:              torch.Size([16, 90, 39]),
             AM_bond_feat:              torch.Size([16, 90, 90, 10]),
             node_color_feat:              torch.Size([16, 90]),
             d:              torch.Size([16, 90]))

             labels:          torch.Size([929, 10, 2])
             relativeenergys:   torch.Size([929])

             masks: torch.Size([num_tors, 2 ])*16

             num_atom_list, angle_group_num, num_confs_list: all have batch_size len 
                  eg:  
                     (18, 28, 27, 20, 19, 51, 34, 22, 40, 23, 21, 26, 35, 23, 17, 27),
                     [3, 8, 4, 4, 4, 7, 8, 4, 5, 9, 6, 6, 10, 4, 3, 6],
                     [36, 138, 102, 3, 9, 138, 10, 8, 121, 66, 39, 54, 96, 34, 39, 36]
                     
         return: 
             prediction:  torch.Size([batch_size, num_confs, max(angle_group_num), 4])
         """        

        # device
        # angle_group_num:   (list)  len = batch_size  eg: [5, 9, 4, 8, 2, 8, 3, 9]
        # atom_feat          torch.Size([batch_size,max_num_atom, 39])
        # AM_bond_feat        torch.Size([batch_size, 9, 9, 10])
        # masks              (list)  len = batch_size  
        # labels             torch.Size([batchxconfs_size, max, 2])  

        
        AM_bond_feat_f = AM_bond_feat.reshape(AM_bond_feat.shape[0],-1).float()  # torch.Size([batch_size, 9*9*10])  1/0
        pos = self.make_pos(AM_bond_feat_f).reshape(AM_bond_feat_f.shape[0], self.max_num_atom,-1)
        # torch.Size([batch_size，max_num_atom，outp_dim]
        pos_wl= self.make_pos_wl(node_color_feat)
        # torch.Size([batch_size，max_num_atom，outp_dim]
        pos_d = self.make_pos_d(d)
        # torch.Size([batch_size，max_num_atom，outp_dim]
        pos = pos+pos_wl+pos_d
        # torch.Size([batch_size，max_num_atom，outp_dim]
        atom_feat_pos = torch.cat((atom_feat,pos),-1)
        # torch.Size([batch_size，max_num_atom，node_feat_dim + outp_dim]
        src_mask=self.make_src_mask_(num_atom_list, self.max_num_atom)
        atom_feat_pos_= self.t_e_encoder(atom_feat_pos, src_mask)
        
        
        batch_mol_tor_feat_list = []
        for i in range(atom_feat.shape[0]):
            smiles = smiles_list[i]
            lista = []
            mask_ = torch.tensor(masks[i]).to(device)
            atom_feat_pos_i = atom_feat_pos_[i]
            AM_bond_feat_i = AM_bond_feat[i]
            for k in mask_:       # k: eg： tensor([1, 2], device='cuda:0')
                # node cat
                # -----------------------------------------------------------------------------
                atom_feat_pos_se = torch.stack(self.get_s_e_feat(k, smiles, atom_feat_pos_i),0)
                # torch.Size([2, 55])
                tor_atom_pos = torch.cat([atom_feat_pos_i[k], atom_feat_pos_se],0
                                        ).reshape(4 * (self.node_feat_dim + self.outp_dim))
                # torch.size([220])
                # =============================================================================
                
                # edge cat
                # ------------------------------------------------------------------------------
                edge_se = self.get_s_e_edge(k, smiles, AM_bond_feat_i)
                # torch.Size([2, 10])                
                
                tor_edge = torch.cat(
                    [AM_bond_feat_i[k[0], k[1]], *edge_se])
                # torch.Size([3*10])
                # =============================================================================
                
                each_tor = torch.cat([tor_atom_pos, tor_edge])
                # torch.Size([220+3*10])
                lista.append(each_tor)

            each_mol_tor_feat = torch.stack(lista)  # torch.Size([angle_group_num[i], 250]
            each_mol_tor_feat = F.pad(each_mol_tor_feat, pad=(0, 0, 0, max(angle_group_num) - angle_group_num[i]),
                                      mode='constant', value=0)  # torch.Size([max_angle_group_num, 220])
            batch_mol_tor_feat_list.append(each_mol_tor_feat)

        batch_mol_tor_feat = torch.stack(
            batch_mol_tor_feat_list)  # torch.Size([8, 28, 250])  torch.Size([batch_sice, max, 250])

        batch_mol_tor_list = []
        for i in range(atom_feat.shape[0]):
            smiles = smiles_list[i]
            listb = []
            mask_ = torch.tensor(masks[i]).to(device)
            pos_i = pos[i]
            for k in mask_:
                # only pos cat
                
                
                pos_se = torch.stack(self.get_s_e_feat(k, smiles, pos_i),0)
                # torch.Size([2, 55])
                tor_pos = torch.cat([pos_i[k], pos_se],0
                                        ).reshape(-1)

                listb.append(tor_pos)

            each_mol_tor_feat = torch.stack(listb)  # torch.Size([angle_group_num[i], 16*4]
            each_mol_tor_feat = F.pad(each_mol_tor_feat, pad=(0, 0, 0, max(angle_group_num) - angle_group_num[i]),
                                      mode='constant', value=0)  # torch.Size([max_angle_group_num, 16*4])
            batch_mol_tor_list.append(each_mol_tor_feat)

        batch_mol_tor = torch.stack(
            batch_mol_tor_list)  # torch.Size([8, 28, 16*4])  torch.Size([batch_sice, max, 16*4])
        
        # repeat by num_confs_list
        batch_mol_tor_feat = self.repeat_tensor_by_numlist(batch_mol_tor_feat, num_confs_list)
        batch_mol_tor = self.repeat_tensor_by_numlist(batch_mol_tor, num_confs_list)
        angle_group_num = self.repeat_list_by_numlist( angle_group_num, num_confs_list)
        
        # add noise
        batch_mol_tor_feat_noise = self.add_noise(batch_mol_tor_feat, self.noise_dim, self.noise_loc, self.noise_scale)
        
        # deal with labels
        labels_dcopy = copy.deepcopy(labels)
        labels_dcopy[:,:,1] = labels_dcopy[:,:,1]/2
        labels_dcopy = torch.cat((torch.stack([relativeenergys, relativeenergys]).t().unsqueeze(1),labels),1)
        # [batchxconfs_size, max+1, 2]
        labels_dcopy = torch.cat((labels_dcopy,(relativeenergys/5-0.5).repeat(labels_dcopy.shape[1],1).t().unsqueeze(-1)),-1)
        #torch.Size([batchxconfs_size, max+1, 3])
        
        
        # seq2seq
        prediction, _ = self.model(batch_mol_tor_feat_noise, labels_dcopy, angle_group_num, batch_mol_tor)
        prediction = self.layerNorm(prediction)
        # torch.Size([8, 29])  torch.Size([batch_sice, max+1])
        prediction = prediction[:,:-1,:]
        # torch.Size([8, 28])  torch.Size([batch_sice, max])
        #         prediction = torch.cat([prediction_[index][0 : group] for index, group in enumerate(angle_group_num)])
        #         # torch.Size([129])
        return prediction, _


class Make_pos(nn.Module):
    def __init__(self, hid_dim, pf_dim, output_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x) * 4

        # x = [batch size, seq len, hid dim]

        return x
    
class Make_pos_wl(nn.Module):
    def __init__(self, pf_dim1, pf_dim2, output_dim, dropout, max_Emb):
        super().__init__()

        self.embedding = nn.Embedding(max_Emb, pf_dim1)
        
        self.fc_1 = nn.Linear(pf_dim1, pf_dim2)
        self.fc_2 = nn.Linear(pf_dim2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, max_num_atom]

        x=self.embedding(x)
        
        # x = [batch size, seq len, pf dim1]
        
        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim2]

        x = self.fc_2(x) * 4

        # x = [batch size, seq len, output_dim]

        return x

class Make_pos_d(nn.Module):
    def __init__(self, pf_dim1, pf_dim2, output_dim, dropout, max_Emb):
        super().__init__()

        self.embedding = nn.Embedding(max_Emb, pf_dim1)
        
        self.fc_1 = nn.Linear(pf_dim1, pf_dim2)
        self.fc_2 = nn.Linear(pf_dim2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, max_num_atom]

        x=self.embedding(x)
        
        # x = [batch size, seq len, pf dim1]
        
        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim2]

        x = self.fc_2(x) * 4

        # x = [batch size, seq len, output_dim]

        return x
    


"""
transormer model=======================================================================
"""
class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def make_src_mask(self, angle_group_num):
        # angle_group_num: (list)  (len:batch siza)
        src_mask = torch.full([len(angle_group_num), max(angle_group_num)], False,device=self.device).unsqueeze(1).unsqueeze(2)

        src_mask_ = torch.full([len(angle_group_num), max(angle_group_num)+1], False,device=self.device).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, max]
        for i, group in enumerate(angle_group_num):
            src_mask[i][0][0][:group] = True
            src_mask_[i][0][0][:group+1] = True

        # src_mask = [batch size, 1, 1, max]

        trg_sub_mask = torch.tril(torch.ones((max(angle_group_num)+1, max(angle_group_num)+1), device=self.device)).bool()
        # trg_sub_mask = [trg len(max), trg len(max)]

        trg_mask = src_mask_ & trg_sub_mask
        # trg_mask = [batch size, 1, max+1, max+1]

        return src_mask, trg_mask

    def forward(self, src, trg, angle_group_num, batch_mol_tor):
        # src = [batch size, max, dim]
        # trg = [batch size, max+1]
        # angle_group_num: type=list len=batch size

        src_mask, trg_mask = self.make_src_mask(angle_group_num)

        # src_mask = [batch size, 1, 1, max]
        # trg_mask = [batch size, 1, ,max, max]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, max, output_dim(96)]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask, batch_mol_tor)

        # output=[8,29]

        return output, attention


class Encoder(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 output_dim):
        super().__init__()

        self.device = device

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src, src_mask):
        # src = [batch size, src len, dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, dim]

        src = self.fc_out(src)

        # src = [batch size, src len, 96]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))  # 3.2554

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,    
                 hid_dim,
                 outp_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device
                 #                  max_length = 100
                 ):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Linear(3, hid_dim)
        #         self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim + 4 * outp_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim + 4 * outp_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.tensor(3).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask, batch_mol_tor):
        # trg =           [batch size, max+1, 2]
        # batch_mol_tor = [batch size, max, 16*4]
        # enc_src = [batch size, max, 96]
        # trg_mask = [batch size, 1, max+1, max+1]
        # src_mask = [batch size, 1, 1, max+1]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        #         pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #         #pos = [batch size, trg len]

#         trg = trg.unsqueeze(2)

        batch_mol_tor_ = F.pad(batch_mol_tor, pad=(0, 0, 1, 0, 0, 0), mode='constant', value=0)  #  [batch size, 29, 16*4]  
        trg = torch.cat((self.tok_embedding(trg) * self.scale/2, batch_mol_tor_/2),
                        -1)  # [batch size, max+1, （16*4 + 32）]  
        # trg = [batch size, max+1, 16*4+32]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, 16*4+hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, max+1, output_dim]    

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]        8,29,96

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]     8,29,96

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention
