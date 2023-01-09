import torch
from utils_base import repeat_list_by_numlist
def batch_inference(model, batch_data, loss_criterion, loss_criterion_, optimizer=None, debug = False, max_re = 5): 
    device = model.device
    atom_feat, AM_bond_feat, node_color_feat, d, labels, masks, relativeenergys, num_atom_list, angle_group_num, num_confs_list, smiles_list = batch_data
    rand_dist = torch.distributions.uniform.Uniform(0,max_re)
    relativeenergys = rand_dist.sample(relativeenergys.shape).to(device)
    max_ = max(angle_group_num)
    batch_size = len(relativeenergys)
    prediction = torch.zeros(batch_size,max_,2, device=device)
    for i in range(max_):
        prediction,_ = model(device, atom_feat, AM_bond_feat, node_color_feat, d, prediction,masks,relativeenergys,num_atom_list,angle_group_num,num_confs_list, smiles_list)
        prediction[:,:,0][prediction[:,:,0] > 0.5] = 0.5
        prediction[:,:,0][prediction[:,:,0] < -0.5] = -0.5    
        prediction = torch.cat((prediction[:,:,0].unsqueeze(-1), prediction[:,:,1:].argmax(-1).unsqueeze(-1)),-1) 
        prediction[:,:,1] = prediction[:,:,1]-1  
    if debug == True:
        pred_lab = torch.stack([prediction, labels],-1)
        pred_lab_ = torch.cat([pred_lab[index][0: group] for index,group in enumerate(repeat_list_by_numlist(angle_group_num,num_confs_list))])
        torch.set_printoptions(profile="full")
        torch.set_printoptions(sci_mode=False) 
        print("value对比：",pred_lab_[:,0])
        print("sig对比：",pred_lab_[:,1])            
    prediction_list = [prediction[index][0: group] for index,group in enumerate(repeat_list_by_numlist(angle_group_num,num_confs_list))]    
    prediction_ = torch.cat(prediction_list)
    labels_list = [labels[index][0: group] for index, group in enumerate(repeat_list_by_numlist(angle_group_num,num_confs_list))]
    labels_ = torch.cat(labels_list)        
    loss_value = loss_criterion(prediction_[:,0], labels_[:,0]).item()
    acc_sig = ((prediction_[:,1] == labels_[:,1]).sum()/len(labels_)).item()    
    return loss_value, acc_sig, prediction_list, labels_list,\
                 masks, relativeenergys, num_atom_list, angle_group_num, num_confs_list, smiles_list,_
def inference(epoch, model, data_loader, loss_criterion, loss_criterion_, optimizer=None, debug = False):
    model.eval()
    loss_values_list = []
    acc_sig_list = []  
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            loss_value, acc_sig, prediction_list, labels_list,\
                masks, relativeenergys, num_atom_list, angle_group_num, num_confs_list, smiles,_ = \
                batch_inference(model, batch_data, loss_criterion, loss_criterion_, optimizer=None, debug = debug)       
            loss_values_list.append(loss_value)       
            acc_sig_list.append(acc_sig)
    loss_value = np.mean(loss_values_list)
    acc_sig = np.mean(acc_sig_list)    
    return loss_value,acc_sig