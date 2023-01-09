import pickle
import time
import torch.nn.functional as F
import numpy as np
import ot
import random
from math import cos, sin, pi



def angle_vector(angle):
    """ 返回一个角（角度值）的向量表示 """
    return [cos(angle*pi/ 180), sin(angle*pi/ 180)]
def getArotateM(angle):  # 得到一个旋转矩阵(二维的)
    """ 返回旋转某个特定角度（角度值）的旋转矩阵 """
    angle_pi = angle*pi/ 180
    return [[cos(angle_pi), -sin(angle_pi)],
             [sin(angle_pi),  cos(angle_pi)]]


def load_smiles(filename):
    with open(filename) as f:
        a = [line.strip('\n') for line in f]
    return a


def batchi_replace(tensor, percent, value, min_max ):
    shape = tensor.shape
    num_dim = len(shape)
    num_points = int(torch.prod(torch.tensor(shape))*percent)
    
    points = torch.zeros(num_points,num_dim).int()
    
    for index, d in enumerate(shape):
        points[:,index] = torch.tensor(np.random.choice(range(d), num_points))
    
    tensor[[p for p in points.T]] = value
    
    return tensor



def compute_distances(P, C):
    """
    计算两组点的距离矩阵
    给tensor返回tensor
    给numpy返回numpy
    
    """
    A = (P**2).sum(axis=1, keepdims=True)   #先对P求平方；然后按行求和，并且保持维度。得到一个5行1列的向量。
 
    B = (C**2).sum(axis=1, keepdims=True).T #得到一个1行4列的向量。
 
    return np.sqrt(A + B - 2* np.dot(P, C.T))  #A+B会广播运算加法。np.dot()是矩阵相乘。
 
# if __name__ == "__main__":
#     P = np.random.randint(1, 5, (5, 3))   #5行3列。5个三维向量。
#     C = np.random.randint(1, 5, (4, 3))   #4行3列。4个三维向量。
#     dist = compute_distances(P, C)
 
#     print(P)
#     print(C)
#     print(dist)



def reorder(a_point,b_point):
    """
    点类型必须为 np.array, 返回也为  np.array
    接受两组点作为输入，改变 b_point 的顺序，使得尽可能与a点距离最近
    return a_point 和 改变顺序后的b_point
    """
    len_a = len(a_point) 
    len_b = len(b_point)
    dis_M = compute_distances(a_point,b_point)
    aw , bw = np.ones(len_a), np.ones(len_b)
    orderM = ot.emd(aw, bw, dis_M)
    b_point_od = np.matmul(orderM,b_point)
    
    return a_point,b_point_od, orderM
    

# 模型保存
def model_save(model,Train_loss_value
                    ,Train_loss_sig
                    ,Train_loss
                    ,Val_loss_value
                    ,Val_loss_sig
                    ,Val_loss):
    time_s = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime())
    torch.save({'epoch': len(Train_loss),
                'model': model.state_dict(),
                'model_optimizer': optimizer.state_dict(),

                "Train_loss_value": Train_loss_value,
                "Train_loss_sig": Train_loss_sig,
                "Train_loss": Train_loss,
                "Val_loss_value": Val_loss_value,
                "Val_loss_sig": Val_loss_sig,
                "Val_loss": Val_loss,
                }, 
               'model_save/model_save_main_drugs_all_{}.pth'.format(time_s))
    print(f'Epoch {len(Train_loss)} - Saved Model')

def repeat_tensor_by_numlist(tensor, numlist):
    """
    写一个函数实现 tensor 的每一行按照指定数目重复，返回每行重复后的tensor
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


def repeat_list_by_numlist(list_, numlist):
    list_new = []
    [list_new.extend([list_[i]]*numlist[i]) for i in range(len(list_))]
    return list_new

def group_by_num(list_, numlist):
    """
    正好是上面那个函数的逆，给一个list_, 按照numlist的值来给list_分组
    """
    group_list = []
    len_ = len(numlist)
    for i in range(len_):
        groupi = list_[:numlist[i]]
        list_ = list_[numlist[i]:]
        
        group_list.append(groupi)
        
    return group_list



def pad_list(x,lengh,value):
    return  x + [value for i in range(lengh - len(x))]


class pickle_:
    def save(file_name, var):
        if file_name[-4:] != ".pkl":
            file_name+=".pkl"
        with open(file_name,"wb") as pf:
            pickle.dump(var, pf)
            print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(f"已经保存到{file_name}中了")
    def load(file_name):
        with open(file_name,"rb") as pf:
            unk = pickle.load(pf)
            return unk
        
        
        
# ---------------------------------------------------------------  
# 将tensor转化为特定形状
def get_intersection_tensor(tensor, toshape):
    intersection_shape = []
    for i in range(len(tensor.shape)):
        if tensor.shape[i] - toshape[i]>=0:
            intersection_shape.append(toshape[i])

        else:
            intersection_shape.append(tensor.shape[i])

    return tensor[: intersection_shape[0], : intersection_shape[1], : intersection_shape[2], : intersection_shape[3]]


def get_pad_tensor(tensor, toshape, value=0):
    pad_list = []
    for i in range(1, len(tensor.shape)+1):
        i = -i
        if  toshape[i] - tensor.shape[i]==0:
            pad_list.extend([0,0])
            
        elif  toshape[i] - tensor.shape[i]>0:
            pad_list.extend([0,toshape[i] - tensor.shape[i]])   
            
        else:
            raise ValueError
            
    pad_tensor = F.pad(tensor, pad_list,"constant", value=value)
    return pad_tensor

def my_reshape_func(tensor, toshape, pad_value=0):
    # 将tensor转化为特定形状
    
    
    intersection_tensor = get_intersection_tensor(tensor, toshape)
    pad_tensor = get_pad_tensor(intersection_tensor, toshape, value=pad_value)
    return pad_tensor

# tensor = torch.rand([3,3,2,6])
# toshape = (4,6,2,3)
# reshape_tensor = my_reshape_func(tensor, toshape, pad_value=0)
# tensor.shape, reshape_tensor.shape

# ========================================================================




def info(tensor):
    return tensor.shape, tensor.min(), tensor.max()



def save_list(list_, file_path):
    f = open(file_path, "w")

    for i in list_:
        f.write(i + "\n")
    f.close()
    
def load_list(file_path):
    with open(file_path, "r") as f:
        list_ = [line.strip('\n') for line in f]
        return list_
    
    
def get_idxs_bylist(list1, list2):
    """
    list1: 短
    list2: 長
    give two lists , return index list of list2 by list1
    """
    idxs = []
    for i in list1:
        try:
            idx = list2.index(i)
            idxs.append(idx)
        except ValueError as e:
            continue    
    return idxs   