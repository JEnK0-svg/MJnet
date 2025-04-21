import numpy as np
from sklearn.metrics import confusion_matrix
import torch


def read_data(filepath):
    '''read training-validation file: site-level samples'''
    with open(filepath) as f:
        data = f.readlines()
        train_positive = []
        train_negative = []
        val_positive = []
        val_negative = []
        for i in data:
            tmp = i.strip('\n').split('\t')
            if tmp[5] == 'train':
                if tmp[4] == '1':
                    train_positive.append((tmp[1], tmp[3], 1))  # 标签1
                elif tmp[4] == '0':
                    train_negative.append((tmp[1], tmp[3], 0))  # 标签0

            elif tmp[5] == 'val':
                if tmp[4] == '1':
                    val_positive.append((tmp[1], tmp[3], 1))  # 标签1
                elif tmp[4] == '0':
                    val_negative.append((tmp[1], tmp[3], 0))  # 标签0

        print('train positive', len(train_positive))
        print('train negative', len(train_negative))
        print('val positive', len(val_positive))
        print('val negative', len(val_negative))

        train = train_positive + train_negative
        val = val_positive + val_negative

        lens = [len(mi) for mi, m, label in train]
        print('min mirna len', np.min(lens))

        return train, val




def read_test(filepath):
    '''read test file: are gene-level samples'''
    all = []
    with open(filepath) as f:
        data = f.readlines()
        for i in data[1:]:
            tmp = i.strip('\n').split('\t')
            if tmp[4] == '1':
                all.append((tmp[1], tmp[3], 1))
            elif tmp[4] == '0':
                all.append((tmp[1], tmp[3], 0))
    return all



def reverse_seq(seq):
    '''reverse the 5-3 direction to 3-5 direction of mRNAs'''
    rseq = ''
    for i in range(len(seq)):
        rseq += seq[len(seq)-1-i]
    return rseq




def get_embedding(rna):
    '''prepared for the embedding'''
    c = {'A':0,'C':1,'G':2,'U':3,'X':4}
    map = []
    for i in range(len(rna)):
        tmp = c[rna[i]]
        map.append(tmp)
    return map


def to_Onehot(rna):
    '''使用字典生成RNA序列的one-hot编码'''
    onehot_dict = {
        'A': [1,0,0,0],
        'C': [0,1,0,0], 
        'G': [0,0,1,0],
        'U': [0,0,0,1],
        'X': [0,0,0,0]  # Padding
    }
    return np.array([onehot_dict[base] for base in rna])
    
    
    # 根据词典将每个碱基映射为对应的One-Hot编码
    onehot_map = [onehot_dict[base] for base in rna]

    return np.array(onehot_map)

    # 根据词典将每个碱基映射为对应的One-Hot编码
    onehot_map = [onehot_dict[base] for base in rna]

    return np.array(onehot_map)

def to_C2(rna):
    '''生成C2编码'''
    C2_dict = {
        'A': [0,0],
        'C': [1,1],
        'G': [1,0], 
        'U': [0,1],
        'X': [-1,-1]
    }
    return np.array([C2_dict[base] for base in rna])


def to_NCP(rna):
    '''生成NCP编码'''
    NCP_dict = {
        'A': [1,1,1],
        'C': [0,1,0],
        'G': [1,0,0],
        'U': [0,0,1],
        'X': [0,0,0]
    }
    return np.array([NCP_dict[base] for base in rna])
    
    
    NCP_map = [NCP_dict[base] for base in rna]  
    return np.array(NCP_map)

    NCP_map = [NCP_dict[base] for base in rna]  
    return np.array(NCP_map)

def to_ND(rna):
    '''生成ND编码'''
    base_counts = {'A':0, 'C':0, 'G':0, 'U':0}
    base_sum = 0
    nd_map = []
    
    for base in rna:
        if base != 'X':
            base_counts[base] += 1
            base_sum += 1
            nd_map.append(base_counts[base] / base_sum)
        else:
            nd_map.append(0.0)
            
    return np.array(nd_map)



def input_preprocess(seq, device):
    '''输入预处理'''
    assert isinstance(seq, np.ndarray)
    return torch.tensor(seq, dtype=torch.float32).to(device)


def encoder(mirna, mrna, device):
    '''编码器'''
    assert mirna == 26 and mrna == 40


def decision_for_whole(data_all):
    '''define if at least one segment is predicted as functional,
    the whole mRNA sequence then will be classified as functional'''
    probabilities = [i[0] for i in data_all]
    count = len([i for i in probabilities if i > 0.5])
    if count >= 1:
        return 1
    else:
        return 0


def specificity_score(y_true, y_pred):
    '''计算特异性指标'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def NPV(y_true, y_pred):
    '''计算阴性预测值'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)