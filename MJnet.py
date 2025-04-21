import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import numpy.random as random

from tqdm import tqdm
from captum.attr import IntegratedGradients
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.metrics import recall_score,f1_score
from sklearn.metrics import accuracy_score,average_precision_score, precision_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PAD_MIRNA_LENGTH = 26
PAD_MRNA_LENGTH = 40


class TextCNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size, kernel_size) for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len) for Conv1d
        conv_results = [torch.relu(conv(x)) for conv in self.convs]  # List of (batch_size, hidden_size, seq_len)
        pooled = [torch.max(conv_result, dim=2)[0] for conv_result in conv_results]  # Max pooling over seq_len
        output = torch.cat(pooled, dim=1)  # (batch_size, len(kernel_sizes) * hidden_size)
        return self.dropout(output)
    

def preprocess_features(mrna, mirna, device):
    reverse_mrna = reverse_seq(mrna)
    # pairing_m, pairing_mi = get_interaction_map(mirna, reverse_mrna)

    features = {
        'C2_m': to_C2(reverse_mrna),
        'C2_mi': to_C2(mirna),
  }
    return {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in features.items()}

class myDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        mirna, mrna, label = self.data[index]
        mirna = mirna + 'X' * (PAD_MIRNA_LENGTH - len(mirna))
        features = preprocess_features(mrna, mirna, device)
        features['label'] = torch.tensor(label, dtype=torch.float32).to(device)
        return features
    
class MJnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, output_size):
        super(MJnet, self).__init__()
        
        # GRU modules
        self.gru_m = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.gru_mi = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True)

        # Self-attention for m and mi
        self.self_attention_m = nn.MultiheadAttention(hidden_size * 2, num_heads, dropout=dropout)
        self.self_attention_mi = nn.MultiheadAttention(hidden_size * 2, num_heads, dropout=dropout)

        # TextCNN
        self.textcnn = TextCNN(input_size=hidden_size * 2, hidden_size=128, kernel_sizes=[3, 5, 7], dropout=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(768, 128)  # 3 kernels * 128 hidden_size * 2
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, C2_m, C2_mi):
        
        m_input = C2_m  # (batch_size, seq_len, 7)
        mi_input = C2_mi  # (batch_size, seq_len, 7)

        # GRU
        m_emb, _ = self.gru_m(m_input)  # (batch_size, seq_len, 2 * hidden_size)
        mi_emb, _ = self.gru_mi(mi_input)  # (batch_size, seq_len, 2 * hidden_size)

        # Self-attention on m
        m_emb = m_emb.permute(1, 0, 2)  # (seq_len, batch_size, 2 * hidden_size)
        m_self_attn, _ = self.self_attention_m(m_emb, m_emb, m_emb)  # Self-attention
        m_self_attn = m_self_attn.permute(1, 0, 2)  # (batch_size, seq_len, 2 * hidden_size)

        # Self-attention on mi
        mi_emb = mi_emb.permute(1, 0, 2)  # (seq_len, batch_size, 2 * hidden_size)
        mi_self_attn, _ = self.self_attention_mi(mi_emb, mi_emb, mi_emb)  # Self-attention
        mi_self_attn = mi_self_attn.permute(1, 0, 2)  # (batch_size, seq_len, 2 * hidden_size)

        # TextCNN on m_self_attn and mi_self_attn
        m_cnn_output = self.textcnn(m_self_attn)  # (batch_size, len(kernel_sizes) * hidden_size)
        mi_cnn_output = self.textcnn(mi_self_attn)  # (batch_size, len(kernel_sizes) * hidden_size)

        # Combine TextCNN outputs
        combined_output = torch.cat((m_cnn_output, mi_cnn_output), dim=1)  # (batch_size, len(kernel_sizes)*hidden_size*2)

        # Fully connected layers
        output = self.fc1(combined_output)
        output = self.dropout(output)
        output = torch.relu(output)
        output = self.batch_norm1(output)
        output = self.fc2(output)

        return output


def process_batch(data, device):
    features = {k: data[k].to(device) for k in data.keys() if k != 'label'}
    labels = data['label'].to(device).unsqueeze(1)
    return features, labels




def get_cts(rmrna, stepsize):
    '''segment full-length mRNAS into 40-nt segments using a sliding window with predefined stepsize'''
    kmers = []

    if len(rmrna) >= 40:
        for i in range(0, len(rmrna),stepsize):
            if i + 40 <= len(rmrna):
                cut = rmrna[i:i + 40]
                kmers.append(cut)

        return kmers
    else:
        pad_rmrna = rmrna + 'X' * (40 - len(rmrna))
        kmers.append(pad_rmrna)
        return kmers



def kmers_predict(kmers,mirna,model):

    mirna = mirna + 'X'*(26-len(mirna))
    fea_C2_m = []
    fea_C2_mi = []


    if len(kmers) == 0:
        return 0
    else:
        for i in kmers:
           
            C2_m = to_C2(i)
            C2_mi = to_C2(mirna)

           
            fea_C2_m.append(C2_m)
            fea_C2_mi.append(C2_mi)
           
            

        fea_C2_m = torch.tensor(np.array(fea_C2_m), dtype=torch.float32).to(device)
        fea_C2_mi = torch.tensor(np.array(fea_C2_mi), dtype=torch.float32).to(device)
        
        
        model = model.to(device)
        model.eval()
        pros = model(fea_C2_m, fea_C2_mi).detach().cpu().numpy().tolist()
        pppp = decision_for_whole(pros)

        return pppp




def perform_test(pathfile, stepsize, model_type):

    test = read_test(pathfile)
    y_true = []
    y_pred = []

    model = MJnet(input_size=2, hidden_size=128, num_layers=2, num_heads=8, dropout=0.4, output_size=1)  # 创建模型实例
    model.load_state_dict(torch.load(model_type))  # 加载模型的 state_dict
    model = model.to(device)
    print('个数',len(test))

    for index in range(len(test)): #range(len(test))
        fasta = test[index]
        
        mirna = fasta[0].upper().replace('T', 'U')
        mrna = fasta[1].upper().replace('T', 'U')
        reverse_mrna = reverse_seq(mrna)
        y_true.append(fasta[2])
        kmers = get_cts(reverse_mrna,stepsize)

        if kmers is None:
            pre = 0
            y_pred.append(pre)

        else:

            pre = kmers_predict(kmers, mirna, model)
            y_pred.append(pre)

    print(y_true)
    print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    npv = NPV(y_true, y_pred)
    
    print('acc', acc)
    print('PPV', pre)
    print('recall', recall)
    print('specificity', spec)
    print('f1', f1)
    print('NPV', npv)

