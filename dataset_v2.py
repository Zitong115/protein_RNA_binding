# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:43:36 2021

@author: 李梓童
"""

import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import  train_test_split

# 需修改：
# 数据路径
# 标签规则replace 4->1 ; 2 -> 1

NUBASE = {'A':0,'C':1,'G':2,'T':3} #atcg
NUBASE_LEN = 4
RAW_DATA_PATH = '../data/YanJian/0_2_prsced_data/'
VALID_DATA_PATH = '../data/YanJian/0_4_prsced_data_Rep2/' # N,ORF7A,ORF3B
# RAW_DATA_PATH = '../data/prsced_data/'
RANDOM_STATE = 1
NEG_POS_PROPORTION = 1
POSITIVE_SAMPLE = 5000

# build matrix for one sequence
def buildMatrixOneSeq(seq):
    try:
        width = len(seq)
    except:
        print(seq)
    height = NUBASE_LEN
    feature = np.zeros([width,height])
    
    for i,nu in enumerate(seq):
        try:
            feature[i][NUBASE[nu]] = 1
        except:
            #print('error when encoding sequences',seq)
            pass
        
    return feature

class SeqDataset(Dataset):
            
    def __init__(self, learning_file, state = 'Train', k = 0):
        self.learning_file = learning_file
        self.state = state
        self.rep2 = 0
        #full_learning_df = pd.DataFrame(columns=['Seq','Cycle'], index=[])
        
        # change labels
        #for datafile in learning_file_list:
        #    tmpDf = pd.read_csv(RAW_DATA_PATH + datafile,header = 0)
        #    full_learning_df = pd.concat([full_learning_df,tmpDf],ignore_index = True)
        
        # read .csv data file
        full_learning_df = pd.read_csv(RAW_DATA_PATH + self.learning_file,header = 0)
        #full_learning_df['Cycle'].replace(1,0,inplace=True)
        full_learning_df['Cycle'].replace(2,1,inplace=True)
        
        if(learning_file == 'ORF7A.csv' # learning_file == 'N.csv' or  
           or learning_file == 'ORF3B.csv'):
            self.rep2 = 1
            
        if(self.rep2 == 1):
            full_valid_df = pd.read_csv(VALID_DATA_PATH + self.learning_file,header = 0)
            full_valid_df['Cycle'].replace(4,1,inplace=True)
            # 去除训练集中的数据
            full_valid_df = full_valid_df.append(full_learning_df)
            full_valid_df = full_valid_df.append(full_learning_df)
            full_valid_df = full_valid_df.drop_duplicates(subset=['Seq','Cycle'],keep=False)
        
        '''
        if(NEG_POS_PROPORTION == -1):
            1
            #self.full_learning_df = full_learning_df.sample(frac=1,random_state = RANDOM_STATE).reset_index(drop=True)
            #self.full_learning_df = full_learning_df.sample(frac=1).reset_index(drop=True)
        else:
            1
            
            tmpdf_positive = full_learning_df[full_learning_df['Cycle']==1].reset_index(drop=True)
            tmpdf_negative = full_learning_df[full_learning_df['Cycle']==0].reset_index(drop=True)
            
            if(POSITIVE_SAMPLE>0):
                positive_sams = min(POSITIVE_SAMPLE,len(tmpdf_positive))
            else:
                positive_sams = len(tmpdf_positive)
            tmpdf_positive = tmpdf_positive[:positive_sams].reset_index(drop=True)
                
            if(len(tmpdf_negative)<positive_sams * NEG_POS_PROPORTION):
                1
            else:
                tmpdf_negative = tmpdf_negative[:positive_sams * NEG_POS_PROPORTION].reset_index(drop=True)
            
            full_learning_df = pd.concat([tmpdf_positive,tmpdf_negative],ignore_index = True).reset_index(drop=True)
            print('72: ',len(tmpdf_positive),len(tmpdf_negative),positive_sams,len(full_learning_df))
        '''
            
        data,label=full_learning_df['Seq'],full_learning_df['Cycle']
        self.full_learning_df = full_learning_df #.sample(frac=1,random_state = RANDOM_STATE).reset_index(drop=True)
        
        if( self.rep2 == 0 ):
            x_train,x_valid,y_train,y_valid=train_test_split(data,label,test_size=0.3)
            
            self.train_learning_df = pd.concat([x_train,y_train],axis = 1)
            self.valid_learning_df = pd.concat([x_valid,y_valid],axis = 1)
        else:
            self.train_learning_df = full_learning_df.sample(frac=1,random_state = RANDOM_STATE)
            self.valid_learning_df = full_valid_df.sample(frac=1,random_state = RANDOM_STATE)
            
        if self.state == 'Train':
            #print('build training set.')
            self.data = self.train_learning_df
            #self.data = pd.concat(
            #    [self.full_learning_df[:int((k % 10) * len(self.full_learning_df) / 10)], 
            #     self.full_learning_df[int((k % 10 + 1) * len(self.full_learning_df) / 10):]])
            self.data = self.data.reset_index(drop = True)
            self.labels = self.data['Cycle']
            
        if self.state == 'Valid':
            #print('build valid set.')
            self.data = self.valid_learning_df
            #self.data = self.full_learning_df[int((k % 10) * len(self.full_learning_df) /10):
            #                            int((k % 10 + 1) * len(self.full_learning_df) /10)]
            self.data = self.data.reset_index(drop = True)
            self.labels = self.data['Cycle']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    
        one_hot_feature = buildMatrixOneSeq(self.data.loc[idx,'Seq'])
        label = self.data.loc[idx,'Cycle']
        #if( self.data.iloc[idx,1] <  ):
        #    label = 0
        #else:
        #    label = 1
        #label =  0 if self.data.iloc[idx,1] == 1 else 1
        #label = torch.LongTensor(2)
        #label[label_idx] = 1
        
        #sample = {'one_hot_feature': one_hot_feature, 'label': label}

        return (one_hot_feature,label)
    
#def main():
    #learning_file_list = ['RBM14_TATGGA40NAGCC.csv']
    #train_data = SeqDataset(learning_file_list, state = 'Train', k = 0)
    #valid_data = SeqDataset(learning_file_list, state = 'Valid', k = 0)
    #a = buildMatrixOneSeq('TAAAAACACAAAACAACCGAATCTCCGACCGTGAGATTTA')
    #print(a)

#main()