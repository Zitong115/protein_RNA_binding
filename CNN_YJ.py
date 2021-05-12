#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np
import dataset
import torch.nn.init as init
from time import time
import logging
import data_preprocessing
from sklearn.metrics import matthews_corrcoef
import math
from LDAM_DRW.losses import LDAMLoss, FocalLoss
import datetime

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

# 参数
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.01 
MAX_EPOCH = 15
BATCH_SIZE = 50
LOG_PATH = 'log_output/'
LOG_FILE = LOG_PATH + 'cnn_v3_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.log'
ATTR_PATH = '0_2_attr_score/'

torch.manual_seed(1)
np.random.seed(1)
LEARNING_FILE_PATH = '../data/YanJian/0_2_prsced_data/'
LEARNING_FILE_LIST =  ['NSP15.csv','ORF7A.csv','ORF9B.csv'] #data_preprocessing.getFastqFile(LEARNING_FILE_PATH) #'NSP4.csv','M.csv','ORF7B.csv']

#data_preprocessing.getFastqFile(LEARNING_FILE_PATH)[:2]data_preprocessing.getFastqFile(LEARNING_FILE_PATH)

''''
['RBFOX1-construct4_TGAAGG40NGAT.csv',
                      'PCBP1_TTTCGG40NGATG.csv',
                      'PABPC5_TACCTG40NACG.csv',
                      'HNRNPA3_TCGTAG40NCAGC.csv',
                      'ZC3H12C_TAAGTT40NGACA.csv',
                      'QKI_TAAGTT40NTCGA.csv',
                      'HNRNPA1L2_TGAATC40NGTA.csv']
QKI_TCACTG40NACC.csv', # mcc:0.65
                      'DAZ3_TGAAGA40NACAG.csv', # mcc:0.59
                      'ZC3H12C_TAAGTT40NGACA.csv', #mcc:0.44
                      'QKI_TAAGTT40NTCGA.csv',] # mcc:0.67
'NOVA2-construct3_TATTCT40NCTC.csv',
                      'BOLL_TCCCAA40NGCGC.csv',
                      'QKI_TCACTG40NACC.csv',
                      'YBX2_TCTCAC40NACCT.csv',
                      'DAZ3_TGAAGA40NACAG.csv',
                      'MSI1_TTCTAA40NAAT.csv',
                      'IGF2BP1_TCTTAA40NGCA.csv',
                      'ZC3H12A_TGATGA40NCATT.csv',
'''
#data_preprocessing.getFastqFile(LEARNING_FILE_PATH)[:3]

#['DAZ1-construct2_TGGCCT40NATTT.csv','RBFOX1-construct2_TTGCGA40NTACG.csv',
# 'PUM2_TCCCGT40NCCGA.csv','RBFOX1-construct2_TCTGAT40NCTA.csv']

def createLog():
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
    # 输出到file
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
    fh.setLevel(logging.DEBUG)
    
    logger_name = "cnn_yj_log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    '''
    logging.basicConfig(filename=LOG_FILE,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', 
                        level = logging.DEBUG,filemode='a',
                        datefmt='%Y-%m-%d%I:%M:%S %p')
    logger_name = "cnn_log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    '''
    return logger

logger = createLog()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                # patch 7 * 7; 1 in channels; 32 out channels; stride is 1
                # padding style is same(that means the convolution opration's input and output have the same size)
                in_channels = 1,
                out_channels = 32,
                kernel_size = (7,5),
                stride = 1,
                padding = 3,
            ),
            nn.ReLU(),        # 激活函数
            nn.MaxPool2d(2),  # 池化函数
        )
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = (5,5),
                stride = 1,
                padding = 2,
            ),
            nn.Tanh(),
            nn.MaxPool2d(2),

        )
            
        for m in self.conv1.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                
        for m in self.conv2.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                
        self.out1 = nn.Linear( 10*1*64 , 256, bias = True)  # 全连接层
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.out2 = nn.Linear(256, 2, bias = True)
        
        init.xavier_uniform_(self.out1.weight, gain=1)
        init.xavier_uniform_(self.out2.weight, gain=1)

    def forward(self, x):
        #print('begin:',x.shape)
        #begin: torch.Size([50, 1, 40, 4])

        x = self.conv1(x)
        #print('conv1:',x.shape)
        #conv1: torch.Size([50, 32, 20, 3])
        
        x = self.conv2(x)
        #print('conv2:',x.shape)
        #conv1: torch.Size([50, 64, 10, 1])
        
        x = x.view(x.shape[0],10*1*64)  
        out1 = self.out1(x)
        out1 = F.relu(out1)
        #out1 = self.dropout(out1)
        out2 = self.out2(out1)
        output = F.softmax(out2, dim = 1)
        return output

def Evaluate(precited,expected):
    
    res = np.logical_xor(precited,expected).astype(int)
    r = np.bincount(res)
    #print(r)
    
    #print(precited[:10],expected[:10],type(precited),type(expected))
    tp_list = np.logical_and(precited,expected).astype(int)
    fp_list = np.logical_and(precited,np.logical_not(expected)).astype(int)
    tp_list = tp_list.tolist()
    fp_list = fp_list.tolist()
    
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = r[0]-tp
    fn = r[1]-fp
    
    print(tp,fp,tn,fn)
    
    if tp + fp > 0 :
        p = tp/(tp+fp)
    else:
        p = 0
    
    if tp + fn > 0:
        recall = tp/(tp+fn)
    else:
        recall = 0
        
    if 2*tp+fn+fp > 0:
        F1=(2*tp)/(2*tp+fn+fp)
    else:
        F1 = 0
        
    acc=(tp+tn)/(tp+tn+fp+fn)
    
    '''
    tp += 1
    tn += 1
    fp += 1
    fn += 1
    '''
    mcc = matthews_corrcoef(expected,precited)
    # (tp*tn-fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return tp,fp,tn,fn, p,recall,F1,acc,mcc

def genScore(score_batch):
    ret = score_batch.detach().numpy()
    ret = np.sum(ret, axis=(0, 1,))
    return ret

def avgScore(score_mat,sum_vec):
    for i in range(score_mat.shape[0]):
        score_mat[i] = score_mat[i] / sum_vec[i]
    
    return score_mat

#测试
def test(cnn,valid_data,valid_loader,learning_file):
    cnn.to("cpu")
    cnn.eval()
    #correct = 0
    #prediction_positive = 0
    
    '''
    my_array = np.array(valid_data.data.values).astype(float)
    print(type(my_array))
    my_tensor = torch.tensor(my_array)
    valid_x = torch.unsqueeze(my_tensor, dim = 1).type(torch.FloatTensor)
    valid_y = valid_data.labels.numpy()
    
    y_pre = cnn(valid_x)
    _,pre_index = torch.max(y_pre, 1)
    pre_index = pre_index.view(-1)
    prediction = pre_index.numpy()
    
    tp,fp,tn,fn, p,recall,F1,acc = Evaluate(prediction,valid_y)
    
    logger.info( "TP:{:d},FP:{:d},TN:{:d},FN:{:d}".format(tp,fp,tn,fn))
    logger.info( "precision:{:.2f},recall:{:.2f},F1:{:.2f},accuracy:{:.2f}".format(p,recall,F1,acc))
    '''
    
    precite = np.array([])
    expected = np.array([])
    attr_score = np.empty([40,4],dtype=np.float64)
    #ig = IntegratedGradients(cnn)
    dl = DeepLift(cnn)
    
    #t1 = time()
    for i, data in enumerate(valid_loader):
        # forward
        
        inputs, labels = data
        inputs = torch.unsqueeze(inputs, dim = 1).type(torch.FloatTensor)
        
        inputs = inputs#.to(device)
        labels = labels#.to(device)
        
        # for IntegratedGradients
        
        #baseline = torch.zeros(inputs.shape)#.to(device)
        baseline_np = np.array([[[0.3,0.2,0.2,0.3]]*40]*inputs.shape[0])
        baseline = torch.from_numpy(baseline_np)
        baseline = torch.unsqueeze(baseline, dim = 1).type(torch.FloatTensor)
        attributions, delta = dl.attribute(inputs, baseline, target=0, return_convergence_delta=True)#.to(device)
        
        #logger.info('IG Attributions:' + str(attributions))
        #logger.info('Convergence Delta:' + str(delta))
        
        outputs = cnn(inputs)
        
        attr_score += genScore(attributions)
        #print('echo test:',i,time()-t1)
        _, pre_index = torch.max(outputs.data, 1)
        pre_index = pre_index.view(-1)
        prediction = pre_index.numpy()
        labels = labels.numpy()
        
        precite = np.concatenate((precite, prediction)) 
        expected = np.concatenate((expected, labels)) 
        #prediction_positive += np.sum(prediction)
        #correct += np.sum(prediction == labels)
    
    attr_score_sum = np.sum(attr_score,axis = 1)
    attr_score = avgScore(attr_score,attr_score_sum)
    np.save(  ATTR_PATH + 'attributions_Rep1_'+learning_file.split('_')[0][:-4]+'.npy',attr_score)
    
    cnn.to("cpu")
    tp,fp,tn,fn, p,recall,F1,acc,mcc = Evaluate(precite,expected)
    
    '''
    logger.info( "TP:{:d},FP:{:d},TN:{:d},FN:{:d}".format(tp,fp,tn,fn))
    logger.info( "precision:{:.2f},recall:{:.2f},F1:{:.2f},accuracy:{:.2f}".format(p,recall,F1,acc))
    logger.info( "MCC:{:.2f}".format(mcc) )
    '''
    
    logger.info( "TP:"+str(tp))
    logger.info( "FP:"+str(fp))
    logger.info( "TN:"+str(tn))
    logger.info( "FN:"+str(fn))
    
    logger.info( "Precision:{:.4f}".format(p))
    logger.info( "Recall:{:.4f}".format(recall))
    logger.info( "F1:{:.4f}".format(F1))
    logger.info( "Accuracy:{:.4f}".format(acc))
    logger.info( "MCC:{:.4f}".format(mcc))
    
    #test_accuracy = correct / len(valid_data)
    #print("Test accuracy: ", test_accuracy )
    
    #logger.info( "Predicted positive: " + str( prediction_positive ) )
    #logger.info( "Test accuracy: " + str( test_accuracy ) )
    
    return tp,fp,tn,fn

#训练
def train(cnn,train_data,train_loader,
          max_epoch = MAX_EPOCH,learning_rate = LEARNING_RATE ):
    
    #[ class 0 count, class 1 count]
    cls_num_list = [len(train_data)-np.sum(np.array(train_data.labels)), \
                    np.sum(np.array(train_data.labels))]
    
    idx = 1 #max_epoch // 12
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    cnn=cnn.to(device)
    
    cnn.train()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    #loss_func = nn.CrossEntropyLoss().to(device)
    
    
    loss_func = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).to(device)
    
    loss_mean = 0.
    total_steps = len(train_loader)
    interval = int(total_steps/5)
    
    #print('total steps: ', total_steps, 'intervals: ', interval)
    logger.debug( 'total steps: ' + str( total_steps ) +  ' intervals: ' + str( interval ))
    
    t1 = time()
    
    for epoch in range(max_epoch):
        # len(train_loader)=1200
        # after 2 epoch, the accuracy can reach 0.96
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = torch.unsqueeze(x, dim = 1).type(torch.FloatTensor)
            
            x = x.to(device)
            y = y.to(device)
            
            output = cnn(x)
            #_, outputs = torch.max(output.data, 1)
            #outputs = outputs.view(-1).float()
            #y = y.float()
            loss = loss_func(output, y)
            
            loss.backward()
            optimizer.step()
            
            loss_mean += loss.item()
                
            if step != 0 and step % interval == 0:
                #print("=" * 10, epoch+1, "=" * 10, step, "=" * 10)
                #print("loss is ", loss_mean/interval )
                logger.debug( "Epoch: " + str( epoch+1 ) + " Step: " + str(step)  + " Loss: " + str(loss_mean/interval))
                #logger.debug( 'total steps: ' + str( total_steps ) +  'intervals: ' + str( interval ))
                loss_mean = 0.
    
    t2 = time()
    time_cost = t2 - t1
    #print('Training time cost: ',time_cost)
    logger.info( 'Training time cost: ' + str(time_cost) )
    
    return cnn
    
    #test_accuracy = test(cnn)
    #print("Test accuracy: ", test_accuracy )
    
    #return 

def read_data(learning_file):
    
    # 设置训练集
    train_data = dataset.SeqDataset(learning_file, state = 'Train', k = 0)

    # 设置验证集
    valid_data = dataset.SeqDataset(learning_file, state = 'Valid', k = 0)
    
    '''
    print('num_of_trainData:', len(train_data))
    print('num_of_validData:', len(valid_data))
    
    print('train positive label sum:', np.sum(np.array(train_data.labels)))
    print('valid positive label sum:', np.sum(np.array(valid_data.labels)))
    '''
    
    logger.info('num_of_trainData:' + str(len(train_data)))
    logger.info('num_of_validData:' + str(len(valid_data)))
    
    logger.info('train positive label sum:'+ str(np.sum(np.array(train_data.labels))))
    logger.info('valid positive label sum:'+ str(np.sum(np.array(valid_data.labels))))
    
    logger.info('Positive samples proportion:{:.5f}'.format((np.sum(np.array(train_data.labels))+np.sum(np.array(valid_data.labels)))/(len(train_data)+len(valid_data))))
    
    train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
    valid_loader = Data.DataLoader(dataset = valid_data, batch_size = int(len(valid_data)/50), shuffle = True)

    return train_data,valid_data,train_loader,valid_loader
    
if __name__ == '__main__':
    
    for learning_file in LEARNING_FILE_LIST:
        logger.info('Processing file:' + learning_file)
        logger.info('Protein name:' + learning_file.split('_')[0])
        # logger.info('Barcode:' + learning_file.split('_')[1][:-4])
        train_data,valid_data,train_loader,valid_loader = read_data(learning_file)
        cnn = CNN()
        cnn = train(cnn, train_data,train_loader,
                    max_epoch = MAX_EPOCH,learning_rate = LEARNING_RATE)
        accuracy = test(cnn,valid_data,valid_loader,learning_file)
        
    #cnn = CNN()
    #train(cnn)
