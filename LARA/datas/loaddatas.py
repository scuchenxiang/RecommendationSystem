import numpy as np
import torch
import pandas as pd
class loadDatas(torch.utils.data.Dataset):
    def __init__(self,**kwargs):
        super(loadDatas,self).__init__()
        self.train=kwargs['train']
        ##1为正的训练样本，0为生成的正样本，-1为负样本
        self.user_emb_matrix = np.array(pd.read_csv(r'../datas/user_emb.csv',header=None))
        if self.train==1:
            self.file = np.array(pd.read_csv('../datas/train_data.csv',header =None))
        elif self.train==-1:
            self.file=np.array(pd.read_csv('../datas/neg_data.csv',header =None))
        else:
            self.testItem = np.array(pd.read_csv('../datas/test_item.csv',header =None).astype(np.int32))
            self.testAttribute = np.array( pd.read_csv( '../datas/test_attribute.csv',header =None).astype(np.int32))

    def __getitem__(self, index):
        if self.train==0:#0为测试情况
            return np.array([self.testItem[index]]),np.array(self.testAttribute[index])
        data=self.file[index]
        user_batch = np.array([data[0]])
        item_batch = np.array([data[1]])
        attr_batch = (data[2][1:-1].split())
        real_user_emb_batch = self.user_emb_matrix[user_batch]
        for i in range(len(attr_batch)):
            attr_batch[i]=int(attr_batch[i])
        attr_batch=np.array(attr_batch)
        return user_batch,item_batch,attr_batch,real_user_emb_batch
    def __len__(self):
        if self.train!=0:
            return len(self.file)
        else:
            return  min(len(self.testItem),len(self.testAttribute))
