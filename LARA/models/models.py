import torch
import numpy as np
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from datas import loaddatas
import pandas as pd
import os
from predict import evaluation
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


parser=argparse.ArgumentParser()
parser.add_argument("--epochs",type=int,default=300,help="number of epochs of training")
parser.add_argument("--batchSize",type=int,default=2048,help="num of batchsize")
parser.add_argument("--attrNum",type=int,default=18,help="")
parser.add_argument("--hiddenDim",type=int,default=100,help="")
parser.add_argument("--alpha",type=int,default=0,help="")
parser.add_argument("--userEmbDim",type=int,default=18,help="")
parser.add_argument("--attrPresentDim",type=int,default=5,help="")
parser.add_argument("--lr",type=float,default=0.0001,help="")

args=parser.parse_args()
cuda=True if torch.cuda.is_available() else False
useCuda=cuda and True


class discriminator(nn.Module):
    def __init__(self,**kwargs):
        super(discriminator,self).__init__()
        self.attrNum=kwargs['attrNum']
        self.attrPresentDim=kwargs['attrPresentDim']
        self.userEmbDim=kwargs['userEmbDim']
        self.hiddenDim=kwargs['hiddenDim']
        self.lineLayer1=nn.Linear(self.attrNum*self.attrPresentDim  + self.userEmbDim , self.hiddenDim)
        self.lineLayer2=nn.Linear(self.hiddenDim, self.hiddenDim)
        self.lineLayer3=nn.Linear(self.hiddenDim, self.userEmbDim)
        self.disAttrMat=nn.Embedding(2*self.attrNum, self.attrPresentDim)
        nn.init.xavier_uniform_(self.lineLayer1.weight)
        nn.init.xavier_uniform_(self.lineLayer2.weight)
        nn.init.xavier_uniform_(self.lineLayer3.weight)
        nn.init.xavier_uniform_(self.disAttrMat.weight)
    def forward(self,attributeId,userEmb):
        attrPresent=self.disAttrMat(attributeId)
        feature=torch.reshape(attrPresent,[-1,self.attrNum*self.attrPresentDim])
        # ipdb.set_trace()
        arrtFeatUser=torch.cat([feature,userEmb],dim=1)
        disRes=self.lineLayer3(torch.tanh(self.lineLayer2(torch.tanh(self.lineLayer1(arrtFeatUser)))))
        disProb=torch.sigmoid(disRes)
        return disProb,disRes


class generator(nn.Module):
    def __init__(self,**kwargs):
        super(generator,self).__init__()
        self.attrNum=kwargs['attrNum']
        self.attrPresentDim=kwargs['attrPresentDim']
        self.userEmbDim=kwargs['userEmbDim']
        self.hiddenDim=kwargs['hiddenDim']
        self.lineLayer1=nn.Linear(self.attrNum*self.attrPresentDim , self.hiddenDim)
        self.lineLayer2=nn.Linear(self.hiddenDim, self.hiddenDim)
        self.lineLayer3=nn.Linear(self.hiddenDim, self.userEmbDim)
        self.genArrtMat=nn.Embedding(2*self.attrNum, self.attrPresentDim)
        nn.init.xavier_uniform_(self.lineLayer1.weight)
        nn.init.xavier_uniform_(self.lineLayer2.weight)
        nn.init.xavier_uniform_(self.lineLayer3.weight)
        nn.init.xavier_uniform_(self.genArrtMat.weight)
    def forward(self,attributeId):
        attrPresent=self.genArrtMat(attributeId)
        feature=torch.reshape(attrPresent,[-1,self.attrNum*self.attrPresentDim])
        fakeuser=torch.tanh(self.lineLayer3(torch.tanh(self.lineLayer2(torch.tanh(self.lineLayer1(feature))))))
        return fakeuser
def train():
    if useCuda:
        print("use Cuda for train")
        torch.cuda.set_device(0)
    else:
        print("Do not use Cuda")
    traindata=loaddatas.loadDatas(train=1)
    negadata=loaddatas.loadDatas(train=-1)
    testdata=loaddatas.loadDatas(train=0)
    train_loader = DataLoader(dataset=traindata, batch_size=args.batchSize, shuffle=True)
    nega_loader = DataLoader(dataset=negadata, batch_size=args.batchSize, shuffle=True)
    test_loader=DataLoader(dataset=testdata,batch_size=args.batchSize,shuffle=True)
    if useCuda:
        G=generator(attrNum=args.attrNum,attrPresentDim=args.attrPresentDim,userEmbDim=args.userEmbDim,hiddenDim=args.hiddenDim).cuda()
        D=discriminator(attrNum=args.attrNum,attrPresentDim=args.attrPresentDim,userEmbDim=args.userEmbDim,hiddenDim=args.hiddenDim).cuda()
    else:
        G=generator(attrNum=args.attrNum,attrPresentDim=args.attrPresentDim,userEmbDim=args.userEmbDim,hiddenDim=args.hiddenDim)
        D=discriminator(attrNum=args.attrNum,attrPresentDim=args.attrPresentDim,userEmbDim=args.userEmbDim,hiddenDim=args.hiddenDim)
    optimizerG = torch.optim.Adam(G.parameters(), lr=args.lr,weight_decay=args.alpha)
    optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr,weight_decay=args.alpha)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=(0.1)**0.5)
    # criterion=torch.nn.CrossEntropyLoss()
    print("start train")
    for epo in range(args.epochs):
        Gloss=0
        Dloss=0
        for i,(user_batch,item_batch,attr_batch,real_user_emb_batch) in enumerate(train_loader):
            j,(nega_user_batch,nega_item_batch,nega_attr_batch,nega_user_emb_batch) = next(enumerate(nega_loader))
            G.train()
            if useCuda:
                itemBatch=torch.from_numpy(np.asarray(attr_batch)).type(torch.LongTensor).cuda()
                negaitemBatch=torch.from_numpy(np.asarray(nega_attr_batch)).type(torch.LongTensor).cuda()

                fakeuseremb=G(itemBatch)

                _,Dfakelogic=D(itemBatch,fakeuseremb.float())
                Dfakelogic=Dfakelogic.cpu()

                _,Dreallogic=D(itemBatch,real_user_emb_batch.squeeze(1).float().cuda())
                Dreallogic=Dreallogic.cpu()

                _, Dnegativelogic = D(negaitemBatch, nega_user_emb_batch.squeeze(1).float().cuda())
                Dnegativelogic=Dnegativelogic.cpu()
            else:
                itemBatch=torch.from_numpy(np.asarray(attr_batch)).type(torch.LongTensor)
                negaitemBatch=torch.from_numpy(np.asarray(nega_attr_batch)).type(torch.LongTensor)

                fakeuseremb=G(itemBatch)

                _,Dfakelogic=D(itemBatch,fakeuseremb.float())

                _,Dreallogic=D(itemBatch,real_user_emb_batch.squeeze(1).float())

                _, Dnegativelogic = D(negaitemBatch, nega_user_emb_batch.squeeze(1).float())

            real_label = (torch.ones_like(Dreallogic)).type(torch.LongTensor)# 定义真实的图片label为1
            fake_label = (torch.zeros_like(Dfakelogic)).type(torch.LongTensor)
            nega_label=(torch.zeros_like(Dnegativelogic)).type(torch.LongTensor)
            G_label=torch.ones_like(Dfakelogic).type(torch.LongTensor)

            fakeloss = sigmoidCrossEntryLoss(Dfakelogic,fake_label)
            realloss=sigmoidCrossEntryLoss(Dreallogic,real_label)
            negaloss=sigmoidCrossEntryLoss(Dnegativelogic,nega_label)

            finalDloss=myLoss(args.alpha,realloss,fakeloss,negaloss)
            optimizerD.zero_grad()  # 判别器清空上一步残余更新参数值
            finalDloss.backward() # 误差反向传播，计算参数更新值
            optimizerD.step() # 将参数更新值施加到net的parmeters上

            if useCuda:
                fakeuseremb=G(itemBatch).cpu()
            else:
                fakeuseremb=G(itemBatch)
            finalGloss=sigmoidCrossEntryLoss(fakeuseremb,G_label)
            optimizerG.zero_grad()  # 清空上一步残余更新参数值
            finalGloss.backward() # 误差反向传播，计算参数更新值
            optimizerG.step() # 将参数更新值施加到net的parmeters上
            Gloss+=finalGloss
            Dloss+=finalDloss
        print("thr {} epoch Gloss= {}".format(epo,Gloss))
        print("thr {} epoch Dloss= {}".format(epo,Dloss))
        if epo %10==0:
            with torch.no_grad():
                G.eval()
                for i,(testItem, testAttribute) in enumerate(test_loader):
                    if useCuda:
                        testFakeItem=torch.from_numpy(np.asarray(testAttribute)).type(torch.LongTensor).cuda()
                        fakeuseremb=G(testFakeItem).cpu()
                    else:
                        testFakeItem=torch.from_numpy(np.asarray(testAttribute)).type(torch.LongTensor)
                        fakeuseremb=G(testFakeItem)
                    predictindex=test(testItem,fakeuseremb)
                    print("the p10,p20,Mp10,Mp20,NDCG10,NDCG20{}".format(predictindex))

def sigmoidCrossEntryLoss(logits,p):
    loss = torch.mean(-p*torch.log(torch.sigmoid(logits)) + (p-1)*torch.log(1-torch.sigmoid(logits)))
    return loss
def myLoss(alpha,D_loss_real,D_loss_fake,D_loss_counter):
    res=(1-alpha)*(D_loss_real + D_loss_fake  + D_loss_counter)
    return res

def getKSimilarUser(GUser, k):
    userArrtLinkMat=np.matmul(GUser,userAttributeMatrix.T)
    sortedIndex = np.argsort(-userArrtLinkMat)#按照降序排列
    return sortedIndex[:, 0:k]


userAttributeMatrix = np.array(pd.read_csv(r'../datas/user_attribute.csv',header=None))
userItemMat = np.array(pd.read_csv(r'../datas/ui_matrix.csv',header=None))

def test(testItemBatch,testGUser):
    kValue = 20
    testBatchSize = np.shape(testItemBatch)[0]
    testInterSectionSimilarUser = getKSimilarUser(testGUser, kValue)
    count = 0
    for item, testUserList in zip(testItemBatch, testInterSectionSimilarUser):
        for user in testUserList:
            if userItemMat[user, item] == 1:
                count = count + 1
    #精度为20的结果
    pAt20 = round(count/(testBatchSize * kValue), 4)

    ans = 0.0
    RS = []
    for item, testUserlist in zip(testItemBatch, testInterSectionSimilarUser):
        r=[]
        for user in testUserlist:
            r.append(userItemMat[user][item])
        RS.append(r)
    #平均精度
    MpAt20 = evaluation.mean_average_precision(RS)

    ans = 0.0
    for item, testUserlist in zip(testItemBatch, testInterSectionSimilarUser):
        r=[]
        for user in testUserlist:
            r.append(userItemMat[user][item])
        ans = ans + evaluation.ndcg_at_k(r, kValue, method=1)
    #归一化贴现累积增益(NDCG)
    NDCGAt20 = ans/testBatchSize


    kValue = 10
    count = 0
    for item, testUserList in zip(testItemBatch, testInterSectionSimilarUser):
        for user in testUserList[:kValue]:
            if userItemMat[user, item] == 1:
                count = count + 1
    pAt10 = round(count/(testBatchSize * kValue), 4)

    ans = 0.0
    RS = []
    for item, testUserList in zip(testItemBatch, testInterSectionSimilarUser):
        r=[]
        for user in testUserList[:kValue]:
            r.append(userItemMat[user][item])
        RS.append( r)
    MpAt10 = evaluation.mean_average_precision(RS)


    ans = 0.0
    for item, testUserList in zip(testItemBatch, testInterSectionSimilarUser):
        r=[]
        for user in testUserList[:kValue]:
            r.append(userItemMat[user][item])
        ans = ans + evaluation.ndcg_at_k(r, kValue, method=1)
    NDCGAt10 = ans/testBatchSize


    return pAt10,pAt20,MpAt10,MpAt20,NDCGAt10,NDCGAt20
if __name__=="__main__":
    train()