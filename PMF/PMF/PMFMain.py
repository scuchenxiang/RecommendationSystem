from model import *
import argparse
import os
from ml_latest_small import loadData
import scipy.io as io
parse=argparse.ArgumentParser()
parse.add_argument("--ratio",default=0.6,type=float,help="train data and test data ratio")
parse.add_argument("--lr",default=1e-4,type=float,help="")
parse.add_argument("--lambdaU",default=0.01,type=float,help="")
parse.add_argument("--lambdaV",default=0.01,type=float,help="")
parse.add_argument("--epoch",default=1000,type=float,help="")
parse.add_argument("--latentSize",default=20,type=float,help="")
parse.add_argument("--momentSize",default=0.9,type=float,help="")

args=parse.parse_args()

if __name__=="__main__":
    print('the PMF model example for class recommendation system ')
    if os.path.exists("matR.mat")==False:
        print("Generating matR")
        loadData.loaddata(args.ratio)
    trainMatR=io.loadmat("matR.mat")['trainMatR']
    testMatR=io.loadmat("matR.mat")['testMatR']

    print('start train')
    model = PMF(matR=trainMatR, lambdaU=args.lambdaU, lambdaV=args.lambdaV,
                latentSize=args.latentSize, momentSize=args.momentSize, lr=args.lr, epoch=args.epoch, seed=1)
    U, V, trainLossList = model.train(trainData=trainMatR,testData=testMatR)
