import numpy as np
import pandas as pd
import scipy.io as io
def loaddata(ratio):
    rating=pd.read_csv("ml_latest_small/ratings.csv")
    rating=rating.sample(frac=1)
    trainRating=rating.iloc[:int(np.round(ratio*rating.shape[0]))]
    testRating=rating.iloc[int(np.round(ratio*rating.shape[0])):]
    movieNum=rating['movieId'].unique()
    userNum=rating['userId'].unique()
    trainMatR = pd.DataFrame(index = userNum, columns = movieNum)
    testMatR = pd.DataFrame(index = userNum, columns = movieNum)
    for i in range(len(trainRating)):
        row = trainRating.iloc[[i]]
        userId = row['userId'].tolist()[0]
        itemName = row['movieId'].tolist()[0]
        scores = row['rating'].tolist()[0]
        trainMatR.at[userId,itemName] = scores
    for i in range(len(testRating)):
        row = testRating.iloc[[i]]
        userId = row['userId'].tolist()[0]
        itemName = row['movieId'].tolist()[0]
        scores = row['rating'].tolist()[0]
        testMatR.at[userId,itemName] = scores
    trainMatR.fillna(0, inplace=True)
    testMatR.fillna(0, inplace=True)
    trainres=np.array(trainMatR)
    testMatR=np.array(trainMatR)
    io.savemat("matR.mat", {'trainMatR':trainres,'testMatR':testMatR})