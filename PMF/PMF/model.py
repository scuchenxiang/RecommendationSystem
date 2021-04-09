import numpy as np
from numpy.random import RandomState
class PMF():
    def __init__(self, **kwargs):
        self.matR = kwargs['matR']
        self.lambdaU = kwargs['lambdaU']
        self.lambdaV = kwargs['lambdaV']
        self.latentSize=kwargs['latentSize']
        self.randomState = RandomState(kwargs['epoch'])
        self.momentSize = kwargs['momentSize']
        self.lr = kwargs['lr']
        self.epoch = kwargs['epoch']
        self.I=self.matR.copy()
        self.I[self.I != 0] = 1

        self.U = 0.1*self.randomState.rand(np.size(self.matR, 0), self.latentSize)
        self.V = 0.1*self.randomState.rand(np.size(self.matR, 1), self.latentSize)


    def loss(self):
        # the loss function of the model
        loss = np.sum(self.I*(self.matR-np.dot(self.U, self.V.T))**2) + self.lambdaU*np.sum(np.square(self.U)) + self.lambdaV*np.sum(np.square(self.V))
        return loss
    def predict(self, data):
        res=np.dot(self.U,self.V.T)
        res[data==0]=0
        rmse=np.sqrt(np.mean(np.square(data-res)))
        return rmse

    def train(self, trainData=None, testData=None):
        trainLoss = []
        testRMSE=[]

        momuntumU = np.zeros(self.U.shape)
        momuntumV = np.zeros(self.V.shape)

        for iter in range(self.epoch):
            # derivate of Vi
            gradsU = np.dot(self.I*(self.matR-np.dot(self.U, self.V.T)), -self.V) + self.lambdaU*self.U

            # derivate of Tj
            gradsV = np.dot((self.I*(self.matR-np.dot(self.U, self.V.T))).T, -self.U) + self.lambdaV*self.V

            # update the parameters
            momuntumU = (self.momentSize * momuntumU) + self.lr * gradsU
            momuntumV = (self.momentSize * momuntumV) + self.lr * gradsV
            self.U = self.U - momuntumU
            self.V = self.V - momuntumV

            # training evaluation
            loss = self.loss()
            trainLoss.append(loss)
            testRmse=self.predict(testData)
            testRMSE.append(testRmse)
            print("the iteration:{} ,trainLoss:{}, testRmse:{}".format(iter, loss, testRmse))


        return self.U, self.V, trainLoss
