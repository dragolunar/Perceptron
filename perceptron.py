import math
import random
import datetime
import enum


class Function(enum.Enum):

    NONE                = 0
    THRESHOLD           = 1
    SIGMOID             = 2
    HYPERBOLIC_TANGENT  = 3


class Perceptron:

    def __init__(self, dimension, function=1):
        
        self.inputVector = []
        self.weightsVector = []
        self.bias = 0.0
        self.activationFunction = 0

        random.seed(datetime.datetime.now())
        self.inputVector = [None]*dimension
        for i in range(dimension):
            self.weightsVector.append(random.random() - 0.5)
        self.bias = random.random()
        self.activationFunction = function
        
    def setWeights(self, weightsVector):
        
        self.weightsVector = weightsVector
        
    def setBias(self, biasParam):
        
        self.bias = biasParam

    def setSample(self, inputVector):
        
        self.inputVector = inputVector
        
    def calculateNet(self):

        action = 0.0
        for i in range(len(self.inputVector)):
            action += self.inputVector[i]*self.weightsVector[i]
            action -= self.bias

        if (self.activationFunction == Function.THRESHOLD):
            if (action >= 0.0):
                action = 1
            else:
                action = 0
        elif (self.activationFunction == Function.SIGMOID):
            action = 1.0/(1.0 + math.exp(-action))
        elif (self.activationFunction == Function.HYPERBOLIC_TANGENT):
            action = (math.exp(2*action) - 1)/(math.exp(2*action) + 1)
        else:
            print('set 1-3 to activation function as THRESHOLD, SIGMOID, or HYPERBOLIC_TANGENT')
            exit()

        return action

    def adjustWeights(self, teachingStep, output, target):

        for i in range(len(self.weightsVector)):
            # error correction learning rule
            self.weightsVector[i] += teachingStep*(target - output)*self.inputVector[i]
        self.bias -= teachingStep*(target - output)

    def recall(self, inputVector):
        
        self.setSample(inputVector)
        return self.calculateNet()


if __name__ == "__main__":
    
    DIMENSION               = 2
    LEAST_MEAN_SQUARE_ERROR = 0.001
    TEACHINGSTEP            = 0.5
    MAX_EPOCH               = 100

    samples = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ]

    neuron = Perceptron(DIMENSION, Function.THRESHOLD)
    
    mse = 999
    epochs = 0

    while (mse > LEAST_MEAN_SQUARE_ERROR and epochs < MAX_EPOCH):
        mse = 0
        error = 0

        for sample in samples:
            temp = []
            for i in range(DIMENSION):
                temp.append(sample[i])
            neuron.setSample(temp)

            output = neuron.calculateNet()
            error += pow((output - sample[DIMENSION]),2)
            neuron.adjustWeights(TEACHINGSTEP, output, sample[DIMENSION])

        mse = error/len(samples)
        print('The mean square error of %d epoch is %.4f' % (epochs, mse))
        epochs += 1
