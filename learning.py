# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:35:55 2016

@author: user
"""

import math
import enum
import time
import numpy as np
import perceptron


class Label(enum.IntEnum):
    RED     = 1
    GREEN   = 2
    BLUE    = 3


def scaling(x):
    
    # mini-max scaling
    return (x - 0)/(255 - 0)


if __name__ == '__main__':
    
    # timer start
    start = time.time()


    TRAINING_DATA   = './samples/samples_10000.txt'
    DIMENSION       = 3
    MSE_THRESH      = 0.01
    LEARNING_RATE   = 0.05
    MAX_EPOCH       = 1000

    samples = np.empty((0,DIMENSION+1))
    fp = open(TRAINING_DATA, 'r')
    lines = fp.readlines()
    for line in lines:
        line.replace('\n', '')
        temp = line.split(',')
        samples = np.append(samples, np.array([[int(temp[0]),int(temp[1]),int(temp[2]),int(temp[3])]]), axis=0)
    fp.close()
        
    #print(samples)
    
    rNeuron = perceptron.Perceptron(DIMENSION, perceptron.Function.SIGMOID)
    gNeuron = perceptron.Perceptron(DIMENSION, perceptron.Function.SIGMOID)
    bNeuron = perceptron.Perceptron(DIMENSION, perceptron.Function.SIGMOID)
    
    mse = 999   # mean square error
    epochs = 0

    while (mse > MSE_THRESH and epochs < MAX_EPOCH):
        mse = 0
        error = 0

        for sample in samples:
            rNeuron.setSample( scaling(sample[0:DIMENSION]) )
            gNeuron.setSample( scaling(sample[0:DIMENSION]) )
            bNeuron.setSample( scaling(sample[0:DIMENSION]) )

            rOutput = rNeuron.calculateNet()
            gOutput = gNeuron.calculateNet()
            bOutput = bNeuron.calculateNet()
            
            rTarget, gTarget, bTarget = 0, 0, 0
            if sample[DIMENSION] == Label.RED:
                rTarget = 1
            elif sample[DIMENSION] == Label.GREEN:
                gTarget = 1
            elif sample[DIMENSION] == Label.BLUE:
                bTarget = 1
            else:
                pass
            rError = math.fabs(rTarget - rOutput)
            gError = math.fabs(gTarget - gOutput)
            bError = math.fabs(bTarget - bOutput)
            error += rError**2 + gError**2 + bError**2
            
            rNeuron.adjustWeights(LEARNING_RATE, rOutput, rTarget)
            gNeuron.adjustWeights(LEARNING_RATE, gOutput, gTarget)
            bNeuron.adjustWeights(LEARNING_RATE, bOutput, bTarget)

        mse = error/(len(samples)*DIMENSION)
        print('The mean square error of %d epoch is %.4f' % (epochs, mse))
        epochs += 1
        
        # timer stlip
        elapsed_time = time.time() - start
        
        fname = './weights/weights_' + str(len(samples)) + '.txt'
        fp = open(fname, 'w')
        out_str = TRAINING_DATA + ',' + str(DIMENSION) + ',' + str(len(samples)) + ',' + str(epochs) + ',' + str(mse) + ',' + '{0}'.format(elapsed_time) + '\n'
        out_str += str(rNeuron.weightsVector[0]) + ',' + str(rNeuron.weightsVector[1]) + ',' + str(rNeuron.weightsVector[2]) + ',' + str(rNeuron.bias) + '\n'
        out_str += str(gNeuron.weightsVector[0]) + ',' + str(gNeuron.weightsVector[1]) + ',' + str(gNeuron.weightsVector[2]) + ',' + str(gNeuron.bias) + '\n'
        out_str += str(bNeuron.weightsVector[0]) + ',' + str(bNeuron.weightsVector[1]) + ',' + str(bNeuron.weightsVector[2]) + ',' + str(bNeuron.bias) + '\n'
        fp.write(out_str)
        fp.close()

    # timer stop
    elapsed_time = time.time() - start
    
    fname = './weights/weights_' + str(len(samples)) + '.txt'
    fp = open(fname, 'w')
    out_str = TRAINING_DATA + ',' + str(DIMENSION) + ',' + str(len(samples)) + ',' + str(epochs) + ',' + str(mse) + ',' + '{0}'.format(elapsed_time) + '\n'
    out_str += str(rNeuron.weightsVector[0]) + ',' + str(rNeuron.weightsVector[1]) + ',' + str(rNeuron.weightsVector[2]) + ',' + str(rNeuron.bias) + '\n'
    out_str += str(gNeuron.weightsVector[0]) + ',' + str(gNeuron.weightsVector[1]) + ',' + str(gNeuron.weightsVector[2]) + ',' + str(gNeuron.bias) + '\n'
    out_str += str(bNeuron.weightsVector[0]) + ',' + str(bNeuron.weightsVector[1]) + ',' + str(bNeuron.weightsVector[2]) + ',' + str(bNeuron.bias) + '\n'
    fp.write(out_str)
    fp.close()
