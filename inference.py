# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:06:18 2016

@author: user
"""

import enum
import numpy as np
import matplotlib.pyplot as plt
import perceptron


class Label(enum.IntEnum):
    RED     = 1
    GREEN   = 2
    BLUE    = 3


def scaling(x):
    
    # mini-max scaling
    return (x - 0)/(255 - 0)


def rgb2xyy(rgb):
    
    M = np.matrix([[ 0.412453, 0.357580, 0.180423],
                   [ 0.212671, 0.715160, 0.072169],
                   [ 0.019334, 0.119193, 0.950227]])

    XYZ = np.dot(M,rgb)
    X = XYZ[0,0]
    Y = XYZ[0,1]
    Z = XYZ[0,2]
    
    x = X/(X + Y + Z)
    y = Y/(X + Y + Z)
    #z = 1 - x - y
    xyY = np.array([x,y,Y])
    
    return xyY


if __name__ == '__main__':
    
    DIMENSION = 3
    WEIGHTS_DATA = './weights/weights_10000_1.txt'
    SAMPLES_DATA = './samples/samples_100000.txt'
    
    weights = np.empty((0,DIMENSION))
    biases = []
    fp = open(WEIGHTS_DATA, 'r')
    lines = fp.readlines()
    lines = lines[1:]   # throw top line
    for line in lines:
        line.replace('\n', '')
        temp = line.split(',')
        weights = np.append(weights, np.array([[float(temp[0]),float(temp[1]),float(temp[2])]]), axis=0)
        biases.append(float(temp[3]))
    fp.close()
        
    samples = np.empty((0,DIMENSION+1))
    fp = open(SAMPLES_DATA, 'r')
    lines = fp.readlines()
    for line in lines:
        line.replace('\n', '')
        temp = line.split(',')
        samples = np.append(samples, np.array([[int(temp[0]),int(temp[1]),int(temp[2]),int(temp[3])]]), axis=0)
    fp.close()
    
    #print(weights)
    #print(biases)
    #print(samples)
    
    rNeuron = perceptron.Perceptron(DIMENSION, perceptron.Function.SIGMOID)
    gNeuron = perceptron.Perceptron(DIMENSION, perceptron.Function.SIGMOID)
    bNeuron = perceptron.Perceptron(DIMENSION, perceptron.Function.SIGMOID)    
    rNeuron.setWeights(weights[0])
    gNeuron.setWeights(weights[1])
    bNeuron.setWeights(weights[2])
    rNeuron.setBias(biases[0])
    gNeuron.setBias(biases[1])
    bNeuron.setBias(biases[2])
    
    results = []
    for sample in samples:
        rOutput = rNeuron.recall( scaling(sample[0:DIMENSION]) )
        gOutput = gNeuron.recall( scaling(sample[0:DIMENSION]) )
        bOutput = bNeuron.recall( scaling(sample[0:DIMENSION]) )
        
        # show results of inference
        if rOutput >= gOutput and rOutput >= bOutput:
            color = 'r'
            results.append(Label.RED)
        elif gOutput >= rOutput and gOutput >= bOutput:
            color = 'g'
            results.append(Label.GREEN)
        elif bOutput >= rOutput and bOutput >= gOutput:
            color = 'b'
            results.append(Label.BLUE)
        # show ground truth
        '''
        if sample[DIMENSION] == 1:
            color = 'r'
        elif sample[DIMENSION] == 2:
            color = 'g'
        elif sample[DIMENSION] == 3:
            color = 'b'
        '''
        
        xyY = rgb2xyy( scaling(sample[0:DIMENSION]) )
        plt.scatter(xyY[0], xyY[1], c=color)
        
    plt.title('scatterplot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0.0,0.7])
    plt.ylim([0.0,0.7])

    plt.show()
    
    
    conf_mat = [[0,0,0],[0,0,0],[0,0,0]]    # (gt, est): [[(r,r),(r,g),(r,b)],[(g,r),(g,g),(g,b)],[(b,r),(b,g),(b,b)]]
    for i in range(len(samples)):
        
        ground_truth = samples[i][DIMENSION]
        estimated = results[i]
        if ground_truth == Label.RED:
            if estimated == Label.RED:
                conf_mat[0][0] += 1
            elif estimated == Label.GREEN:
                conf_mat[0][1] += 1
            elif estimated == Label.BLUE:
                conf_mat[0][2] += 1
            else:
                pass
        elif ground_truth == Label.GREEN:
            if estimated == Label.RED:
                conf_mat[1][0] += 1
            elif estimated == Label.GREEN:
                conf_mat[1][1] += 1
            elif estimated == Label.BLUE:
                conf_mat[1][2] += 1
            else:
                pass
        elif ground_truth == Label.BLUE:
            if estimated == Label.RED:
                conf_mat[2][0] += 1
            elif estimated == Label.GREEN:
                conf_mat[2][1] += 1
            elif estimated == Label.BLUE:
                conf_mat[2][2] += 1
            else:
                pass
        else:
            pass

    # show confusion matrix
    print('%-8s|%8s|%8s|%8s|' % ('','RED','GREEN','BLUE'))
    print('%-8s|%8d|%8d|%8d|' % ('RED  ', conf_mat[0][0], conf_mat[0][1], conf_mat[0][2]))
    print('%-8s|%8d|%8d|%8d|' % ('GREEN', conf_mat[1][0], conf_mat[1][1], conf_mat[1][2]))
    print('%-8s|%8d|%8d|%8d|' % ('BLUE ', conf_mat[2][0], conf_mat[2][1], conf_mat[2][2]))
