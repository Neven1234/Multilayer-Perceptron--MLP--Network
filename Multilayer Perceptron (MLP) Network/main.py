# This is a sample Python script.
import random

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import GUI as g
from math import *
from random import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def Replace_gender(dataset):
    dataset = pd.DataFrame(dataset)
    dataset['gender'] = np.select(
        [dataset['gender'].eq('female'), dataset['gender'].eq('male')], [0, 1], default=2
    )
    return dataset;


def Scaling(dataset, col):
    for i in range(len(col)):
        dataset[col[i]] = (dataset[col[i]] - dataset[col[i]].min()) / (dataset[col[i]].max() - dataset[col[i]].min())
    return dataset;


def Spliting(dataset):
    shuffle = np.random.permutation(len(dataset))
    test_size = int(len(dataset) * 0.4)
    test_aux = shuffle[:test_size]
    train_aux = shuffle[test_size:]
    TRAIN_DF = dataset.iloc[train_aux]
    TEST_DF = dataset.iloc[test_aux]
    return TRAIN_DF, TEST_DF

    ####


# read the dataset
ALLData = pd.read_csv('penguins.csv')
ALLData = Replace_gender(ALLData)
ALLData['species'] = np.select(
    [ALLData['species'].eq('Adelie'), ALLData['species'].eq('Gentoo'), ALLData['species'].eq('Chinstrap')],
    ['100', '010', '001'],
)

s = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
ALLData = Scaling(ALLData, s)
print(ALLData)
ALLData_train, ALLData_test = Spliting(ALLData)
print(ALLData_train)
print(ALLData_test)
ALLData_train = ALLData_train.sample(frac=1).reset_index()
ALLData_test = ALLData_test.sample(frac=1).reset_index()

# visualization
X_train = ALLData_train.iloc[:, 2:]
X_test = ALLData_test.iloc[:, 2:]
Y_train = ALLData_train.iloc[:, :2]
Y_test = ALLData_test.iloc[:, :2]
print("one sample", Y_train['species'][0][0])
print("one sample", Y_train['species'][0][1])
print("one sample", Y_train['species'][0][2])

####

# spliting to train and test


###############################


############################################################
print("learningrate_epochs", g.inputs)
nerun_each_layer = []
learning_rate = g.inputs[0]
epoch = g.inputs[1]
layers = g.inputs[2]
funcType = g.function
nerun_each_layer = g.inputs[3].split(',')
print('learning rate:', learning_rate)
print('epoch:', epoch)
print('layers:', layers)
print('neuran in each level:', nerun_each_layer)


######################################### implimintation

###############calculate weights

def ActivationFunction(net):
    for Type in range(len(g.function)):
        if (g.function[Type] == 'Sigmoid function'):
            f = 1.0 / (1.0 + np.exp(-net))
            break
        elif (g.function[Type] == 'Hyperbolic Tangent'):
            tempo = float(1.0 - np.exp(-net))
            tempo2 = float(1.0 + np.exp(-net))
            f = float(tempo / tempo2)
            break
        else:
            continue
    return f


# def Hyperbolic(net):
#     return 1-exp(-net)/1+exp(-net)
n = list()


def initialize_weights(inputs, hidden):
    hidden_layer = [{random() for i in range(inputs)} for i in range(hidden)]
    n.append(hidden_layer)
    # output_layer = [{random() for i in range(hidden )} for i in range(outputs)]
    # n.append(output_layer)
    return n


if (g.flages[0] == 0):
    for i in range(int(layers)):
        if (int(layers) == 1):
            weights = initialize_weights(5, int(nerun_each_layer[i]))
            weights = initialize_weights(int(nerun_each_layer[i]), int(3))
            break

        if (i == int(layers) - 1):
            weights = initialize_weights(int(nerun_each_layer[i]), int(3))
            break;
        if (i == 0):
            weights = initialize_weights(5, int(nerun_each_layer[i]))
            weights = initialize_weights(int(nerun_each_layer[i]), int(nerun_each_layer[i + 1]))


        else:
            weights = initialize_weights(int(nerun_each_layer[i]), int(nerun_each_layer[i + 1]))
else:
    for i in range(int(layers)):
        if (int(layers) == 1):
            weights = initialize_weights(6, int(nerun_each_layer[i]))
            weights = initialize_weights(int(nerun_each_layer[i]) + 1, int(3))
        if (i == int(layers) - 1):
            weights = initialize_weights(int(nerun_each_layer[i]) + 1, int(3))
            break;
        if (i == 0):
            weights = initialize_weights(6, int(nerun_each_layer[i]))
            weights = initialize_weights(int(nerun_each_layer[i]) + 1, int(nerun_each_layer[i + 1]))

        else:
            weights = initialize_weights(int(nerun_each_layer[i]) + 1, int(nerun_each_layer[i + 1]))

print("weights", weights)

####end calculate weights##############################################

################################net input and f (forward)
step = 0
step2 = 0
# X_train = pd.DataFrame(X_train)
net = 0
functions = []
row = 0
row2 = 0
error = 0
nn = 0
m = 0
v = 0
errors = []
AccuricyCount = 0
MaxAccuricytrain = 0
for k in range(int(epoch)):
    for fn in range(len(X_train)):  # len of x train3
        row = X_train.iloc[fn]
        errors = []
        for layer, i in zip(weights, range(len(weights))):
            new_inputs_out = []

            for neuron in layer:
                if (g.flages[0] == 1):
                    lenth = len(row) + 1
                else:
                    lenth = len(row)
                for num, j in zip(neuron, range(lenth)):
                    if (g.flages[0] == 1 and j == lenth - 1):
                        net += num * 1
                    else:
                        net += num * row[j]

                f = ActivationFunction(net)
                functions.append(f)
                new_inputs_out.append(f)
                net = 0
            row = new_inputs_out

        # functions = functions[::-1]
        # print("array",functions)
        fun = np.flip(functions)
        ######accuricy
        ActualOut = functions[-3:]
        # print('ac:', ActualOut)

        maxmumm = max(ActualOut)
        for i in range(len(ActualOut)):
            if (ActualOut[i] == maxmumm):
                ActualOut[i] = 1
            else:
                ActualOut[i] = 0
        # print('ac2:', ActualOut)
        d = [int(x) for x in Y_train['species'][fn]]
        # print('d', d)
        for i in range(len(d)):
            if (d[i] == ActualOut[i] and ActualOut[i] == 1):
                AccuricyCount += 1
        # print('Acuuricy train:', (AccuricyCount ))

        for layer, i in zip(reversed(weights), reversed(range(len(weights)))):

            if i == 0:
                break
            if i == len(weights) - 1:
                h = len(weights)
                for nn in range(len(layer)):
                    # for num, j in zip(neuron, range(len(neuron))):
                    error = (int(Y_train['species'][fn][h - 1]) - fun[nn]) * fun[nn] * (1 - fun[nn])
                    errors.append(error)
                    # errors.append(num * functions[len(functions) - len(neuron)] * ( 1 - functions[len(functions) - len(neuron)]) * errors[j - 1])
                    # step2=step2+len(weights[f])-1
                    h -= 1
                h = 0
            # print("neuron",neuron)
            neuron = layer[i]
            for c in reversed(range(len(neuron))):
                if (g.flages[0] == 1):
                    if (c == len(neuron) - 1):
                        continue
                for neuron in reversed(layer):
                    new = []
                    for num, w in zip(neuron, range(len(neuron))):
                        new.append(fun[(nn + 1)])
                        error += list(neuron)[c] * errors[m]
                        m = m + 1
                        break
                error = error * fun[(nn + 1)] * (1 - fun[(nn + 1)])
                m = m - len(layer)
                errors.append(error)
                row2 = new
                nn += 1
                v = v + 1

            # nn += int(nerun_each_layer[c])-1
            m = m + len(layer)

        ########## updat weights
        # print('mirvat error:',errors)
        # print('beforrr',weights)
        num_erreor = 0
        errors = np.flip(errors)
        num_fun = 3
        first_row = 0
        row3 = functions[0:len(fun) - 3]
        o = 0
        O = 0
        updatWieghts = list()
        for a in range(len(weights)):
            if (a == 0):
                first_row = X_train.iloc[fn]
                lenth = 0
                arrNewWeithts = []
                for b in range(len(weights[a])):
                    tempArr = []
                    if (g.flages[0] == 1):
                        lenth = len(first_row) + 1
                    else:
                        lenth = len(first_row)
                    for c, e in zip(range(len(weights[a][b])), range(lenth)):
                        if (g.flages[0] == 1 and c == len(weights[a][b]) - 1):
                            temp1 = float(learning_rate) * errors[num_erreor] * 1
                            temp2 = temp1 + list(weights[a][b])[c]
                            tempArr.append(temp2)
                        else:
                            temp1 = float(learning_rate) * errors[num_erreor] * first_row[e]
                            temp2 = temp1 + list(weights[a][b])[c]
                            tempArr.append(temp2)
                    arrNewWeithts.append(tempArr)

                    num_erreor += 1
            else:
                new_inputs_out_ = []
                arrNewWeithts = []

                for b in range(len(weights[a])):
                    tempArr = []
                    for c in range(len(weights[a][b])):
                        if (g.flages[0] == 1 and c == len(weights[a][b]) - 1):
                            temp = float(learning_rate) * errors[num_erreor] * 1
                            temp2 = temp + (list(weights[a][b])[c])
                            tempArr.append(temp2)

                        else:
                            temp = float(learning_rate) * errors[num_erreor] * row3[o]
                            temp2 = temp + (list(weights[a][b])[c])
                            tempArr.append(temp2)
                            new_inputs_out_.append(row3[o])
                            o += 1
                        # num_fun+=1
                    arrNewWeithts.append(tempArr)
                    row3 = new_inputs_out_
                    # num_fun=3
                    num_erreor += 1
                    o = 0
                    O = len(weights[a][b])
            updatWieghts.append(arrNewWeithts)
            o += O
            row3 = functions[o:len(functions)]
            o = 0
        functions = []
        errors = []
        nn = 0
        m = 0
        v = 0
        o = 0
        weights = updatWieghts
        first_row = 0
        num_erreor = 0
        net = 0
        ActualOut = 0
        O = 0
        # print('update', weights)
    print("epoche", k)
    print('Acc of epoch:', (AccuricyCount / 90) * 100)
    if (((AccuricyCount / 90) * 100) > MaxAccuricytrain):
        MaxAccuricytrain = (AccuricyCount / 90) * 100
    # weights = updatWieghts
    row3 = 0
    O = 0
    arrNewWeithts = []
    tempArr = []
    temp1 = 0
    temp2 = 0
    temp = 0
    lenth = 0
    net = 0
    row = 0
    row2 = 0
    error = 0
    functions = []
    errors = []
    nn = 0
    m = 0
    v = 0
    o = 0
    first_row = 0
    num_erreor = 0
    AccuricyCount = 0
    net = 0

    # step2 = step2 + len(weights[f]) - 1
print('best traning accuricy:',MaxAccuricytrain)
############### testing
print('testing:##########')
AccuricyCountTest = 0
acutalout = []
for q in range(len(X_test)):
    functions2 = []
    row9 = X_test.iloc[q]
    for layer, i in zip(weights, range(len(weights))):
        new_inputs_out__ = []

        for neuron in layer:
            if (g.flages[0] == 1):
                lenth = len(row9) + 1
            else:
                lenth = len(row9)
            for num, j in zip(neuron, range(lenth)):
                if (g.flages[0] == 1 and j == lenth - 1):
                    net += num * 1
                else:
                    net += num * row9[j]

            f = ActivationFunction(net)
            functions2.append(f)
            new_inputs_out__.append(f)
            net = 0
        row9 = new_inputs_out__
    ######accuricy
    des = []
    ActualOut = functions2[-3:]
    # print('ac:', ActualOut)
    pre = []
    desired = []
    maxmumm = max(ActualOut)
    for i in range(len(ActualOut)):
        if (ActualOut[i] == maxmumm):
            ActualOut[i] = 1
        else:
            ActualOut[i] = 0
    acutalout.append(ActualOut)
    # print('ac2:', ActualOut)
    d = [int(x) for x in Y_test['species'][q]]
    # print('d', d)
    for i in range(len(d)):
        if (d[i] == ActualOut[i] and ActualOut[i] == 1):
            AccuricyCountTest += 1
    functions2 = []

actualstring = []

for i in range(len(acutalout)):
    actualstring.append(''.join([str(elem) for elem in acutalout[i]]))

c = confusion_matrix(Y_test['species'], actualstring)
print('Acuuricy test:', (AccuricyCountTest / 60) * 100)
print('matrix', c)

# print("functions",functions)
# print("errors", errors)