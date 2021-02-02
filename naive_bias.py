from math import sqrt
from math import pi
from math import exp
import numpy as np
import pandas as pd 
def class_separation(data_1):
    sep=dict()
    for i in range(len(data_1)):
        vector=data_1[i]
        class_v=vector[-1]
        if (class_v not in sep):
            sep[class_v]=list()
        sep[class_v].append(vector)
    return sep
def mean(numbers): 
    return sum(numbers) / float(len(numbers)) 
  

def std_dev(numbers): 
    avg = mean(numbers) 
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1) 
    return sqrt(variance) 
def summarize(dataset):
    summaries=[(mean(column),std_dev(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
def summarize_cls(data_2):
    sep_1=class_separation(data_2)
    sum_1=dict()
    for class_value,rows in sep_1.items():
        sum_1[class_value]=summarize(rows)
    return sum_1
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent
def calculateClassprobablities(info,test):
    probablities={}
    for classvalue,class_des in info.items():
        probablities[classvalue]=1
        for i in range(len(class_des)):
            mean,std_dev=class_des[i]
            x=test[i]
            probablities[classvalue]*=calculate_probability(x, mean, stdev)
    return probablities
def predict(info,test):
    probablities=calculateClassprobablities(info,test)
    bestlabel,bestprob=None,-1
    for classvalue,probablity in probablities.items():
        if bestlabel is None or probablity>bestprob:
            bestprob=probablity
            bestlabel=classvalue
    return bestlabel
def get_pridictions(info,test):
    predictions=[]
    for i in range(len(test)):
        result=predict(info,test[i])
        predictions.append(result)
    return predictions
def accuracy_rate(test,predictions):
    correct=0
    for i in range(len(test)):
        if test[i][-1]==predictions[i]:
            correct+=1
    return (correct/float(len(test)))
 
               
        






