import argparse
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import os
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
#from tensorflow.contrib import learn
import tflearn
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from auxiliares import *
import random


#Réplica del experimento original descrito en la literatura
def run_model_exp1(oversampling_rate, vector_type, embed_size,flag):    
    if flag == 'binary':
        model_type = "binary_blstm"
    else:
        model_type = "blstm"
    
    x_text, labels = load_data('train')
    # c = list(zip(x_text, labels))
    # random.shuffle(c)
    # x_text, labels = zip(*c)

    x1 = x_text[:7334]
    l1 = labels[:7334]
    x2 = x_text[7334:]
    l2 = labels[7334:]
    x_text, labels = x1, l1

    #Oversampling before cross-validation
    x_text, labels = oversampling(x_text, labels,oversampling_rate)
    
    #cross-validation with oversampled data
    cv_object = KFold(n_splits=5, shuffle=True, random_state=42)
    e = 0.
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    NO_OF_FOLDS = 5

    for train_index, test_index in cv_object.split(x_text):
        X_train, y_train, X_test, y_test = [],[],[],[]
        for i in range(len(x_text)):
            if i in train_index:
                X_train.append(x_text[i])
                y_train.append(labels[i])
            else:
                X_test.append(x_text[i])
                y_test.append(labels[i])
        
               
        data_dict = data_processor(x_text,X_train,y_train,X_test,y_test,flag)
        ece, precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem, model = train(data_dict, model_type, vector_type,flag, embed_size)
        e += ece
        p += precisionw
        p1 += precisionm
        r += recallw
        r1 += recallm
        f1 += f1_scorew
        f11 += f1_scorem
        pn += precision
        rn += recall
        fn += f1_score
    print_scores(e, p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS)
    
    #X_test1, y_test1 = load_data('test')
    data_dict = data_processor(x_text,x1,l1,x2,l2,flag)
    testX1 = data_dict['testX']
    testY1 = data_dict['testY']
    trainX1 = data_dict['trainX']
    trainY1 = data_dict['trainY']
    evaluate_model(model, trainX1, trainY1,flag, "train")
    evaluate_model(model, testX1[:967,:], testY1[:967],flag, 'test')
    evaluate_model(model, testX1[967:,:], testY1[967:],flag, 'dev')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 1. Original Experiment Replica')
    parser.add_argument('--type', choices=['binary', 'categorical'], default = 'binary')
    vector_type = "sswe"
    oversampling_rate = 3
    flag=parser.parse_args().type
    NO_OF_FOLDS = 5
    embed_size = 50
    run_model_exp1(oversampling_rate,  vector_type, embed_size,flag)
