#from data_handler import get_data
from models import get_model
import random
from netcal.metrics import ECE
import argparse
import pickle
import string
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import preprocessor as p
from collections import Counter
import os
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
from scipy import stats
import tflearn
import json
from nltk import tokenize as tokenize_nltk
TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
import pandas as pd
import operator
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import random
from models import *

import tensorflow.keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
MAX_FEATURES = 2
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf

from tensorflow.keras import backend as K
#from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, optimizers
from gensim.parsing.preprocessing import STOPWORDS
#from tensorflow.keras.utils import np_utils
from tensorflow.python.keras import utils
import codecs
import gensim, sklearn
from string import punctuation
from collections import defaultdict
import sys
#rom nltk import tokenize as tokenize_nltk
#from my_tokenizer import glove_tokenize
from numpy import savetxt

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

def prod(list, num):
    res=[]
    for i in list:
        res.append(i/num)
    return res

def print_scores(ece, p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS):
    
    print ("None average results are:")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))       
    
    print ("weighted results are")
    print ("average ece is %f" %(ece/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("macro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))  
     
def evaluate_model(model, testX, testY,flag, m):
    if flag=='binary':
        temp = model.predict(testX)
        y_pred = []
        for i in temp:
            if i >0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)  

        y_true = []
        for i in testY:
            if i >0.5:
                y_true.append(1)
            else:
                y_true.append(0) 
#        print('temp')
#        print(temp)
#        print('yTrue')
#        print(y_true)
    elif flag == 'categorical':
        y_true = np.argmax(testY, axis=1)
        y_pred1 = model.predict(testX)
        y_pred = np.argmax(y_pred1, axis=1)
        y_true_c = []
        print('argmax')
        print(y_pred)
        print('no argmax')
        print(y_pred1)
        print('testY')
        print(testY)
    
    else:
        temp = model.predict(testX)
        temp = np.argmax(temp, axis=1)
        y_true = np.argmax(testY, axis=1)
        y_pred=[]
        for i in temp:
            if i == 2:
                y_pred.append(1)
            else:
                y_pred.append(i)
        
    precision = metrics.precision_score(y_true, y_pred, average=None)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    
    precisionw = metrics.precision_score(y_true, y_pred, average='weighted')
    recallw = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_scorew = metrics.f1_score(y_true, y_pred, average='weighted')
    
    precisionm = metrics.precision_score(y_true, y_pred, average='macro')
    recallm = metrics.recall_score(y_true, y_pred, average='macro')
    f1_scorem = metrics.f1_score(y_true, y_pred, average='macro')
    
    logits = np.append(temp, np.zeros([temp.size,1]),axis=1)
    logits[:,1] = 1-logits[:,0]
    logits[:,[0, 1]] = logits[:,[1, 0]]
    if m=="test":
        savetxt('groundTruth_test_1.csv', y_true, delimiter=',')
        savetxt('logits_test_1.csv', logits, delimiter=',')
    if m=="train":
        savetxt('groundTruth_train_1.csv', y_true, delimiter=',')
        savetxt('logits_train_1.csv', logits, delimiter=',')
    if m=="dev":
        savetxt('groundTruth_dev_1.csv', y_true, delimiter=',')
        savetxt('logits_dev_1.csv', logits, delimiter=',')
    ece1 = ECE(bins = 10)
    ece = ece1.measure(X=np.max(logits, axis=1), y=np.array(y_true))
    print('ECE = ', ece)
    
    print("Precision: " + str(precision) + "\n")
    print("Recall: " + str(recall) + "\n")
    print("f1_score: " + str(f1_score) + "\n")
    #print(confusion_matrix(y_true, y_pred))
    
    print(":: Classification Report")
    print(classification_report(y_true, y_pred))
    def save_object(obj, filename):
        with open(filename, 'wb') as fp:
            pickle.dump(obj, fp)

    def load_object(filename):
        with open(filename, 'rb') as fp:
            obj = pickle.load(fp)
        return obj
    save_object(y_pred, 'y_pred_texto.pkl')
    return ece, precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem, model

def load_data(dataset):
    x_text = []
    labels = []
    if dataset =='train':
        print("Loading data from file: " + dataset)
        data = pickle.load(open('drive/My Drive/Colab Notebooks/hate_speech_data/Data/Waseem_Dataset.pkl', 'rb'))
        random.shuffle(data)
        print('len(dataaaaaa)')
        print(len(data))
        for i in range(len(data)):
            x_text.append(data[i]['text'])
            labels.append(data[i]['label'])
    elif dataset == 'test':
        print("Loading data from file: " + dataset)
#        with open('../Data/en_dev.json', 'r', encoding = 'utf-8') as sem_file:
#            data = sem_file.readlines()
        data = pickle.load(open('drive/My Drive/Colab Notebooks/hate_speech_data/Data/en_test.pkl', 'rb'))
        print(data.loc[0,'text'])
        for i in range(len(data)):
            x_text.append(data.loc[i,'text'])
            labels.append(data.loc[i,'HS'])
        data = pickle.load(open('drive/My Drive/Colab Notebooks/hate_speech_data/Data/en_dev.pkl', 'rb'))
        for i in range(len(data)):
            x_text.append(data.loc[i,'text'])
            labels.append(data.loc[i,'HS'])
        #data = pickle.load(open('../Data/SemEval_Dataset.pkl', 'rb'))
    
    elif dataset == 'dev':
            print("Loading data from file: " + dataset)
    #        with open('../Data/en_dev.json', 'r', encoding = 'utf-8') as sem_file:
    #            data = sem_file.readlines()
            data = pickle.load(open('drive/My Drive/Colab Notebooks/hate_speech_data/Data/en_dev.pkl', 'rb'))
            print(data.loc[0,'text'])
            for i in range(len(data)):
                x_text.append(data.loc[i,'text'])
                labels.append(data.loc[i,'HS'])
                
    elif dataset == 'data_new':
        print("Loading data from file: " + dataset)
        data = pickle.load(open('drive/My Drive/Colab Notebooks/hate_speech_data/Data/Data_new.pkl', 'rb'))
        #print(data)
        for i in range(len(data)):
            x_text.append(data[i]['text'])
            labels.append(data[i]['label'])
 
#    for i in range(len(data)):
#        x_text.append(data[i]['text'])
#        labels.append(data[i]['label'])

    from collections import Counter
    print(Counter(labels))
    return x_text, labels

def data_processor(x_text,X_train,y_train,X_test,y_test,flag):
    #x_text = np.concatenate((X_train,X_test),axis = 0)
    post_length = np.array([len(x.split(" ")) for x in x_text])
    max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))
    y_map = {
        'none': 0,
        'racism': 1,
        'sexism': 2,
        'hate':1}
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)

    dict1 = {'sexism':2,'racism':1,'none':0,'hate':1,'none':0,0:0,1:1,2:2}
    #NUM_CLASSES = 1
    y_train = [dict1[b] for b in y_train]
    y_test = [dict1[b] for b in y_test]
    
    
    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))
     
    
    print('trainX[0]')
    print(trainX[0])
    trainY = np.asarray(y_train)
    testY = np.asarray(y_test)

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    print('data_processor')
    print(len(trainY))
    print(len(testY))

    if flag != 'binary':
        trainY = to_categorical(trainY, nb_classes=3)
        testY = to_categorical(testY, nb_classes=3)

    data_dict = {
            "data": "twitter",
            "trainX" : trainX,
            "trainY" : trainY,
            "testX" : testX,
            "testY" : testY,
            "vocab_processor" : vocab_processor
        }

    return data_dict

def get_model(m_type,inp_dim, vocab_size, embed_size, num_classes, learn_rate):
    if m_type == 'cnn':
        model = cnn(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == 'lstm':
        model = lstm_keras(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == "blstm":
        model = blstm(inp_dim, vocab_size, embed_size, num_classes, learn_rate)#(inp_dim)
    elif m_type == "binary_blstm":
        print('blstm_binary')
        model = binary_blstm(inp_dim, vocab_size, embed_size, num_classes, learn_rate)#(inp_dim)
    elif m_type == "blstm_attention":
        model = blstm_atten(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    else:
        print ("ERROR: Please specify a correst model")
        return None
    return model

models = ['cnn', 'lstm', 'blstm', 'blstm_attention']
word_vectors = ["random", "glove" ,"sswe"]
EPOCHS = 10
BATCH_SIZE = 512
MAX_FEATURES = 2
NUM_CLASSES = None
DROPOUT = 0.25
LEARN_RATE = 0.01
HASH_REMOVE = None
output_folder_name = "results/"
    
def train(data_dict, model_type, vector_type,flag, embed_size, dump_embeddings=False):
    print("trainnn")   
    data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)
    
    vocab_size = len(vocab_processor.vocabulary_)
    NUM_CLASSES = 3
    print(NUM_CLASSES)
    print(trainX.shape[1])
    print("Vocabulary Size: {:d}".format(vocab_size))
    vocab = vocab_processor.vocabulary_._mapping
    print(model_type)
    print(vector_type)
    print("Running Model: " + model_type + " with word vector initiliazed with " + vector_type + " word vectors.")
    model = get_model(model_type, trainX.shape[1], vocab_size, embed_size, NUM_CLASSES, LEARN_RATE)

    initial_weights = model.get_weights()
    shuffle_weights(model, initial_weights)
    
    if(vector_type!="random"):
        print("Word vectors used: " + vector_type)
        model.layers[0].set_weights([map_embedding_weights(get_embeddings_dict(vector_type, embed_size), vocab, embed_size)])
        model.fit(trainX, trainY, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE, 
              verbose=1)
    else:
        model.fit(trainX, trainY, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE, 
              verbose=1)
            
    if (dump_embeddings==True):
        if(model_type == 'cnn'):
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
        else:
            embed = model.layers[0].get_weights()[0]
    
        embed_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + ".pkl"
        embed.dump(embed_filename)
        
        vocab_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + "_dict.json"
        reverse_vocab_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + "_reversedict.json"
        
        with open(vocab_filename, 'w') as fp:
            json.dump(vocab_processor.vocabulary_._mapping, fp)
        with open(reverse_vocab_filename, 'w') as fp:
            json.dump(vocab_processor.vocabulary_._reverse_mapping, fp)
    
    
    #print(testX)
    #print(testY)
    #ece2, precision2, recall2, f1_score2,precisionw2, recallw2, f1_scorew2,precisionm2, recallm2, f1_scorem2 =evaluate_model(model, trainX, trainY,flag,"train")
    #ece2, precision2, recall2, f1_score2,precisionw2, recallw2, f1_scorew2,precisionm2, recallm2, f1_scorem2 =evaluate_model(model, testX[3001:,:], testY[3001:],flag,"dev")
    #return  evaluate_model(model, testX[:3001,:], testY[:3001],flag,"test")
    return  evaluate_model(model, testX, testY,flag,"other")

def get_embedding_weights(filename, sep):
    embed_dict = {}
    file = open(filename,'r', encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(sep)
        embed_dict[row[0]] = row[1:]
    print('Loaded from file: ' + str(filename))
    file.close()
    return embed_dict

def map_embedding_weights(embed, vocab, embed_size):
    vocab_size = len(vocab)
    embeddingWeights = np.zeros((vocab_size , embed_size))
    n = 0
    words_missed = []
    for k, v in vocab.items():
        try:
            embeddingWeights[v] = embed[k]
        except:
            n += 1
            words_missed.append(k)
            pass
    print("%d embedding missed"%n, " of " , vocab_size)
    return embeddingWeights

def return_data(data_dict):
    return data_dict["data"], data_dict["trainX"], data_dict["trainY"], data_dict["testX"], data_dict["testY"],data_dict["vocab_processor"]

def get_embeddings_dict(vector_type, emb_dim):
    if vector_type == 'sswe':
        emb_dim==50
        sep = '\t'
        vector_file = 'drive/My Drive/Colab Notebooks/hate_speech_data/Vectors/sswe-u.txt'
    elif vector_type =="glove":
        sep = ' '
        if data == "wiki":
            vector_file = 'word_vectors/glove.6B.' + str(emb_dim) + 'd.txt'
        else:
            vector_file = 'word_vectors/glove.twitter.27B.' + str(emb_dim) + 'd.txt'
    else:
        print ("ERROR: Please specify a correst model or SSWE cannot be loaded with embed size of: " + str(emb_dim) )
        return None
    
    embed = get_embedding_weights(vector_file, sep)
    return embed

def gen_sequence(tweets):
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2,
            'hate':1
#             'both': 1,
#             'abusive': 1,
#             'ofenssive': 1,
#             'normal': 0,
#             'neither': 0,
#             'hateful': 1,
#             'abusive': 1,
#             '1':1,
#             '0':0,
#              1:1,
 #            0:0
            }
    X, y = [], []
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        #print(len(words))
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tweet['label']])
    return X, y

def oversampling(x_text, labels,oversampling_rate):
    y_map = {
        'none': 0,
        'racism': 1,
        'sexism': 2,
        'hate':1,
#         'both': 1,
#         'abusive': 1,
#         'ofenssive': 1,
#         'normal': 0,
#         'neither': 0,
#         'hateful': 1,
#         'abusive': 1,
#         '1':1,
#         '0':0,
#          1:1,
#          0:0,
#          2:2
        }
    dict1 = {'sexism':2,'racism':1,'none':0,'hate':1}
    #NUM_CLASSES = 1
    labels = [dict1[b] for b in labels]
    racism = [i for i in range(len(labels)) if labels[i]==2]
    sexism = [i for i in range(len(labels)) if labels[i]==1]
    x_text = x_text + [x_text[x] for x in racism]*(oversampling_rate-1)+ [x_text[x] for x in sexism]*(oversampling_rate-1)
    labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
    print('oversampling')
    print(len(x_text))
    print(len(labels))
    return x_text, labels

def cv_sorted_data(x_text):
    train_indexes = []
    part0,part1,part2,part3,part4,part5,part6,part7,part8,part9 =[],[],[],[],[],[],[],[],[],[]
    for i in range(len(x_text)):
        if i >=0 and i <700:
            part0.append(i)
        elif i >=700 and i <1400:
            part1.append(i)

        elif i >=1400 and i <2100:   
            part2.append(i)

        elif i >=2100 and i <2800:
            part3.append(i)

        elif i >=2800 and i <3494:
            part4.append(i)

        elif i >=3494 and i <4198:
            part5.append(i)
        elif i >=4198 and i <4897:
            part6.append(i)
        elif i >=4897 and i <5599:
            part7.append(i)
            
        elif i >=5599 and i <6299:
            part8.append(i)
            
        elif i >=6299 and i <7006:
            part9.append(i)
    train_indexes.append(part0) 
    train_indexes.append(part1) 
    train_indexes.append(part2) 
    train_indexes.append(part3) 
    train_indexes.append(part4) 
    train_indexes.append(part5) 
    train_indexes.append(part6) 
    train_indexes.append(part7) 
    train_indexes.append(part8) 
    train_indexes.append(part9) 

    return train_indexes  

def get_data_waseem3(s):
    tweets=[]
    data = pickle.load(open('drive/My Drive/Colab Notebooks/hate_speech_data/Data/Waseem_Dataset.pkl', 'rb'))
    for tweet_full in data:
    #tweet_full = json.loads(line)
        tweets.append({
            'id': tweet_full['id'],
            'name': tweet_full['name'],
            'text': tweet_full['text'].lower(),
            'label': tweet_full['label'],
            })
    Odiosos=[]
    none=0
    dict_users_none={}
    dict_users_sexist={}
    dict_users_racist={}
    strategy = s
    for tweet_full in data:
        #tweet_full = json.loads(line)
        tweets.append({
            #'id': tweet_full['id'],
            'text': tweet_full['text'].lower(),
            'label': tweet_full['label'],
            'name': tweet_full['name']
            })
        if tweet_full['label'] != 'none':
            Odiosos.append(tweet_full['name'])
        if tweet_full['label'] == 'none':
            none+=1
            if tweet_full['name'] in dict_users_none.keys():
                dict_users_none[tweet_full['name']] += 1
            else:
                dict_users_none[tweet_full['name']] = 1 
                
        if tweet_full['label'] == 'sexism':
            none+=1
            if tweet_full['name'] in dict_users_sexist.keys():
                dict_users_sexist[tweet_full['name']] += 1
            else:
                dict_users_sexist[tweet_full['name']] = 1 
                
        if tweet_full['label'] == 'racism':
            none+=1
            if tweet_full['name'] in dict_users_racist.keys():
                dict_users_racist[tweet_full['name']] += 1
            else:
                dict_users_racist[tweet_full['name']] = 1 
                
    resultado = sorted(dict_users_none.items(), key=operator.itemgetter(1))
    resultado.reverse()
    
    users_train = []
    for i in resultado[:1400]:
        if i[0] not in Odiosos:
            users_train.append(i[0])


    None_users = sorted(dict_users_none.items(), key=operator.itemgetter(1))
    None_users.reverse()
    Sexist_users = sorted(dict_users_sexist.items(), key=operator.itemgetter(1))
    Sexist_users.reverse()
    Racist_users = sorted(dict_users_racist.items(), key=operator.itemgetter(1))
    Racist_users.reverse()         
    if strategy == 1:
        print('strategy')
        print(strategy)
        t = [Sexist_users[0][0],Racist_users[0][0],Sexist_users[1][0]]
        for i in t:
            users_train.append(i)
            
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(1))
        resultado.reverse()


        resultado = sorted(dict_users_racist.items(), key=operator.itemgetter(1))
        resultado.reverse()


    elif strategy == 2:
        print('strategy')
        print(strategy)
        for i in resultado:
            if i[0] not in Odiosos:
                users_train.append(i[0])

        t = [Racist_users[0][0]]
        for i in t:
            users_train.append(i)
            
        resultado = sorted(dict_users_none.items(), key=operator.itemgetter(0))
        resultado.reverse()
    
        count =0
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado:
            if i[0] != Sexist_users[1][0] and i[0] not in  dict_users_racist.keys() and (count < 160):
                users_train.append(i[0]) 
            count += 1
        for i in sorted(dict_users_sexist.items(), key=operator.itemgetter(1)):
            if i[0] not in dict_users_none.keys():
                users_train.append(i) 
                count += 1
    
    elif strategy == 3:
        print('strategy')
        print(strategy)

        for i in resultado:
             if i[0] not in Odiosos:
                users_train.append(i[0])

        t = [Racist_users[0][0]]
        for i in t:
            users_train.append(i)

        count =0
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado:
            if i[0] != Sexist_users[0][0] and i[0] not in  dict_users_racist.keys():
                users_train.append(i[0]) 
            count += 1
    
    return tweets,users_train
def get_data_dict(data,X_train,X_test,y_train,y_test,flag):
    x_text = np.concatenate((X_train,X_test),axis = 0)
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))
    NUM_CLASSES = 3
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)

    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))

    trainY = np.asarray(y_train)
    testY = np.asarray(y_test)

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    
    if flag != 'binary':
        trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
        testY = to_categorical(testY, nb_classes=NUM_CLASSES)

    data_dict = {
            "data": data,
            "trainX" : trainX,
            "trainY" : trainY,
            "testX" : testX,
            "testY" : testY,
            "vocab_processor" : vocab_processor
        }
    return data_dict
    data = pickle.load(open(get_filename("twitter"), 'rb'))
    x_text=[]
    labels=[]
    x_text_train = []
    labels_train = [] 
    x_text_test = []
    labels_test = [] 
    
    dict1 = {'racism':2,'sexism':1,'none':0}
    for i in data:
        labels.append(i['label'])
        x_text.append(i['text'])       

        
    labels = [dict1[b] for b in labels]

    print("Counter before oversampling ALL")
    from collections import Counter
    print(Counter(labels))
    
    racism = [i for i in range(len(labels)) if labels[i]==2]
    sexism = [i for i in range(len(labels)) if labels[i]==1]
    
    print(len(racism))
    print(len(sexism))
    
    data = data + [data[x] for x in racism]*(oversampling_rate-1)+ [data[x] for x in sexism]*(oversampling_rate-1)
    labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
     
    print("Counter AFTER oversampling ALL")
    from collections import Counter
    print(Counter(labels))
    
    data_dicts=[]
    for test_users in particion1:
        for i in range(len(data)):
            if data[i]['user'] not in test_users:
                x_text_train.append(data[i]['text'])
                labels_train.append(labels[i])
            else:
                x_text_test.append(data[i]['text'])
                labels_test.append(labels[i])
        
        NUM_CLASSES = 3
        print("Counter TRAIN")
        from collections import Counter
        print(Counter(labels_train))

        print("Counter test")
        from collections import Counter
        print(Counter(labels_test))

        post_length = np.array([len(x.split(" ")) for x in x_text])
        if(data != "twitter"):
            max_document_length = int(np.percentile(post_length, 95))
        else:
            max_document_length = max(post_length)
        print("Document length : " + str(max_document_length))

        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
        vocab_processor = vocab_processor.fit(x_text)

        trainX = np.array(list(vocab_processor.transform(x_text_train)))
        testX = np.array(list(vocab_processor.transform(x_text_test)))

        trainY = np.asarray(labels_train)
        testY = np.asarray(labels_test)

        trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
        testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

        trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
        testY = to_categorical(testY, nb_classes=NUM_CLASSES)

        data_dict = {
            "data": data,
            "trainX" : trainX,
            "trainY" : trainY,
            "testX" : testX,
            "testY" : testY,
            "vocab_processor" : vocab_processor
        }
        data_dicts.append(data_dict)
    return data_dicts

def Holdout_partition(oversampling_rate,strategy,flag): #seleccionando para training los usuarios mas prol??feros
    data ,train_users = get_data_waseem3(strategy)
    print('len(train_users)')

    print(len(train_users))
    train_index = []
    x_text=[]
    labels=[]
    x_text_train = []
    labels_train = [] 
    x_text_test = []
    labels_test = [] 
    test_index = []
    clasestrain = {}
    clasestest ={}
    none = 0
    for i in range(len(data)):

        if data[i]['name'] in train_users:
            x_text_train.append(data[i]['text'])
            labels_train.append(data[i]['label'])
        else:
            x_text_test.append(data[i]['text'])
            labels_test.append(data[i]['label'])
            
    print(len(x_text_train))        
    if flag == 'binary':
        dict1 = {'sexism':1,'racism':1,'none':0}
    else:
        dict1 = {'sexism':2,'racism':1,'none':0}
    #NUM_CLASSES = 1
    labels_train = [dict1[b] for b in labels_train]
    print("Counter TRAIN")
    from collections import Counter
    print(Counter(labels_train))
    
    labels_test = [dict1[b] for b in labels_test]
    print("Counter test")
    from collections import Counter
    print(Counter(labels_test))
    
    
    racism = [i for i in range(len(labels_train)) if labels_train[i]==1]
    sexism = [i for i in range(len(labels_train)) if labels_train[i]==2]
    
    print(len(racism))
    #print(len(sexism))
    oversampling_rate = 3

    x_text_train = x_text_train + [x_text_train[x] for x in racism]*(oversampling_rate-1)+ [x_text_train[x] for x in sexism]*(oversampling_rate-1)
    
    labels_train = labels_train + [1 for i in range(len(racism))]*(oversampling_rate-1) + [2 for i in range(len(sexism))]*(oversampling_rate-1)
   

    print("Counter TRAIN afterrr")
    from collections import Counter
    print(Counter(labels_train))
          
    post_length = np.array([len(x.split(" ")) for x in x_text_train])
    
    max_document_length = int(np.percentile(post_length, 95))
    print("Document length : " + str(max_document_length))
    
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text_train)

    trainX = np.array(list(vocab_processor.transform(x_text_train)))
    testX = np.array(list(vocab_processor.transform(x_text_test)))
    
    trainY = np.asarray(labels_train)
    testY = np.asarray(labels_test)
        
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    if flag != 'binary':
        trainY = to_categorical(trainY, nb_classes=3)
        testY = to_categorical(testY, nb_classes=3)

    data_dict = {
        "data": data,
        "trainX" : trainX,
        "trainY" : trainY,
        "testX" : testX,
        "testY" : testY,
        "vocab_processor" : vocab_processor
    }
    print('testYyyyyyyyyyyyy')
    print(len(trainX))
    print(len(trainY))
    print(len(testX))
    print(len(testY))
    return data_dict
