import argparse
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import pdb
from auxiliares import *
from models import *
from numpy import savetxt
import random

def run_model_exp1():        
    #Experimento 1 Cross-Validation
    tweets,_ = select_tweets('waseem',None)
    random.shuffle(tweets)
    tweets_test = tweets[7734:]
    tweets = tweets[:7734]

    print(tweets)
    print(len(tweets))
    print(tweets_test)
    print(len(tweets_test))

    vocab = gen_vocab(tweets)
    
    MAX_SEQUENCE_LENGTH = max_len(tweets)
    
    train_LSTM_variante1(tweets,tweets_test,vocab, MAX_SEQUENCE_LENGTH)

def train_LSTM_variante1(tweets, tweets_test, vocab, MAX_SEQUENCE_LENGTH):
    #Step 1: Training the embeddings with all data
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.

    X, y = gen_sequence(tweets, vocab, 'binary' )
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    X, y = sklearn.utils.shuffle(X, y)
    y = np.array(y)
    
    y_train = y.reshape((len(y), 1))
    X_temp = np.hstack((X, y_train))

    X_test, y_test = gen_sequence(tweets_test,vocab,'binary')
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    savetxt('ytest.csv', y_test, delimiter=',')
    savetxt('ytrain.csv', y, delimiter=',')
   
    model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,vocab)
    
    #initializing with random embeddings
    shuffle_weights(model)
    
    for epoch in range(EPOCHS):
        for X_batch in batch_gen(X_temp, BATCH_SIZE):
            x = X_batch[:, :MAX_SEQUENCE_LENGTH]
            y_temp = X_batch[:, MAX_SEQUENCE_LENGTH]
            try:
                y_temp = to_categorical(y_temp, nb_classes=3)
            except Exception as e:
                print (e)
            #print(y_temp)
#            print(to_categorical(y_temp,nb_classes=3))
            loss, acc = model.train_on_batch(x, y_temp, class_weight=None)

  
    #tweets,_ = select_tweets('waseem', None)

    #Extracting learned embeddings 
    wordEmb = model.layers[0].get_weights()[0]

    #Step 2: Cross- Validation Using the XGB classifier and the learned embeddings
    word2vec_model = create_model(wordEmb,vocab)

    tweets = select_tweets_whose_embedding_exists(tweets, word2vec_model)

    X, y = gen_data(tweets, word2vec_model,'binary')
    X_test, y_test = gen_data(tweets_test,word2vec_model,'binary')

    print(y_test)

    cv_object = StratifiedKFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    
    for train_index, test_index in cv_object.split(X,y):
        X_train, y_train = X[train_index],y[train_index]
        #X_test, y_test = X[test_index],y[test_index]
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

        #X_test, y_test = sklearn.utils.shuffle(X_test, y_test)
        # print('size')
        # print(train_index)
        # print(test_index)
        import xgboost as xgb
        clf = xgb.XGBClassifier(use_label_encoder=False)
        clf.fit(X_train, y_train)
        model=clf
        #model = gradient_boosting_classifier(X_train, y_train)
        print(5)
        #precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_train, y_train, 'binary', 'train')
        #precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_test, y_test, 'binary', 'test')
        
    #     p += precisionw
    #     p1 += precisionm
    #     r += recallw
    #     r1 += recallm
    #     f1 += f1_scorew
    #     f11 += f1_scorem
    #     pn += precision
    #     rn += recall
    #     fn += f1_score
    # print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS)
    
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_train, y_train, 'binary', 'train')
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_test[967:,:], y_test[967:], 'binary','dev')
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_test[:967,:], y_test[:967], 'binary','test')
    
if __name__ == "__main__":
    TOKENIZER = 'glove'
    GLOVE_MODEL_FILE = 'glove.txt'
    EMBEDDING_DIM = 200
    OPTIMIZER = 'adam'
    LEARN_EMBEDDINGS = True
    EPOCHS = 10
    BATCH_SIZE = 128
    SCALE_LOSS_FUN = None
    NO_OF_FOLDS = 10
    SEED = 42
    np.random.seed(SEED)
    run_model_exp1()
