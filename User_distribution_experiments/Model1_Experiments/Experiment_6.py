import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *


def run_exp_6 (flag, strategy):
    #Experimento Cross Domain dataset partition
    tweets_train,_ = select_tweets('data_new',strategy)
    tweets_test,_ =  select_tweets('sem_eval',strategy)
    
    vocab = gen_vocab(tweets_train)

    MAX_SEQUENCE_LENGTH = max_len(tweets_train)
    
    train_LSTM_Cross_Domain(tweets_train,tweets_test,MAX_SEQUENCE_LENGTH,vocab)
    
def train_LSTM_Cross_Domain(tweets_train,tweets_test,MAX_SEQUENCE_LENGTH,vocab):
        a, p, r, f1 = 0., 0., 0., 0.
        a1, p1, r1, f11 = 0., 0., 0., 0.
        pn,rn,fn = 0.,0.,0.
        flag='binary'
        
        model = lstm_model_bin(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, vocab)

        shuffle_weights(model)
        
        X_train, y_train = gen_sequence(tweets_train,vocab,'binary')
        
        X_test, y_test = gen_sequence(tweets_test,vocab,'binary')
        
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
        
        y_train = np.array(y_train)
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        
        for epoch in range(EPOCHS):
            for X_batch in batch_gen(X_temp, BATCH_SIZE):
                x = X_batch[:, :MAX_SEQUENCE_LENGTH]
                y_temp = X_batch[:, MAX_SEQUENCE_LENGTH]

                class_weights = None
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                #print (loss, acc)
                
        temp = model.predict_on_batch(X_test)
        y_pred=[]
        for i in temp:
            if i[0] > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
#         print (classification_report(y_test, y_pred))
#         print (precision_recall_fscore_support(y_test, y_pred))
        
        wordEmb = model.layers[0].get_weights()[0]
        
        word2vec_model = create_model(wordEmb,vocab)

        X_train, y_train = gen_data(tweets_train,word2vec_model,flag)
        X_test, y_test = gen_data(tweets_test,word2vec_model,flag)
        
        import xgboost as xgb
        clf = xgb.XGBClassifier(use_label_encoder=False)
        clf.fit(X_train, y_train)
        model=clf
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_train, y_train, 'binary','train')
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_test[3001:,:], y_test[3001:], 'binary','dev')
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_test[:3001,:], y_test[:3001], 'binary','test')
#         a += acc
        p += precisionw
        p1 += precisionm
        r += recallw
        r1 += recallm
        f1 += f1_scorew
        f11 += f1_scorem
        pn += precision
        rn += recall
        fn += f1_score
        print_scores(p, p1, r,r1, f1, f11,pn, rn, fn, 1)

    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    #parser.add_argument('-t', '--type',choices=['binary', 'categorical'], default = 'categorical')
    TOKENIZER = 'glove'
    GLOVE_MODEL_FILE = 'glove.txt'
    EMBEDDING_DIM = 200
    OPTIMIZER = 'adam'
    INITIALIZE_WEIGHTS_WITH = 'glove'
    LEARN_EMBEDDINGS = True
    EPOCHS = 100
    BATCH_SIZE = 128
    SCALE_LOSS_FUN = None
    SEED = 42
    np.random.seed(SEED)

    run_exp_6('binary', None)
