# -*- coding: utf-8 -*-
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import time
import random
import argparse
import numpy as np
import pickle as pkl
import tensorflow as tf
import re
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed, Input, Layer, MultiHeadAttention
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split



def set_seed(seed_value):
    # Set random seeds for reproducibility
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def remove_diacritics(data_raw):
  ''' Returns undiacritized text'''
  return data_raw.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))


def print_unicode(text):
    import sys
    sys.stdout.buffer.write(text.encode('utf-8'))
    

def split_data(data_raw):
    ''' Splits data into a list of sentences, clauses, or lines shorter than 500 characters. '''

    # initialize returned list
    data_new = list()

    # create new lines at paranthesis or puctuation
    for line in data_raw:
        
        # loop on created new lines
        for sub_line in line.split('\n'):
            # do nothing if line is empty
            if len(remove_diacritics(sub_line).strip()) == 0:
                continue
            # append line to list if line, without diacritics, is shorter than 500 characters
            if len(remove_diacritics(sub_line).strip()) > 0 and len(remove_diacritics(sub_line).strip()) <= 100:
                data_new.append(sub_line.strip())

            # split line if its longer than 500 characters
            else:
                sub_line = sub_line.split()
                tmp_line = ''
                for word in sub_line:
                    # append line without current word if new word will make it exceed 500 characters and start new line
                    if len(remove_diacritics(tmp_line).strip()) + len(remove_diacritics(word).strip()) + 1 > 100:
                        if len(remove_diacritics(tmp_line).strip()) > 0:
                            data_new.append(tmp_line.strip())
                        tmp_line = word
                    else:
                        # set new line to current word if line is still empty
                        if tmp_line == '':
                            tmp_line = word
                        # add whitespace and word to line if line is not empty but shorter than 500 characters
                        else:
                            tmp_line += ' '
                            tmp_line += word
                if len(remove_diacritics(tmp_line).strip()) > 0:
                    data_new.append(tmp_line.strip())

    return data_new



def map_data(data_raw):
    ''' Splits data lines into an array of characters as integers '''

    X = []
    Y = []

    max_seq_len = 0  # Initialize maximum sequence length


    for line in data_raw:
        x = [CHARACTERS_MAPPING['<SOS>']]
        y = [CLASSES_MAPPING['<SOS>']]

        for idx, char in enumerate(line):
            if char in DIACRITICS_LIST:
                continue

            x.append(CHARACTERS_MAPPING[char])

            if char not in ARABIC_LETTERS_LIST:
                y.append(CLASSES_MAPPING[''])
            else:
                char_diac = ''
                if idx + 1 < len(line) and line[idx + 1] in DIACRITICS_LIST:
                    char_diac = line[idx + 1]
                    if idx + 2 < len(line) and line[idx + 2] in DIACRITICS_LIST and char_diac + line[idx + 2] in CLASSES_MAPPING:
                        char_diac += line[idx + 2]
                    elif idx + 2 < len(line) and line[idx + 2] in DIACRITICS_LIST and line[idx + 2] + char_diac in CLASSES_MAPPING:
                        char_diac = line[idx + 2] + char_diac
                y.append(CLASSES_MAPPING[char_diac])

        x.append(CHARACTERS_MAPPING['<EOS>'])
        y.append(CLASSES_MAPPING['<EOS>'])

        X.append(x)
        Y.append(y)

        max_seq_len = max(max_seq_len, len(x))  # Update maximum sequence length

    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_seq_len, padding='post', value=CHARACTERS_MAPPING['<PAD>'])
    Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=max_seq_len, padding='post', value=CLASSES_MAPPING['<PAD>'])

    return X, Y
    
    
    


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model  # Initialize d_model attribute
        self.num_heads = num_heads  # Initialize num_heads attribute
        self.dff = dff  # Initialize dff attribute
        self.rate = rate  # Initialize rate attribute
        
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = layers.Dropout(rate)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        self.dropout2 = layers.Dropout(rate)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, inputs, training=False):
        attention_output = self.multi_head_attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.layer_norm1(inputs + attention_output)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        block_output = self.layer_norm2(attention_output + ffn_output)

        return block_output

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
        }
        return config


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings

    def get_config(self):
        config = {
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        }
        return config

def build_transformer_model(maxlen, vocab_size, d_model, num_heads, dff, num_blocks, dropout_rate=0.5):
    inputs = Input(shape=(None,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
    x = embedding_layer(inputs)
    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, dff, dropout_rate)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_LSTM_model(maxlen, vocab_size, output_size, d_model, num_heads, \
                         dff, num_blocks, dropout_rate=0.5):

    # Regular input branch
    inputs = Input(shape=(None,))

    embeddings = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)

    blstm1 = Bidirectional(LSTM(units=dff, return_sequences=True))(embeddings)
    dropout1 = Dropout(dropout_rate)(blstm1)
    blstm2 = Bidirectional(LSTM(units=dff, return_sequences=True))(dropout1)
    dropout2 = Dropout(dropout_rate)(blstm2)
    dense1 = TimeDistributed(Dense(units=dff, activation='relu'))(dropout2)
    dense2 = TimeDistributed(Dense(units=dff, activation='relu'))(dense1)
    outputs = TimeDistributed(Dense(units=output_size))(dense2)
    outputs = layers.Activation('softmax')(outputs)

    # Create the model
    model = Model(inputs=[inputs], outputs=[outputs])
    #model = Model(inputs=[inputs, input_asr], outputs=[outputs, cross_attention_output])
    return(model)


class DataGenerator(Sequence):
    ''' Customized data generator to input line batches into the model '''
    def __init__(self, lines, batch_size):
        print("initial lines", len(lines))

        self.lines = lines

        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.lines) / float(self.batch_size)))
    def __getitem__(self, idx):
        lines = self.lines[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch, Y_batch = map_data(lines)


        #print("X_asr_batch inside DataGeneration: ", X_asr_batch.shape)
        X_max_seq_len = np.max([len(x) for x in X_batch])
        Y_max_seq_len = np.max([len(y) for y in Y_batch])
        
        assert(X_max_seq_len == Y_max_seq_len)

       
        return X_batch, Y_batch



def fit_model(model, epochs, batch_size, train_split, val_split):
    #random.shuffle(train_split)
    #train_split = list(sorted(train_split, key=lambda line: len(remove_diacritics(line))))
    #random.shuffle(val_split)
    #val_split = list(sorted(val_split, key=lambda line: len(remove_diacritics(line))))

    checkpoint_path = 'checkpoints/epoch{epoch:02d}.ckpt'
    checkpoint_cb = ModelCheckpoint(checkpoint_path, verbose=0)

    training_generator = DataGenerator(train_split, batch_size)
    val_generator = DataGenerator(val_split, batch_size)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        callbacks=[checkpoint_cb])





def predict(line, model):
    ''' predict test line '''
    X, _ = map_data([line])
    predictions = model.predict(X).squeeze()
    # get most probable diacritizations for each character
    predictions = predictions[1:]

    # initialize empty output line
    output = ''
    # loop on input characters and predicted diacritizations
    for char, prediction in zip(remove_diacritics(line), predictions):
        # append character
        output += char
        # if character is not an arabic letter continue
        if char not in ARABIC_LETTERS_LIST:
            continue

        if '<' in REV_CLASSES_MAPPING[np.argmax(prediction)]:
            continue

        # if character in arabic letters append predicted diacritization
        output += REV_CLASSES_MAPPING[np.argmax(prediction)]

    return output






def _predict(line, model):
    ''' predict test line '''
    X, _ = map_data([line])
    #X_asr  = map_asr_data([line_asr], expanded_vocabulary)
    predictions = model.predict(X, verbose=0).squeeze()

    # get most probable diacritizations for each character
    predictions = predictions[1:]
    #_, attention_scores=attention.predict([X, X_asr])

    #print("cross_attention_weights: ", attention_scores[0][0][15])
    #print(X.shape, X_asr.shape)
    #print(attention_scores.shape)
    return predictions#, attention_scores



def break_and_predict(line, model):

  line=remove_diacritics(line)

  _len=len(line)

  #_len_ratio=len(line_asr)/_len



  output=''

  if _len > 270:

    start_idx=0

    end_idx=50

    while end_idx < _len:

      start=max(0, start_idx-25)

      end_idx=min(_len, (start_idx+50))

      end=min(_len,(end_idx+25))

      _line=line[start:end]



      res=_predict(_line, model)

      #print(start_idx, end_idx, start, end)

      for i in range (start_idx-start, end_idx-start):

        #print (i, _line)

        char=_line[i]

        prediction=res[i]

        output+= char

        if char not in ARABIC_LETTERS_LIST:

              continue

        if '<' in REV_CLASSES_MAPPING[np.argmax(prediction)]:

              continue

        output += REV_CLASSES_MAPPING[np.argmax(prediction)]



      start_idx=end_idx

  else:

    res=_predict(line, model)

    for i in range (len(line)):

        #print (i, _line)



        char=line[i]

        prediction=res[i]

        output+= char

        if char not in ARABIC_LETTERS_LIST:

              continue

        if '<' in REV_CLASSES_MAPPING[np.argmax(prediction)]:

              continue

        output += REV_CLASSES_MAPPING[np.argmax(prediction)]

  return output




 

def parse_arguments():

    parser = argparse.ArgumentParser(description='Diacritic Restoration Fine-tuning and Inference with Speech and Text Inputs')

    parser.add_argument('--seed_value', default=43)

    parser.add_argument('--with_extra_train', action="store_true")

    parser.add_argument('--aux_dataset_path', default="/l/users/sara.shatnawi/diacritization/")

    parser.add_argument('--dataset_path', default="/l/users/sara.shatnawi/datasets/")

    parser.add_argument('--train_dataset', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_train_Ref.txt")#
    #/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_train_Ref.txt
    #/l/users/sara.shatnawi/datasets/traingit.txt
    parser.add_argument('--batch_size', default=32)

    parser.add_argument('--test_dataset', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_test_Ref.txt")#
    #/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_test_Ref.txt
    #/l/users/sara.shatnawi/datasets/valgit.txt
    
    parser.add_argument('--vocab_size', default=77)
    
    parser.add_argument('--d_model', default=128)
    
    parser.add_argument('--num_heads', default=4)
    
    parser.add_argument('--dff', default=128)
    
    parser.add_argument('--num_blocks', default=2)
    
    parser.add_argument('--dropout_rate', default=0.2)
    
    parser.add_argument('--epochs', default=1)

    parser.add_argument('-c', '--count')      # option that takes a value

    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    
    parser.add_argument('--fine_tuned_model_path', default="/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/models/LSTM_ClarTTS_model_b.h5")
   
    parser.add_argument('--modelType', default='Bi_LSTM')


    args = parser.parse_args()

    return args


def split_into_training_validation(lines, split_ratio=0.9):

    # Calculate the split index

    split_index = int(len(lines) * split_ratio)



    # Split the lines into training and validation sets

    train_raw_c = lines[:split_index]

    val_raw_c = lines[split_index:]

    train_split_c = split_data(train_raw_c)

    val_split_c = split_data(val_raw_c)

    return train_split_c, val_split_c







if __name__=="__main__":

    args = parse_arguments()

    batch_size=args.batch_size

    set_seed(args.seed_value)
    physical_devices = tf.config.list_physical_devices('GPU')
    print("num of GPUs= ",physical_devices)



    if args.with_extra_train:

        CHARACTERS_MAPPING = pkl.load(open(args.aux_dataset_path +'RNN_BIG_CHARACTERS_MAPPING.pickle', 'rb'))

    else:

        CHARACTERS_MAPPING = pkl.load(open(args.aux_dataset_path +'RNN_SMALL_CHARACTERS_MAPPING.pickle', 'rb'))

    



    ARABIC_LETTERS_LIST = pkl.load(open(args.aux_dataset_path +'ARABIC_LETTERS_LIST.pickle', 'rb'))

    DIACRITICS_LIST = pkl.load(open( args.aux_dataset_path +'DIACRITICS_LIST.pickle', 'rb'))

    CLASSES_MAPPING = pkl.load(open(args.aux_dataset_path +'RNN_CLASSES_MAPPING.pickle', 'rb'))

    REV_CLASSES_MAPPING = pkl.load(open(args.aux_dataset_path +'RNN_REV_CLASSES_MAPPING.pickle', 'rb'))





    #train with datsset

    # Read the contents of the file into a list

    lines = open(args.train_dataset, 'r', encoding='utf-8').readlines()
    print("length train: ", len(lines))

    train_split, val_split =  split_into_training_validation(lines)

    # Calculate the maximum sequence length in the dataset
    maxlen =max(len(line) for line in train_split + val_split)
    output_size=len(CLASSES_MAPPING)


  
    # Build and compile the model
      
    if args.modelType=='Transformer':
      model = build_transformer_model(maxlen, args.vocab_size, args.d_model, args.num_heads, args.dff, args.num_blocks, args.dropout_rate)
      model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,           epsilon=1e-07), metrics=['accuracy'])
    else:
      model = build_LSTM_model(maxlen,args.vocab_size,output_size, args.d_model, args.num_heads, args.dff, args.num_blocks, args.dropout_rate)
      #learning_rate=0.001, epsilon=1e-07
      model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    
    model.summary()
    fit_model(model, args.epochs, args.batch_size, train_split, val_split)




    # Save the fine-tuned model

    model.save(args.fine_tuned_model_path)
    
    









    
    
    
    
    





