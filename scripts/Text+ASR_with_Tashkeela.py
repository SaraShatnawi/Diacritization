# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import pickle as pkl
import tensorflow as tf
import math
import re
import copy
import os
import argparse
os.environ["PYTHONIOENCODING"] = "utf-8"
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed, Input, Layer
from tensorflow.keras.utils import to_categorical,Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from keras.layers import Embedding, Layer





def set_seed(seed_value):
    # Set random seeds for reproducibility
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    

def print_unicode(text):
    import sys
    sys.stdout.buffer.write(text.encode('utf-8'))    
    
def remove_diacritics(data_raw):
  ''' Returns undiacritized text'''
  return data_raw.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))


def split_data_tashkeela(data_raw):
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
            if len(remove_diacritics(sub_line).strip()) > 0 and len(remove_diacritics(sub_line).strip()) <= 270:
                data_new.append(sub_line.strip())

            # split line if its longer than 500 characters
            else:
                sub_line = sub_line.split()
                tmp_line = ''
                for word in sub_line:
                    # append line without current word if new word will make it exceed 500 characters and start new line
                    if len(remove_diacritics(tmp_line).strip()) + len(remove_diacritics(word).strip()) + 1 > 270:
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
    
    


def build_LSTM_model(maxlen, vocab_size,asr_vocab_size, output_size, d_model, num_heads, \
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
    outputs_text = TimeDistributed(Dense(units=output_size))(dense2)
       
    
    
     # ASR input branch
    input_asr = Input(shape=(None,))
    asr_embeddings = Embedding(input_dim=asr_vocab_size, output_dim=d_model)(input_asr)
    
    blstm1_asr = Bidirectional(LSTM(units=dff, return_sequences=True))(asr_embeddings)
    dropout1_asr = Dropout(dropout_rate)(blstm1_asr)
    blstm2_asr = Bidirectional(LSTM(units=dff, return_sequences=True))(dropout1_asr)
    dropout2_asr = Dropout(dropout_rate)(blstm2_asr)
    dense1_asr = TimeDistributed(Dense(units=dff, activation='relu'))(dropout2_asr)
    dense2_asr = TimeDistributed(Dense(units=dff, activation='relu'))(dense1_asr)
    outputs_asr = TimeDistributed(Dense(units=output_size))(dense2_asr)
    
    # Multi-head attention between regular and ASR inputs
    cross_attention_output, attention_weights = MultiHeadAttention(num_heads, d_model, name="cross_attention")(outputs_text, outputs_asr, return_attention_scores=True)            

    # Combine branches
    combined = layers.Concatenate()([outputs_text, cross_attention_output])
    
    if args.with_conn:
        outputs = TimeDistributed(Dense(units=output_size))(combined)  # with Concatenation
    else:
        outputs = TimeDistributed(Dense(units=output_size))(cross_attention_output)  # without concatenation
      
    outputs = layers.Activation('softmax')(outputs)

    # Create the model
    model = Model(inputs=[inputs, input_asr], outputs=[outputs])
    
   
    
    return(model)    




class DataGenerator(Sequence):
    ''' Customized data generator to input line batches into the model '''
    def __init__(self, lines, batch_size):
        self.lines = lines
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        lines = self.lines[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, Y_batch = map_data(lines)

        X_max_seq_len = np.max([len(x) for x in X_batch])
        Y_max_seq_len = np.max([len(y) for y in Y_batch])

        assert(X_max_seq_len == Y_max_seq_len)

        X = list()
        for x in X_batch:
            x = list(x)
            x.extend([CHARACTERS_MAPPING['<PAD>']] * (X_max_seq_len - len(x)))
            X.append(np.asarray(x))

        Y_tmp = list()
        for y in Y_batch:
            y_new = list(y)
            y_new.extend(to_categorical([CLASSES_MAPPING['<PAD>']] * (Y_max_seq_len - len(y)), num_classes=len(CLASSES_MAPPING)))
            Y_tmp.append(np.asarray(y_new, dtype='int32'))
        Y_batch = Y_tmp

        Y_batch = np.array(Y_batch, dtype='int32')

        return (np.array(X), Y_batch)

    def __len__(self):
        return int(np.ceil(len(self.lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        lines = self.lines[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, Y_batch = map_data(lines)

        X_max_seq_len = np.max([len(x) for x in X_batch])
        Y_max_seq_len = np.max([len(y) for y in Y_batch])

        assert(X_max_seq_len == Y_max_seq_len)

        X = list()
        for x in X_batch:
            x = list(x)
            x.extend([CHARACTERS_MAPPING['<PAD>']] * (X_max_seq_len - len(x)))
            X.append(np.asarray(x))

        Y_tmp = list()
        for y in Y_batch:
            y_new = list(y)
            y_new.extend(to_categorical([CLASSES_MAPPING['<PAD>']] * (Y_max_seq_len - len(y)), len(CLASSES_MAPPING)))
            Y_tmp.append(np.asarray(y_new))
        Y_batch = Y_tmp

        Y_batch = np.array(Y_batch)

        return (np.array(X), Y_batch)    



def split_data(data_raw,n):
    ''' Splits data into a list of sentences, clauses, or lines shorter than 270 characters. '''

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
    print("splited_data_fun  ", len(data_new))
    return data_new[:n]


def expand_vocabulary(CHARACTERS_MAPPING, CLASSES_MAPPING):
  ExpandedVocabulary = {}
  max_id = max(CHARACTERS_MAPPING.values())
  for input_char, index in CHARACTERS_MAPPING.items():
    ExpandedVocabulary[input_char] = index

  index = 1
  for diacritic in CLASSES_MAPPING:
    if diacritic not in ExpandedVocabulary and \
      diacritic.strip() not in ["", "<N/A>"]:
      ExpandedVocabulary[diacritic] =  max_id + index
      index += 1

  return ExpandedVocabulary

def map_asr_data(data_raw, expanded_vocabulary):
    ''' Splits data lines into an array of characters as integers '''

    X = []

    max_seq_len = 0  # Initialize maximum sequence length


    for line in data_raw:
        x = [expanded_vocabulary['<SOS>']]

        for idx, char in enumerate(line):
            x.append(expanded_vocabulary[char])


        x.append(expanded_vocabulary['<EOS>'])

        X.append(x)

        max_seq_len = max(max_seq_len, len(x))  # Update maximum sequence length

    #the input sequences in X and output sequences in Y are padded to ensure they have a uniform length
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_seq_len, padding='post', value=CHARACTERS_MAPPING['<PAD>'])

    return X

class DataGenerator(Sequence):
    ''' Customized data generator to input line batches into the model '''
    def __init__(self, lines, asr_lines, batch_size, expanded_vocabulary):
        print("initial lines", len(lines))
        print("initial asr_lines", len(asr_lines))

        self.lines = lines
        self.asr_lines = asr_lines  # ASR data

        self.batch_size = batch_size
        self.expanded_vocabulary = expanded_vocabulary

    def __len__(self):
        return int(np.ceil(len(self.lines) / float(self.batch_size)))
    def __getitem__(self, idx):
        lines = self.lines[idx * self.batch_size:(idx + 1) * self.batch_size]
        asr_lines = self.asr_lines[idx * self.batch_size:(idx + 1) * self.batch_size]  # ASR data

        X_batch, Y_batch = map_data(lines)
        #print("#######################")
        #print("X_batch inside DataGeneration: ", X_batch.shape)
        #print("Y_batch inside DataGeneration: ", Y_batch.shape)


        X_asr_batch  = map_asr_data(asr_lines, expanded_vocabulary)  # ASR data

        #print("X_asr_batch inside DataGeneration: ", X_asr_batch.shape)
        X_max_seq_len = np.max([len(x) for x in X_batch])
        Y_max_seq_len = np.max([len(y) for y in Y_batch])
        #print("#######################3")

        assert(X_max_seq_len == Y_max_seq_len)

        X = list()
        for x in X_batch:
            x = list(x)
            x.extend([CHARACTERS_MAPPING['<PAD>']] * (X_max_seq_len - len(x)))
            X.append(np.asarray(x))


        Y_tmp = list()
        for y in Y_batch:
            y_new = list(y)
            y_new.extend(to_categorical([CLASSES_MAPPING['<PAD>']] * (Y_max_seq_len - len(y)), len(CLASSES_MAPPING)))
            Y_tmp.append(np.asarray(y_new))
        Y_batch = Y_tmp

        Y_batch = np.array(Y_batch)
        return (np.array(X_batch), np.array(X_asr_batch)), Y_batch


    
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



def build_modified_model(maxlen, vocab_size, asr_vocab_size, d_model, num_heads, dff, num_blocks, dropout_rate=0.5):

    # Regular input branch
    inputs = Input(shape=(None,))
    x = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)(inputs)

    # Transformer blocks for regular input
    for _ in range(2):
        x = TransformerBlock(d_model, num_heads, dff, dropout_rate)(x)

    # Dense layer for regular input
    x = tf.keras.layers.Dense(d_model)(x)

    # ASR input branch with dropout
    input_asr = Input(shape=(None,))
    asr_embeddings = TokenAndPositionEmbedding(maxlen, asr_vocab_size, d_model)(input_asr)
    #asr_embeddings = tf.keras.layers.Dropout(0.2)(asr_embeddings)  
    
    # Transformer blocks for ASR input with dropout
    for _ in range(2):
        asr_embeddings = TransformerBlock(d_model, num_heads, dff, dropout_rate)(asr_embeddings)

    # Dense layer for ASR input with dropout
    asr_attention = tf.keras.layers.Dense(d_model)(asr_embeddings)
    

    # Multi-head attention between regular and ASR inputs
    cross_attention_output, attention_weights = MultiHeadAttention(num_heads, d_model, name="cross_attention")(x, asr_attention, return_attention_scores=True)


 # Combine branches
    combined = layers.Concatenate()([x, cross_attention_output])
    if args.with_conn:
        outputs = TimeDistributed(Dense(units=output_size))(combined)  # with Concatenation
    else:
        outputs = TimeDistributed(Dense(units=output_size))(cross_attention_output)  # without concatenation
      
    outputs = layers.Activation('softmax')(outputs)
 

    # Create the model
    model = Model(inputs=[inputs, input_asr], outputs=outputs)
    return model


def map_weights_by_position(pretrained_model, modified_model, pretrained_layer_positions, modified_layer_positions):

# Specify the layer positions you want to transfer weights for

  for pretrained_position, modified_position in zip(pretrained_layer_positions, modified_layer_positions):
    pretrained_layer = pretrained_model.layers[pretrained_position]
    modified_layer = modified_model.layers[modified_position]
    shapes_modified = [x.shape for x in modified_layer.get_weights()]
    shapes_pretrained = [x.shape for x in pretrained_layer.get_weights()]
    assert shapes_modified == shapes_pretrained, "shapes are not the same for modified and pre-trained \n {} and \n {}".format(shapes_modified, shapes_pretrained)
    modified_model.layers[modified_position].set_weights(pretrained_layer.get_weights())




def fit_model(model, epochs, batch_size, train_split, val_split, asr_train, asr_val, expanded_vocabulary):
    ''' Fits model '''

    print("rfom fit ")
    print("train_split ", len(train_split))
    print(" asr_train ", len(asr_train))
    print("val_split ", len(val_split))
    print("asr_val ", len(asr_val))

    # Create data generators for regular input and ASR data
    training_generator = DataGenerator(train_split, asr_train, batch_size, expanded_vocabulary)
    print("LENGTH training_generator ", len(training_generator))

    val_generator = DataGenerator(val_split, asr_val, batch_size, expanded_vocabulary)
    print("LENGTH val_generator ", len(val_generator))

    # Define the checkpoint path
    checkpoint_path = 'fineTune_CLarTTS_ASR.h5'

    # Define the ModelCheckpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'  # Save the weights at the end of each epoch
    )

    print("outside looooop")
    (batch_x, batch_x_asr), batch_y = training_generator[5]
    print("batch_x ", batch_x.shape)
    print("batch_x_asr ", batch_x_asr.shape)
    print("batch_y ", batch_y.shape)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")


        # Training loop
        for batch_idx in range(len(training_generator)):
            (batch_x, batch_x_asr), batch_y = training_generator[batch_idx]  # Unpack the generator output
            model.fit([batch_x, batch_x_asr], batch_y, batch_size=batch_size, verbose=0)


        # Validation loop
        print("starting validation")
        start_time = time.time()
        val_loss, val_accuracy = model.evaluate(val_generator,
                                                batch_size=batch_size,
                                                verbose=0)
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")
        end_time = time.time()
        print('--- %s seconds ---' % round(end_time - start_time, 2))

        # Save model weights
        #model.save_weights(checkpoint_path)

    return


def predict(line, line_asr, model):
    ''' predict test line '''
    X, _ = map_data([line])
    #X_asr, _ = map_data([line_asr])
    X_asr  = map_asr_data([line_asr], expanded_vocabulary) 
    predictions = model.predict([X, X_asr], verbose=0).squeeze()

    # get most probable diacritizations for each character
    predictions = predictions[1:]
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


def _predict(line, line_asr, model):
    ''' predict test line '''
    X, _ = map_data([line])
    
    X_asr  = map_asr_data([line_asr], expanded_vocabulary) 
    predictions = model.predict([X, X_asr], verbose=0).squeeze()

    # get most probable diacritizations for each character
    predictions = predictions[1:]
    _, attention_scores=attention.predict([X, X_asr])
    return predictions

def break_and_predict(line, line_asr, model):

  line=remove_diacritics(line)

  _len=len(line)
  

  _len_ratio=len(line_asr)/_len
  


  output=''

  if _len > 100:

    start_idx=0

    end_idx=50

    while end_idx < _len:

      start=max(0, start_idx-25)

      end_idx=min(_len, (start_idx+50))

      end=min(_len,(end_idx+25))

      _line=line[start:end]

      #_line_asr=[ math.floor(start*_len_ratio) : math.floor(end*_len_ratio)]
      start_index_asr = math.floor(start * _len_ratio)
      end_index_asr = math.floor(end * _len_ratio)

      _line_asr = line_asr[start_index_asr:end_index_asr]
      
      res=_predict(_line, _line_asr, model)

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

    res=_predict(line, line_asr, model)

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

    parser.add_argument('--train_dataset', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_train_Ref.txt")

    parser.add_argument('--test_dataset', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_test_Ref.txt")  
   
    parser.add_argument('--train_dataset_ASR', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_train_finetunClart.txt")

    parser.add_argument('--test_dataset_ASR', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_test_finetunClart.txt")
    
    parser.add_argument('--batch_size', default=32)
    
    parser.add_argument('--vocab_size', default=77)
    
    parser.add_argument('--d_model', default=128)
    
    parser.add_argument('--num_heads', default=4)
    
    parser.add_argument('--dff', default=128)
    
    parser.add_argument('--num_blocks', default=2)
    
    parser.add_argument('--dropout_rate', default=0.2)
    
    parser.add_argument('--epochs', default=1)

    parser.add_argument('-c', '--count')      # option that takes a value

    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    
    parser.add_argument('--with_conn', default=True)
    
    parser.add_argument('--modelType', default='Transformer')
    
    
    
    parser.add_argument('--fine_tuned_model_path', default="/l/users/sara.shatnawi/diacritization/fixingCode/models/fineTuned_multiInput.h5")


    args = parser.parse_args()

    return args



def split_into_training_validation(lines, split_index=9000, val_split_index=1000):
  
    # Split the lines into training and validation sets
    
    
    train_raw_c = lines[:split_index]

    val_raw_c = lines[split_index:]

    train_split_c = split_data(train_raw_c, split_index)

    val_split_c = split_data(val_raw_c,val_split_index)

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

    # perform split_data on training and validation data

    train_raw = open(args.dataset_path+'traingit.txt', 'r',encoding='utf-8').readlines()

    val_raw = open(args.dataset_path+'valgit.txt', 'r',encoding='utf-8').readlines()
   

    train_split = split_data_tashkeela(train_raw)

    val_split = split_data_tashkeela(val_raw)




    # Load the pre-trained model with custom layers

    #/l/users/sara.shatnawi/diacritization/fixingCode/models/Tashkeela_Basic_model.h5
    
       
    
    if args.modelType=='Transformer':
      previousModel="/l/users/sara.shatnawi/diacritization/fixingCode/code/Tashkeela_Basic_model_sent_2Blocks.h5"
    else:
      previousModel="/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/models/LSTM_Tashkeela_model_b.h5"
   

    loaded_model = load_model(previousModel, 
                              custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 

                                              'split_data':split_data,

                                              'DataGenerator':DataGenerator, 

                                              'build_transformer_model': build_transformer_model, 

                                              'TransformerBlock':TransformerBlock, 

                                              'map_data': map_data})



    loaded_model.summary()
  
 
   
    
  
  
    #train with datsset

    # Read the contents of the file into a list (CLarTTS train data)

    lines = open(args.train_dataset, 'r', encoding='utf-8').readlines()
    print("length train_claratts: ", len(lines))
    train_split, val_split =  split_into_training_validation(lines)

    
    # Read the contents of the file into a list (ASR CLarTTS train data)
    lines = open(args.train_dataset_ASR, 'r', encoding='utf-8').readlines()
    print("length train_claratts: ", len(lines))

    train_split_ASR, val_split_ASR =  split_into_training_validation(lines)
    
    
    # Calculate the maximum sequence length in the dataset
    maxlen =186
    expanded_vocabulary = expand_vocabulary(CHARACTERS_MAPPING, CLASSES_MAPPING)
    vocab_asr_size = len(CHARACTERS_MAPPING) + len(expanded_vocabulary)
    output_size=len(CLASSES_MAPPING)
    
    # Build and compile the model
    modified_model = build_modified_model(maxlen, args.vocab_size, vocab_asr_size,args.d_model, args.num_heads, args.dff, args.num_blocks,                args.dropout_rate)

    modified_model.summary()
    print("the above is the CLar_ASR model" )
    
    


    if args.modelType=='Transformer':
             
       pretrained_layer_positions = [1,2,3] # Add all the desired positions
       modified_layer_positions = [2,4,6]# Add corresponding positions for the modified model
    else:
       pretrained_layer_positions =  [1,2,3,4,5,6,7,8] #[1,2,3] # Add all the desired positions
       modified_layer_positions = [2,4,6,8,10,12,14,16] #[2,4,6]# Add corresponding positions for the modified model
  

    # Call the function to get the mapping between layers based on the specified positions
    map_weights_by_position(loaded_model, modified_model, pretrained_layer_positions, modified_layer_positions)
    updated_layer = {}
    for index, x in enumerate(modified_layer_positions):
     updated_layer[index] = modified_model.layers[x].get_weights()
     
    
    
    modified_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
                       metrics=['accuracy'])

    modified_model.summary()
    print (" this is new model")


    hist_1 =fit_model(modified_model,args.epochs, args.batch_size, train_split, val_split,  train_split_ASR, val_split_ASR, expanded_vocabulary)
    attention = keras.Model(inputs=modified_model.inputs,
                                 outputs=modified_model.get_layer("cross_attention").output)  

    # Save the fine-tuned model

    modified_model.save(args.fine_tuned_model_path)









        
 
        
