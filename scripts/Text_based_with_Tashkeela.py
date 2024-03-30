# -*- coding: utf-8 -*-

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import argparse
import time
import random
import re
import numpy as np
import pickle as pkl
import tensorflow as tf
import inference

from tensorflow.keras.models import Model, load_model
from keras.layers import Embedding, Layer
from tensorflow.keras import layers
from tensorflow.keras.layers import (Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed, Input, Layer, MultiHeadAttention)
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


def set_seed(seed_value):

    # Set random seeds for reproducibility

    np.random.seed(seed_value)

    tf.random.set_seed(seed_value)



def remove_diacritics(data_raw):

  ''' Returns undiacritized text'''

  return data_raw.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))



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

    predictions = model.predict(X, verbose = 0).squeeze()

    # get most probable diacritizations for each character

    predictions = predictions[1:]

    return predictions



def break_and_predict(line, model):

  line=remove_diacritics(line)

  _len=len(line)

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



def print_unicode(text):

    import sys

    sys.stdout.buffer.write(text.encode('utf-8'))





def parse_arguments():

    parser = argparse.ArgumentParser(description='Diacritic Restoration Fine-tuning and Inference with Speech and Text Inputs')

    parser.add_argument('--seed_value', default=43)

    parser.add_argument('--with_extra_train', action="store_true")

    parser.add_argument('--aux_dataset_path', default="/l/users/sara.shatnawi/diacritization/")

    parser.add_argument('--dataset_path', default="/l/users/sara.shatnawi/datasets/")

    parser.add_argument('--asr_dataset_path', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_train_Ref.txt")

    parser.add_argument('--batch_size', default=32)

    parser.add_argument('--test_dataset', default="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_test_Ref.txt") #mal_finetunClart_Ref.txt femal_finetunClart_Ref.txt  ASC_Ref_test_ArabicText_TextOnly.txt
    
    parser.add_argument('--modelType', default='BiLSTM')
    
    parser.add_argument('--epoch', default=1)

    parser.add_argument('--fine_tuned_model_path', default="/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/models/FineTune_Tashkeela_ClarTTS_model.h5")



    parser.add_argument('-c', '--count')      # option that takes a value

    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag


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



    # perform split_data on training and validation data

    train_raw = open(args.dataset_path+'traingit.txt', 'r',encoding='utf-8').readlines()

    val_raw = open(args.dataset_path+'valgit.txt', 'r',encoding='utf-8').readlines()

    

    train_split = split_data(train_raw)

    val_split = split_data(val_raw)



    #train with CLArTTs datsset

    # Read the contents of the file into a list

    lines = open(args.asr_dataset_path, 'r', encoding='utf-8').readlines()

    # Shuffle the lines randomly

    #random.shuffle(lines)



    print("length train_claratts: ", len(lines))

    train_split_c, val_split_c =  split_into_training_validation(lines)

    new_train_generator = DataGenerator(train_split_c, batch_size)

    new_val_generator = DataGenerator(val_split_c, batch_size)



    # Load the pre-trained model with custom layers

    #/l/users/sara.shatnawi/diacritization/fixingCode/models/Tashkeela_Basic_model.h5
     
    if args.modelType=='Transformer':
      previousModel="/l/users/sara.shatnawi/diacritization/fixingCode/code/Tashkeela_Basic_model_sent_2Blocks.h5"
    else:
      previousModel="/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/LSTM_Tashkeela_model.h5"
      #"/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/models/LSTM_Tashkeela_model.h5"
      
    print(previousModel)

    loaded_model = load_model(previousModel, 

                              custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 

                                              'split_data':split_data,

                                              'DataGenerator':DataGenerator, 

                                              'build_transformer_model': build_transformer_model, 
                                              
                                              'build_LSTM_model': build_LSTM_model,
                                              
                                              'TransformerBlock':TransformerBlock, 

                                              'map_data': map_data})



    loaded_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07), metrics=['accuracy'])
    
   
    
    



    # Fit the model with the checkpoint callback

    new_history = loaded_model.fit(

        x=new_train_generator,

        validation_data=new_val_generator,

        epochs=args.epoch,

        #callbacks=[checkpoint_callback]

    )



    # Save the fine-tuned model

    loaded_model.save(args.fine_tuned_model_path)
    
    
    
    #### ###################### Predict Clartts ###############################

    reference_file_path = "/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/QASR_Ref.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = break_and_predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    
    #"/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_clart_textBased/pred_lstm_clartts_textBased_tashkeela.txt"
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/QASR_results/pred_lstm_QASR_text_fineText.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")
    
    
    
    #### ###################### Predict Clartts ###############################

    """reference_file_path = "/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_test_Ref.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    
    #"/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_clart_textBased/pred_lstm_clartts_textBased_tashkeela.txt"
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/wthoutInference/clar2.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")

    #### ###################### Predict QASR Male ###############################

    reference_file_path = "/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/mal_finetunClart_Ref.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    
    #"/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_clart_textBased/pred_lstm_Male_textBased_tashkeela.txt"
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/wthoutInference/male2.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")
    
    
    
    #### ###################### Predict QASR Female ###############################

    reference_file_path ="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/femal_finetunClart_Ref.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    
    #"/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_clart_textBased/pred_lstm_female_textBased_tashkeela.txt"
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/wthoutInference/female2.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")
    
    
    #### ###################### Predict ASC ###############################

    reference_file_path = "/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ASC_Ref_test_ArabicText_TextOnly.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    
    #"/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_clart_textBased/pred_lstm_ASC_textBased_tashkeela.txt"
    #"/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/wthoutInference/ASC2.txt"
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_clart_textBased/pred_lstm_ASC_textBased_fineTuned.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")"""
    
    
    

    #### ###################### Predict Clartts ###############################

    """reference_file_path = "/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ClarTTS_test_Ref.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = break_and_predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_fineTune_tashkeela_clartts/pred_lstm_clar.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")

    #### ###################### Predict QASR Male ###############################

    reference_file_path = "/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/mal_finetunClart_Ref.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = break_and_predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_fineTune_tashkeela_clartts/pred_lstm_male.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")
    
    
    
    #### ###################### Predict QASR Female ###############################

    reference_file_path ="/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/femal_finetunClart_Ref.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = break_and_predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_fineTune_tashkeela_clartts/pred_lstm_female.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")
    
    
    #### ###################### Predict ASC ###############################

    reference_file_path = "/l/users/sara.shatnawi/diacritization/whisper/Evaluate_Whisper/ASC_Ref_test_ArabicText_TextOnly.txt"
    
  
    with open(reference_file_path, 'r', encoding='utf-8') as reference_file:
        reference_lines = reference_file.readlines()

    print("len of reference_lines: ", len(reference_lines))

   
    predicted_lines = []

    for reference_line in reference_lines:
     reference_contents = reference_line.strip()

     #prediction_fun
     predicted_line = break_and_predict(remove_diacritics(reference_contents), loaded_model)
     predicted_lines.append(predicted_line)  # Add predicted line to the list


    # Write predicted lines to a file
    with open("/l/users/sara.shatnawi/diacritization/fixingCode/GitHub_clean_code/lstm/results/by_fineTune_tashkeela_clartts/pred_lstm_ASC.txt", 'w', encoding='utf-8') as output_file:
        for line in predicted_lines:
            output_file.write(line + '\n')

    print("****Done****")"""





