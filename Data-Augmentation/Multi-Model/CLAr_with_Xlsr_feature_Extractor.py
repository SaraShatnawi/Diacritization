# -*- coding: utf-8 -*-

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import time
import torch
import random
import numpy as np
import pickle as pkl
import tensorflow as tf
import math
import random
import copy
import re
import os
import argparse
import librosa
from tensorflow import keras
import torchaudio
from sklearn.cluster import MiniBatchKMeans as KMeans
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed, Input, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from itertools import groupby
from transformers import Wav2Vec2Processor
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC    
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor




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
            # append line to list if line, without diacritics, is shorter than 270 characters
            if len(remove_diacritics(sub_line).strip()) > 0 and len(remove_diacritics(sub_line).strip()) <= 270:
                data_new.append(sub_line.strip())


            # split line if its longer than 270 characters
            else:
                sub_line = sub_line.split()
                tmp_line = ''
                for word in sub_line:
                    # append line without current word if new word will make it exceed 270 characters and start new line
                    if len(remove_diacritics(tmp_line).strip()) + len(remove_diacritics(word).strip()) + 1 > 270:
                        if len(remove_diacritics(tmp_line).strip()) > 0:
                            data_new.append(tmp_line.strip())
                        tmp_line = word
                    else:
                        # set new line to current word if line is still empty
                        if tmp_line == '':
                            tmp_line = word
                        # add whitespace and word to line if line is not empty but shorter than 270 characters
                        else:
                            tmp_line += ' '
                            tmp_line += word
                if len(remove_diacritics(tmp_line).strip()) > 0:
                    data_new.append(tmp_line.strip())
    print("splited_data_fun  ", len(data_new))
    return data_new[:n]

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

class DataGenerator(Sequence):

    ''' Customized data generator to input line batches into the model '''

    def __init__(self, lines, seq_data, batch_size, expanded_vocabulary=None):
      self.lines = lines
      self.seq_data = seq_data
      self.batch_size = batch_size
      self.expanded_vocabulary = expanded_vocabulary


    def __len__(self):

      return int(np.ceil(len(self.lines) / float(self.batch_size)))

    def __getitem__(self, idx):
      lines = self.lines[idx * self.batch_size:(idx + 1) * self.batch_size]
      seq_data = self.seq_data[idx * self.batch_size:(idx + 1) * self.batch_size]
      X_batch, Y_batch = map_data(lines)
      #print("X_batch ", X_batch)
      #print("Y_batch ", Y_batch)
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

      if self.expanded_vocabulary:
        pass

      else:

        seq_max_len = np.max([len(seq) for seq in seq_data])
        seq_padded = pad_sequences(seq_data, maxlen=seq_max_len, padding='post', value=0)


      return ([np.array(X), seq_padded], Y_batch)



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

#Build the text encoder part
def build_text_encoder(inputs_text, maxlen, vocab_size, d_model, num_heads, dff,num_blocks, dropout_rate):

    x_text = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)(inputs_text)
    # Transformer blocks for regular text input
    for _ in range(num_blocks):
      x_text = TransformerBlock(d_model, num_heads, dff, dropout_rate)(x_text)
    # Dense layer for regular text input
    x_text = tf.keras.layers.Dense(d_model)(x_text)

    return x_text

#Build the Cluster part
def build_cluster_encoder(inputs_cluster, maxlen,cluster_feature_size, d_model, num_heads, dff,num_blocks, dropout_rate,architecture_type):


    #cluster_embeddings = TokenAndPositionEmbedding(maxlen, cluster_feature_size, d_model)(inputs_cluster)
    if architecture_type == "dense":
      #add a Dense layer and activation to process the cluster features
      cluster_embeddings = Dense(d_model, activation='relu')(inputs_cluster)
    if architecture_type == "transformer_blocks":
      # Transformer blocks for regular text input
      for _ in range(num_blocks):
        cluster_embeddings = TransformerBlock(d_model, num_heads, dff, dropout_rate)(cluster_embeddings)
      cluster_embeddings = tf.keras.layers.Dense(d_model)(cluster_embeddings)
    if architecture_type == "None":
      #inputs_cluster = tf.keras.layers.Dense(d_model)(inputs_cluster)
      return inputs_cluster

    return cluster_embeddings

#from keras.layers import Reshape

def build_modified_model(maxlen_text, max_len, vocab_size,cluster_feature_size,d_model, num_heads, dff,num_blocks_text_encoder, num_blocks_speech_encoder, dropout_rate=0.5
                          ,architecture_type="None",with_concatenation=True):

    # Regular text input branch
    inputs_text = Input(shape=(None,))
    x_text = build_text_encoder(inputs_text, maxlen_text, vocab_size,d_model, num_heads, dff,num_blocks_text_encoder, dropout_rate)

    # Cluster input branch
    inputs_cluster = Input(shape=(None,52))
    #inputs_cluster =  #Input(shape=(batch_size, None, 52))# Input((None, 11648))


    #(batch_size, sequence_length, features)



    cluster_embeddings = build_cluster_encoder(inputs_cluster, max_len,cluster_feature_size,d_model, num_heads, dff,num_blocks_speech_encoder, dropout_rate,architecture_type)

    # Multi-head attention between regular text and cluster inputs
    cross_attention_output, attention_weights = MultiHeadAttention(num_heads, d_model, name="cross_attention")(x_text, inputs_cluster,
     return_attention_scores=True)

    # concatenate branches
    combined = layers.Concatenate()([x_text, cross_attention_output]) if with_concatenation else cross_attention_output
    outputs = TimeDistributed(Dense(vocab_size))(combined) # with Concatenation
    #outputs = TimeDistributed(Dense(vocab_size))(cross_attention_output) # without Concatenation
    outputs = layers.Activation('softmax')(outputs)




    # Create the model
    model = Model(inputs=[inputs_text, inputs_cluster], outputs=outputs)
    return model

def fit_model(model, epochs, batch_size,train_split, val_split, seq_data_train,seq_data_val,expanded_vocabulary=None):

    ''' Fits model '''
    print("from fit ")
    print("train_split ", len(train_split))
    print("val_split ", len(val_split))
    print("seq_data ", len(seq_data_train))
    print("seq_data val ", len(seq_data_val))
    # Create data generator for regular input and the new sequence data
    training_generator = DataGenerator(train_split, seq_data_train, batch_size)
    print("LENGTH training_generator ", len(training_generator))
    val_generator = DataGenerator(val_split, seq_data_val, batch_size)
    print("LENGTH val_generator ", len(val_generator))

    # Define the checkpoint path
    checkpoint_path = 'fineTune_CLarTTS_AudioFeatures.h5'

    # Define the ModelCheckpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      save_weights_only=True,
      monitor='val_loss',
      mode='min',
      save_best_only=True,
      save_freq='epoch' # Save the weights at the end of each epoch
      )

    """print("outside loop")
    (batch_x, batch_seq), batch_y = training_generator[5]
    print("batch_x ", batch_x.shape)
    print("batch_seq ", batch_seq.shape)
    print("batch_y ", batch_y.shape)"""

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training loop
        """for batch_idx in range(len(training_generator)):
            (batch_x, batch_seq), batch_y = training_generator[batch_idx]  # Unpack the generator output
            model.fit([batch_x, batch_seq], batch_y, batch_size=batch_size, verbose=0)"""

        for batch_idx in range(len(training_generator)):
         (batch_x, batch_seq), batch_y = training_generator[batch_idx]  # Unpack the generator output
         #print("Batch X shape:", batch_x.shape)
         #print("Batch Y shape:", batch_y.shape)
         model.fit([batch_x, batch_seq], batch_y, batch_size=batch_size, verbose=0)

        # Validation loop
        print("starting validation")
        start_time = time.time()
        val_loss, val_accuracy = model.evaluate(val_generator,
                                                batch_size=batch_size,
                                                verbose=0)
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")
        end_time = time.time()
        print('--- %s seconds ---' % round(end_time - start_time, 2))
    return
    
    
    
    
    
def split_into_training_validation(lines, split_index=9000, val_split_index=1000):
#def split_into_training_validation(lines, split_index=8, val_split_index=2):

    # Split the lines into training and validation sets


    train_raw_c = lines[:split_index]

    val_raw_c = lines[split_index:]
    #val_raw_c = lines[8:10]

    train_split_c = split_data(train_raw_c, split_index)

    val_split_c = split_data(val_raw_c,val_split_index)

    return train_split_c, val_split_c
    
    
def extract_features(audio_path):

    audio_input, _ = librosa.load(audio_path, sr=16000)
    inputs = xlsr_processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Move input tensor to GPU
    inputs = {key: value.to("cuda") for key, value in inputs.items()}


    with torch.no_grad():
        logits = xlsr_model(**inputs).logits

    features = logits.squeeze(0).cpu().numpy()
    return features
    
    
#prepare the data
def prep_data (filename,path_audio):
  metadata_df=[]
  #Sorted and mapping the validation data
  for line in filename:
    parts = line.strip().split('|')
    if len(parts) == 2:
      label, text = parts
      #print_unicode(label)
      seq=extract_features(path_audio+label+".wav")
      data_entry = {"path": label, "sentence" : text, "seq" :seq }
      metadata_df.append(data_entry)

  #metadata_df = pd.DataFrame(metadata_df)
  print("length ", len(metadata_df))
  return metadata_df


def predict_with_Audio_features(audio_file,line, model):
    ''' predict test line '''
    X, _ = map_data([line])
    #X_asr, _ = map_data([line_asr])
    xlsr_audio_features  = extract_features(audio_file)
    #print("xlsr_audio_features ", xlsr_audio_features)
    xlsr_audio_features = np.array(xlsr_audio_features)
    xlsr_audio_features = np.expand_dims(xlsr_audio_features, axis=0)#convert it to 2d array


    predictions = model.predict([X, xlsr_audio_features], verbose=0).squeeze()

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
 
    
def parse_arguments():

    parser = argparse.ArgumentParser(description='Diacritic Restoration Fine-tuning and Inference with Speech and Text Inputs')

    parser.add_argument('--seed_value', default=43)

    parser.add_argument('--with_extra_train', action="store_true")

    parser.add_argument('--aux_dataset_path', default="/l/users/sara.shatnawi/diacritization/")

    parser.add_argument('--dataset_path', default="/l/users/sara.shatnawi/datasets/")

    parser.add_argument('--train_dataset', default="/l/users/sara.shatnawi/datasets/training.txt")

    parser.add_argument('--test_dataset', default="/l/users/sara.shatnawi/datasets/validation.txt")  
   
    parser.add_argument('--train_wav', default="/l/users/sara.shatnawi/datasets/wav/train/") 

    parser.add_argument('--saved_model_path', default="/l/users/sara.shatnawi/test/code/output/fine_tuned_xlsr_Trans_WoC.h5") 
   
    
    parser.add_argument('--batch_size', default=32)#32
    
    parser.add_argument('--vocab_size', default=79)
    
    parser.add_argument('--d_model', default=128)#128
    
    parser.add_argument('--num_heads', default=4)
    
    parser.add_argument('--dff', default=128)
    
    parser.add_argument('--num_blocks_text_encoder', default=2)
    
    parser.add_argument('--num_blocks_speech_encoder', default=2)
    
    
    parser.add_argument('--dropout_rate', default=0.2)
    
    parser.add_argument('--epochs', default=27)#25 #50

    parser.add_argument('-c', '--count')      # option that takes a value

    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    
        
    args = parser.parse_args()

    return args


   
    

  
def predict_with_Audio_features(audio_file,line, model):
    ''' predict test line '''
    X, _ = map_data([line])
    #X_asr, _ = map_data([line_asr])
    xlsr_audio_features  = extract_features(audio_file)
    #print("xlsr_audio_features ", xlsr_audio_features)
    xlsr_audio_features = np.array(xlsr_audio_features)
    xlsr_audio_features = np.expand_dims(xlsr_audio_features, axis=0)#convert it to 2d array


    predictions = model.predict([X, xlsr_audio_features], verbose=0).squeeze()

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
    
    xlsr_model = Wav2Vec2ForCTC.from_pretrained("/l/users/sara.shatnawi/test/code/wav2vec2-large-xlsr-demo/saved_model", from_flax=True).to("cuda") 
    xlsr_processor = Wav2Vec2Processor.from_pretrained("/l/users/sara.shatnawi/test/code/wav2vec2-large-xlsr-demo/saved_processor")
   





    #train with datsset

    # Read the contents of the file into a list (CLarTTS train data)

    lines = open(args.train_dataset, 'r', encoding='utf-8').readlines()
    print("length train_claratts: ", len(lines))
    train_split, val_split =  split_into_training_validation(lines)
    print("len train_split ", len(train_split))
    print("len val_split ", len(val_split))
    

    # Calculate the maximum sequence length in the dataset
    maxlen = max(len(line) for line in train_split + val_split ) 
    max_len=300
    cluster_feature_size = 101
    model = build_modified_model(maxlen, max_len,args.vocab_size,cluster_feature_size,
                             args.d_model,args.num_heads,args.dff,args.num_blocks_text_encoder,
                             args.num_blocks_speech_encoder,args.dropout_rate)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
              metrics=['accuracy'])

    #optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, epsilon=1e-07),
    model.summary()
    
    train_dataset=prep_data(train_split,args.train_wav)
    val_dataset=prep_data(val_split,args.train_wav)
    
    print("len train_dataset ", len(train_dataset))
    print("len val_dataset ", len(val_dataset))    
  
    mapped_train_split=[entry["sentence"] for entry in train_dataset]
    mapped_val_split=[entry["sentence"] for entry in val_dataset]


    train_pathes=[entry["path"] for entry in train_dataset]
    test_pathes=[entry["path"] for entry in val_dataset]
    
    
    xlsr_train_data=[]
    for audio in  train_pathes:
      xlsr_features=extract_features(args.train_wav+audio+".wav")
      xlsr_train_data.append(xlsr_features)

    xlsr_val_data=[]
    for audio in  test_pathes:
      xlsr_features=extract_features(args.train_wav+audio+".wav")
      #print(xlsr_features)
      xlsr_val_data.append(xlsr_features)

    print("len(xlsr_train)", len(xlsr_train_data))
    print("len(xlsr_test)", len(xlsr_val_data))
    print("mapped_train_split" , len(mapped_train_split))
    print("mapped_val_split ", len(mapped_val_split))

    maxlen = max(len(line) for line in train_split + val_split )
    fit_model(model, epochs=args.epochs,batch_size=args.batch_size, train_split=mapped_train_split, val_split=mapped_val_split, seq_data_train
        =xlsr_train_data,seq_data_val=xlsr_val_data, expanded_vocabulary=None)

    # Save the fine-tuned model
    model.save(args.saved_model_path)
