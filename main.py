import sys
from keras.saving.saved_model import load_context
import numpy as np
from typing import Tuple
from attention import AttentionLayer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"
vocab = {' ':0,'<START>':1, '<END>':2}
vocab.update({str(i):i+3 for i in range(10)})
vocab.update({item.strip():i+13 for i,item in enumerate('a, c, h, i, j, k, n, o, s, t, x, y, z'.split(','))})
vocab.update({'sin':26, 'cos':27, 'tan': 28, '+':29, '-':30, '*':31, '/':32, '**':33, '(':34, ')':35})



def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
class CharacterTable(object):
    """Given a set of characters:
    + Encode them based on the vocabulary created
    + Decode the representation to their character output
    """
    def __init__(self, chars):
        # initialize the character table
        self.chars = list(chars)
        self.char_indices = chars
        self.indices_char = {v:k for k,v in self.char_indices.items()}

    def encode(self, C, num_rows):
        # encode the string
        x = np.array([self.char_indices[c] for i, c in enumerate(C)])
        x = np.concatenate((x, np.zeros((1,num_rows-len(x)))), axis=None)
        return x

    def decode(self, x, calc_argmax=True):
        # decode the vectorized string
        return [self.indices_char[c] for c in x]

def train(factors, expansion):
    print('------BUILD VOCAB------')
    global vocab
    ctable = CharacterTable(vocab)

    print('------PREPROCESSING DATA------')
    x = np.array([ctable.encode(item,29) for item in factors])
    y = []
    for i in range(len(expansion)):
        temp = np.append(ctable.encode(expansion[i],len(expansion[i])), ctable.char_indices['<END>'], axis=None)
        temp = np.append(temp, [0]*(28-len(expansion[i])), axis=None)
        y.append(temp)
    y = np.array(y)
    y = np.hstack([np.reshape(np.ones(len(y)),(-1,1)), y])


    vocab = ctable.char_indices

    print('------BUILD ENCODER------')
    # Encoder input
    encoder_inputs = Input(shape=(29,))

    # Embedding layer- i am using 1024 output-dim for embedding you can try diff values 100,256,512,1000
    enc_emb = Embedding(len(vocab), 654)(encoder_inputs)

    # Bidirectional lstm layer
    enc_lstm1 = Bidirectional(LSTM(256,return_sequences=True,return_state=True))
    encoder_outputs1, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm1(enc_emb)

    # Concatenate both h and c
    final_enc_h = Concatenate()([forw_state_h,back_state_h])
    final_enc_c = Concatenate()([forw_state_c,back_state_c])

    # get Context vector
    encoder_states =[final_enc_h, final_enc_c]

    print('------BUILD DECODER------')
    decoder_inputs = Input(shape=(None,))

    # decoder embedding with same number as encoder embedding
    dec_emb_layer = Embedding(len(vocab), 654)
    dec_emb = dec_emb_layer(decoder_inputs)   # apply this way because we need embedding layer for prediction

    # In encoder we used Bidirectional so it's having two LSTM's so we have to take double units(256*2=512) for single decoder lstm
    # LSTM using encoder's final states as initial state
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Using Attention Layer
    attention_layer = AttentionLayer()
    attention_result, attention_weights = attention_layer([encoder_outputs1, decoder_outputs])

    # Concat attention output and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])
    # Dense layer with softmax
    decoder_dense = Dense(len(vocab), activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input)


    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

    # compile model
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint("give Your path to save check points", monitor='val_accuracy')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, early_stopping]

    # Training set
    encoder_input_data = X_train

    # To make same as target data skip last number which is just padding
    decoder_input_data = y_train[:,:-1]

    # Decoder target data has to be one step ahead so we are taking from 1 as told in keras docs
    decoder_target_data =  y_train[:,1:]

    # devlopment set
    encoder_input_test = X_test
    decoder_input_test = y_test[:,:-1]
    decoder_target_test=  y_test[:,1:]

    print('------FIT MODEL------')
    EPOCHS= 3
    history = model.fit([encoder_input_data, decoder_input_data],decoder_target_data,
                        epochs=EPOCHS,
                        batch_size=128,
                        validation_data = ([encoder_input_test, decoder_input_test],decoder_target_test),
                        callbacks= callbacks_list)

    encoder_model = Model(encoder_inputs, outputs = [encoder_outputs1, final_enc_h, final_enc_c])


    # Decoder Inference
    decoder_state_h = Input(shape=(512,)) # This numbers has to be same as units of lstm's on which model is trained
    decoder_state_c = Input(shape=(512,))

    # we need hidden state for attention layer
    decoder_hidden_state_input = Input(shape=(29,512))
    # get decoder states
    dec_states = [decoder_state_h, decoder_state_c]

    # embedding layer
    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=dec_states)


    # Attention inference
    attention_result_inf, attention_weights_inf = attention_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_concat_input_inf = Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, attention_result_inf])

    dec_states2= [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_concat_input_inf)

    # get decoder model
    decoder_model= Model(
                        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_h, decoder_state_c],
                        [decoder_outputs2]+ dec_states2)

   # save model
    model.save("model.h5")
    encoder_model.save("encdoder.h5")
    decoder_model.save("decoder.h5")
    return
        
def get_predicted_sentence(input_seq, ctable, encoder_model, decoder_model):

    vocab = ctable.char_indices
    
    # Encode the input as state vectors.
    enc_output, enc_h, enc_c = encoder_model.predict(input_seq)
  
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = vocab['<START>']
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [enc_output, enc_h, enc_c ])
       
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # convert max index number to marathi word
        sampled_char = ctable.indices_char[sampled_token_index]
        
        # append it to decoded sent
        decoded_sentence += sampled_char
        
        # Exit condition: either hit max length or find stop token.
        if (sampled_char == '<END>' or len(decoded_sentence.split()) >= 29):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        enc_h, enc_c = h, c
    
    return decoded_sentence
    
def predict(factor):

    # define character table
    global vocab
    ctable = CharacterTable(vocab)

    # load encoder model
    enc_model = load_model('encoder.h5', compile=False)
    
    # load decoder model with custom attention layer
    custom_obj = {"CustomLayer": AttentionLayer}
    dec_model = load_model('decoder.h5', custom_objects={'AttentionLayer': AttentionLayer}, compile=False)

    # encode input
    encoded = np.array(ctable.encode(factor,29))
    
    # predict output
    op = ''.join(get_predicted_sentence(np.reshape(encoded, (1,29)), ctable, enc_model, dec_model)[:-5])

    return op
# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):

    factors, expansions = load_file(filepath)
    if 'train' in filepath:
        train(factors, expansions)
        print('DONE TRAINING')
        return
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print('SCORE:', np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")
