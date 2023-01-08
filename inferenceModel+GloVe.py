import keras.models
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM

'''
REWRITING SOME CODE FROM vocabulary.py
=> for being able to use the previous Tokenizer and the afferent word_index
'''
encoder_inputs = np.load('ConvDataset_utils/encoder_inputs.npy')
encoder_inputs = encoder_inputs.tolist()
decoder_inputs = np.load('ConvDataset_utils/decoder_inputs.npy')
decoder_inputs = decoder_inputs.tolist()

all_data = encoder_inputs + decoder_inputs

vocabulary = []
for sentence in all_data:
    sentence = sentence.split()
    for word in sentence:
        if word not in vocabulary: vocabulary.append(word)
tokenizer = Tokenizer(num_words=len(vocabulary))
tokenizer.fit_on_texts(all_data)

encoder_OH_input = tokenizer.texts_to_sequences(encoder_inputs) # a 2D neregular list
maxlen_OH_input_en = max(len(no) for no in encoder_OH_input) # 22

decoder_OH_input = tokenizer.texts_to_sequences(decoder_inputs)
maxlen_OH_input_dec = max(len(no) for no in decoder_OH_input) # 60
'''
A FUNCTION WHICH TRANSFORMS THE INPUT STRING INTO TOKENS
'''

def input_to_tokens(sentence):
    sentence = sentence.lower()

    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "what is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "can not", sentence)
    # remove all the punctuation
    sentence = re.sub(r"[\,.?:;_'!()\"-]", "", sentence)

    words = sentence.split()
    tokens_list=[]
    for w in words:
        result = tokenizer.word_index.get(w, '')
        if result != '' : tokens_list.append(result)

    input_question = pad_sequences([tokens_list], maxlen=maxlen_OH_input_en, padding='post')
    return input_question

'''
ENCODER for the Inference Model:
    input = questions
    outputs = LSTM states

DECODER for the Inference Model:
    input = LSTM states & the answer input sentence 
'''

model = keras.models.load_model('ConvDataset_utils/Model/Model+GloVe2.h5')
# Extract the layers plus theirs weights from the model

encoder_in = model.input[0]
encoder_out, state_h, state_c = model.get_layer('lstm').output
encoder_states = [state_h, state_c]

decoder_in = model.input[1]
decoder_embedding = model.layers[2].get_output_at(1)
decoder_lstm = model.get_layer('lstm_1')
# decoder_lstm = model.layers[5]
decoder_dense = model.get_layer('dense')


def make_inference_models():
    encoder_model = Model(encoder_in, encoder_states)

    decoder_state_in_h = Input(shape=(300, ), name='anotherInput1')
    decoder_state_in_c = Input(shape=(300, ), name='anotherInput2')
    decoder_states_in = [decoder_state_in_h, decoder_state_in_c]

    decoder_out, decoder_state_out_h, decoder_state_out_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_in)
    decoder_states_out = [decoder_state_out_h, decoder_state_out_c]

    output = decoder_dense(decoder_out)

    decoder_model = Model([decoder_in] + decoder_states_in,
                          [output] + decoder_states_out)

    # decoder_model = Model(inputs=[decoder_in].append(decoder_states_in),
    #                       outputs=[output].append(decoder_states_out))

    return encoder_model, decoder_model

enc_model, dec_model = make_inference_models()

### CREATE THE MAIN CODE FOR ANSWERING THE QUESTIONS

question = 'START'

while question != '0':
    question = input('You are typing: ')
    if question == '0': break
    predictions = enc_model.predict(input_to_tokens(question))
    empty_target_sequence = np.zeros((1,1))
    empty_target_sequence[0, 0] = tokenizer.word_index['boa']

    decoded_answer = ''
    ok = True
    while ok:
        # feed the one word target sequence + predictions from the enc_model
        dec_out , c, h = dec_model([empty_target_sequence]+predictions) # => predictions for the next word

        # take the index of the word with the highest probability
        new_word_index = np.argmax(dec_out[0,-1,:])
        new_word = None

        # append the new word to the decoded_answer
        for word, index in tokenizer.word_index.items():
            if new_word_index == index:
                if word != 'eoa': decoded_answer += '{} '.format(word)
                new_word = word

        # predicting the next word or ending the loop and giving the final result
        if new_word != 'eoa' and len(decoded_answer.split()) <=  maxlen_OH_input_dec:
            empty_target_sequence = np.zeros((1, 1))
            empty_target_sequence[0, 0] = new_word_index
            predictions = [c, h]

        else: ok = False

    print(decoded_answer)