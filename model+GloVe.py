import numpy as np
from keras.layers import Embedding
from keras.layers import Input, LSTM, Dense
from keras.models import Model, save_model
from keras.activations import softmax

from keras.callbacks import EarlyStopping

import os
import matplotlib.pyplot as plt

'''
Extract again the word_index because it is necessary for the glove embedding layer  
'''
from keras.preprocessing.text import Tokenizer

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
vocabulary_size = 1872 +1

word_index = tokenizer.word_index #=> a dictionary of words - to - an index
print('The word_index is obtained')

### IMPORTING THE GLOVE

embedding_dict = {}
with open('glove.6B_/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = coefs
    f.close()

print('Glove Loaded!')  # => 400000 words in this GloVe version


### CREATE THE EMBEDDING MATRIX
embedding_dimension = 50

def embedding_matrix_creator(embedding_dim, w_i):
    embedding_matix = np.zeros((len(w_i)+1, embedding_dim))
    for word, i in w_i.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None: # the word is the embedding_dict
            embedding_matix[i] = embedding_vector
    return embedding_matix

embedding_matrix = embedding_matrix_creator(embedding_dimension, word_index)

embed_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dimension, trainable=True, mask_zero=True)
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

### CREATE THE ENCODER - DECODER MODEL

encoder_input_data = np.load('ConvDataset_utils/encoder_input_data.npy')
decoder_input_data = np.load('ConvDataset_utils/decoder_input_data.npy')
decoder_output_data = np.load('ConvDataset_utils/decoder_output_data.npy')


# Create the input layer and process it.
encoder_in = Input(shape=(encoder_input_data.shape[1],), dtype='int32')
encoder_embedding = embed_layer(encoder_in)
encoder_out, state_h, state_c = LSTM(units=300, return_state=True)(encoder_embedding) # (None, 300)

# Discard the outputs of the encoder, but keep the states
encoder_states = [state_h, state_c]

# Define the decoder: using the encoder states
# For decoder_out: return_sequences=True =>   All Hidden States (Hidden State of ALL the time steps), in this case a 3D output
decoder_in = Input(shape=(decoder_input_data.shape[1], ), dtype='int32')
decoder_embedding = embed_layer(decoder_in)
decoder_out, _, _ = LSTM(units=300, return_state=True, return_sequences=True)(decoder_embedding, initial_state=encoder_states) # (None, 300)

output = Dense(vocabulary_size, activation=softmax)(decoder_out)


model = Model([encoder_in, decoder_in], output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


BATCH_SIZE = 32
EPOCHS = 200

callbacks_list = [
    EarlyStopping(monitor='accuracy', patience=5),
]

history = model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                    callbacks=callbacks_list)

model_name = 'Model+GloVe.h5'
model_path = os.path.join('ConvDataset_utils/Model/', model_name)
model.save(model_path)

# Loss plotting
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('ConvDataset_utils/Model/loss_glove.png')
plt.close()

# Accuracy plotting
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('ConvDataset_utils/Model/accuracy_glove.png')
