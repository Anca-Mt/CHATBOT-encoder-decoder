import numpy as np
from keras.layers import Embedding
from keras.layers import Input, LSTM, Dense
from keras.models import Model, save_model
from keras.activations import softmax

from keras.callbacks import EarlyStopping

import os
import matplotlib.pyplot as plt

encoder_input_data = np.load('ConvDataset_utils/encoder_input_data.npy')
decoder_input_data = np.load('ConvDataset_utils/decoder_input_data.npy')
decoder_output_data = np.load('ConvDataset_utils/decoder_output_data.npy')

vocabulary_size = 1872+1 # from the previous file

### ENCODER - DECODER MODEL

# Create the input layer and process it.
encoder_in = Input(shape=(encoder_input_data.shape[1],), dtype='int32')
encoder_embedding = Embedding(input_dim=vocabulary_size, output_dim=50, mask_zero=True)(encoder_in) # (None, 22, 50)
encoder_out, state_h, state_c = LSTM(units=300, return_state=True)(encoder_embedding) # (None, 300)

# Discard the outputs of the encoder, but keep the states
encoder_states = [state_h, state_c]

# Define the decoder: using the encoder states
# For decoder_out: return_sequences=True =>   All Hidden States (Hidden State of ALL the time steps), in this case a 3D output
decoder_in = Input(shape=(decoder_input_data.shape[1], ), dtype='int32')
decoder_embedding = Embedding(input_dim=vocabulary_size, output_dim=50, mask_zero=True)(decoder_in)
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

model_name = 'Classic_Model.h5'
model_path = os.path.join('ConvDataset_utils/Model/', model_name)
model.save(model_path)

# Loss plotting
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('ConvDataset_utils/Model/loss_classic.png')
plt.close()

# Accuracy plotting
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('ConvDataset_utils/Model/accuracy_classic.png')
