import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical

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
## vocabulary_size = 1872 (for further use)

# #Extracting the vocabulary by tokenization
#
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(all_data)
# word_index = tokenizer.word_index

word_index = tokenizer.word_index #=> a dictionary of words - to - an index
index_word = tokenizer.index_word

### ONE - HOT ENCODING for vectorizing the data: we will use text_to_sequence
# each word = a vector of integers

encoder_OH_input = tokenizer.texts_to_sequences(encoder_inputs) # a 2D neregular list
maxlen_OH_input_en = max(len(no) for no in encoder_OH_input) # 22
encoder_input_data = pad_sequences(encoder_OH_input, maxlen=maxlen_OH_input_en, padding='post') # => the output is a numpy array (764,22)

decoder_OH_input = tokenizer.texts_to_sequences(decoder_inputs)
maxlen_OH_input_dec = max(len(no) for no in decoder_OH_input) # 60
decoder_input_data = pad_sequences(decoder_OH_input, maxlen=maxlen_OH_input_dec, padding='post') # (764, 60)

decoder_OH_output = tokenizer.texts_to_sequences(decoder_inputs)
# I am removing the BOA from the answers (the word from the beginning because it isn't required when delivering the answers)
for i in range(len(decoder_OH_output)):
    decoder_OH_output[i] = decoder_OH_output[i][1:]
decoder_output_d = pad_sequences(decoder_OH_output, maxlen=maxlen_OH_input_dec, padding='post') # (746,60)
decoder_output_data = to_categorical(decoder_output_d, len(vocabulary)+1) # (764, 60, 1872+1)



np.save('ConvDataset_utils/encoder_input_data', encoder_input_data)
np.save('ConvDataset_utils/decoder_input_data', decoder_input_data)
np.save('ConvDataset_utils/decoder_output_data', decoder_output_data)
