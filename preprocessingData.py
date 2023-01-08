import yaml
from yaml.loader import SafeLoader
import re
import numpy as np
import pandas as pd

### 1. EXTRACT THE DATA ; SEPARATE THE QUESTIONS AND ANSWERS

# Open the file and load the file

name_of_file = ['ai.yml', 'botprofile.yml', 'computers.yml', 'emotion.yml', 'food.yml', 'gossip.yml', 'greetings.yml',
                'health.yml', 'history.yml', 'humor.yml', 'literature.yml', 'money.yml', 'movies.yml', 'politics.yml',
                'psychology.yml', 'science.yml', 'sports.yml', 'trivia.yml']

q_a = []
for name in name_of_file:
    with open('ConvDataset/' + name) as f:
        data = yaml.load(f, Loader=SafeLoader)
        que_ans = data['conversations']  # ex: que_ans[0] represents a conversation
        q_a = q_a + que_ans


questions = []
answers = []

for conv in q_a:
    if len(conv) == 2 :
        questions.append(conv[0])
        answers.append(conv[1])
    elif len(conv) > 2 :
        for i in range(len(conv)-1):
            questions.append(conv[0])
            answers.append(conv[i+1])

# In the answers list there are some answers which are dictionaries, not strings.'
i=0
while i< len(answers):
    if type(answers[i]) == dict:
        answers.pop(i)
        questions.pop(i)
        i -= 1
    i +=1

### 2. PREPROCESSING THE DATA

# Clean the data

def clean_data(sentence):
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
    return sentence

clean_questions = []
clean_answers = []

for data in questions:
    clean_questions.append(clean_data(data))

for data in answers:
    clean_answers.append(clean_data(data))

# for the decoder add <START> and <END>

START = "boa "
END = " eoa"

decoder_inputs = []

for answer in clean_answers:
    dec_answer = START + answer + END
    decoder_inputs.append(dec_answer)

encoder_inputs = clean_questions

encoder_inputs_np = np.array(encoder_inputs)
np.save('ConvDataset_utils/encoder_inputs', encoder_inputs_np)
pd.DataFrame(encoder_inputs_np).to_csv('ConvDataset_utils/encoder_inputs.csv')

decoder_inputs_np = np.array(decoder_inputs)
np.save('ConvDataset_utils/decoder_inputs', decoder_inputs_np)
pd.DataFrame(decoder_inputs_np).to_csv('ConvDataset_utils/decoder_inputs.csv')

