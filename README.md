# CHATBOT-encoder-decoder

In this project, I implemented a chatbot using the encoder-decoder algorithm as part of my exploration into the field of deep learning. 
The encoder-decoder architecture has gained widespread attention in the field of natural language processing for tasks such as machine translation and text summarization. 
Furthermore, I aim to demonstrate the effectiveness of the encoder-decoder algorithm for the development of chatbots. 

For training my Q&A chatbot a dataset of human-human question-answer examples was used. The chatbot is capable of answering questions from distinctive domains, 
such as: AI, history, politics, sports, … (The complete dataset was uploaded in ConvDataset directory).

The use of a deep learning encoder-decoder algorithm will enable our chatbot to understand and generate human-like responses to user queries, making it a more effective 
and engaging tool for providing information and assistance in day-to-day life.



<img src="https://user-images.githubusercontent.com/121876169/212167700-1209214d-c641-45f0-9f1a-6a41a240b502.jpg" width= "750" height="350" >



<img src="https://user-images.githubusercontent.com/121876169/212126542-08d54a8a-7d96-462c-8989-ac685394a42d.jpg" width = "800" height = "450">


## Detailed ENCODER/DECODER description ##


<p align="center">
  <img src="https://user-images.githubusercontent.com/121876169/212126650-5a055afa-c045-497c-9f55-6ea25342a5c6.jpg" width = "450" height = "300">
  <img src="https://user-images.githubusercontent.com/121876169/212126687-bf700a13-fd10-4dc8-a182-d1ae5d8e22f8.jpg" width="800" height="650">
</p>

## Model description: ##
*Encoder:* Input Layer, Embedding Layer, LSTM

*Decoder:* Input Layer, Embedding Layer, LSTM + encoder states

*Output:* Dense Layer

<img src="https://user-images.githubusercontent.com/121876169/212162310-97a64975-c8d3-4d14-9de5-ca9f6439de90.jpg" weight = "700" height = "100">

<img src="https://user-images.githubusercontent.com/121876169/212162976-1d5a9f2f-6d60-48fc-ab9c-cfd445edee5f.jpg" weight = "1000" height = "400">


## The EMBEDDING LAYER  ## 

Word embeddings can be thought of as an alternate to one-hot encoding along with dimensionality reduction.
It can understand the context of a word so that similar words have similar embeddings. 
In this project I used Keras embedding layer and GloVe (a pre-trained word embedding). The GloVe file utilized can be downloaded from: https://www.kaggle.com/datasets/watts2/glove6b50dtxt.

<p align ="center">
  <img src="https://user-images.githubusercontent.com/121876169/212167023-2ab4715f-bcce-4802-90df-b33c5e88b5e4.jpg" weight ="800" height ="600">
</p>

*In this particular model, the accuracy and loss metrics prove that modifying the type of the embedding layer doesn't have a prodigious impact on the chatbot's functionality.*
