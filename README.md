# CHATBOT-encoder-decoder

In this project, I implemented a chatbot using the encoder-decoder algorithm as part of my exploration into the field of deep learning. 
The encoder-decoder architecture has gained widespread attention in the field of natural language processing for tasks such as machine translation and text summarization. 
Furthermore, I aim to demonstrate the effectiveness of the encoder-decoder algorithm for the development of chatbots. 

For training my Q&A chatbot a dataset of human-human question-answer examples was used. The chatbot is capable of answering questions from distinctive domains, 
such as: AI, history, politics, sports, … (The complete dataset was uploaded in ConvDataset directory).

The use of a deep learning encoder-decoder algorithm will enable our chatbot to understand and generate human-like responses to user queries, making it a more effective 
and engaging tool for providing information and assistance in day-to-day life.


![Encoder_Decoder](https://user-images.githubusercontent.com/121876169/212126542-08d54a8a-7d96-462c-8989-ac685394a42d.jpg)

<p align="center"> 
  <b> Detailed ENCODER/DECODER description </b>
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/121876169/212126650-5a055afa-c045-497c-9f55-6ea25342a5c6.jpg" width = "450" height = "300">
  <img src="https://user-images.githubusercontent.com/121876169/212126687-bf700a13-fd10-4dc8-a182-d1ae5d8e22f8.jpg" width="800" height="650">
</p>

## Model description: ##
*Encoder:* Input Layer, Embedding Layer, LSTM

*Decoder:* Input Layer, Embedding Layer, LSTM + encoder states

*Output:* Dense Layer
