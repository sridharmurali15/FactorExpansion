# README

### A seq2seq, encode-decoder with attention model was built to expand the factors

'requirements.txt' file has all the packages with the versions needed to run main.py

'network.txt' has the model summary of the encoder and decoder.
As per the file, model_1 is the encoder and model_4 is the decoder

'encoder.h5' and 'decoder.h5' are the saved models which are loaded into main.py for prediction

'attention.py' is the custom made attention layer built which is added to the architecture of the seq2seq model for better performance

'train.ipynb' is the notebook used for training the model

'python main.py' will run train the model provided there is a 'train.txt' present in the same folder

'python main.py -t' will use the 'test.txt' file and predict using the built models. The accuracy score will be displayed

