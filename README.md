# Q1

This project implements a sequence-to-sequence (seq2seq) model for transliterating text from Latin (English) to Devanagari script using RNN-based architecture. The model is flexible and allows customization of input character embeddings, hidden state dimensions, the choice of RNN cell (SimpleRNN, LSTM), and the number of layers in both the encoder and decoder.

## Features
- **Flexible Architecture**: Allows you to choose the dimension of character embeddings, hidden states, and the type of RNN cell (SimpleRNN, LSTM).
- **Encoder-Decoder Model**: The encoder processes the input text (Latin), and the decoder generates output text (Devanagari), used bengali for experiment.


## Requirements
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Keras

## Installation
1. Clone this repository.
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. **Data Preprocessing**
- The input and target texts are read from a `.tsv` file, where each line contains a pair of English (Latin) and Devanagari text.
- Texts are tokenized and encoded into sequences of integers using Keras' Tokenizer.
- Sequences are padded to a consistent length using `pad_sequences`.

### 2. **Model Architecture**
- **Encoder**: 
  - The encoder is an RNN-based layer (SimpleRNN, LSTM) that reads the input character sequence one token at a time.
  - The encoder outputs a final state which is passed to the decoder.
  
- **Decoder**: 
  - The decoder RNN takes the encoder's final state as its initial state and generates one output character at a time.
  - The decoder outputs the probability distribution over the vocabulary for each time step, using a Dense layer with a softmax activation.

### 3. **Hyperparameters**:
- `embedding_dim`: Dimension of the character embeddings (default: 64).
- `hidden_dim`: Hidden state dimension of the RNN (default: 256).
- `cell_type`: Type of RNN cell (choose from 'RNN', 'LSTM', 'GRU').
- `num_layers`: Number of RNN layers in both encoder and decoder (default: 1).

### 4. **Training the Model**:
- Use the following code to train the model:
    ```python
    model.fit([encoder_input_data, decoder_input_data], 
              decoder_target_data_onehot, 
              batch_size=64, 
              epochs=20, 
              validation_split=0.2)
    ```

### 5. **Evaluating the Model**:
- To evaluate the model on a test set:
    ```python
    test_loss, test_acc = model.evaluate([encoder_test_input_data, decoder_test_input_data],
                                         decoder_test_target_data_onehot)
    print(f"Test Accuracy: {test_acc:.4f}")
    ```

### 6. **Decoding Test Sentences**:
- A decoding function is implemented to generate the Devanagari transliteration for a given Latin input:
    ```python
    decoded_sentence = decode_sequence(input_seq)
    ```

## Functions Used in the Code

- **build_seq2seq_model()**: Defines the seq2seq model architecture, including the encoder and decoder with customizable layers and cell types.
- **decode_sequence()**: Given an input sequence, this function decodes it to produce the corresponding output sequence using the trained model.
- **load_test_data()**: Loads the test data from a file and prepares it for evaluation.


