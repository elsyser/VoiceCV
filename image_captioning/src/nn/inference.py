import numpy as np
from keras import Model
from keras.layers import Input
from keras.preprocessing.sequence import pad_sequences

__all__ = [
    'NICInference',
]


class NICInference:
    """Neural Image Captioning - Inference
    
    Implements Inference for the Neural Image Captioning (NIC) Model.
    
    Parameters:
    -----------
    neural_image_captioning : the trained model
    word2idx : dictionary, mapping from vocabulary word to uniques indexes
    
    """

    def __init__(self, neural_image_captioning, word2idx):
        self.neural_image_captioning = neural_image_captioning
        self.word2idx = word2idx
        self.idx2word = {val: key for key, val in word2idx.items()}

        self.inference_model = None
        self.build_inference_model()

    def build_inference_model(self):
        """Builds the Inference Model from the Neural Image Captioning building blocks"""

        # Inputs
        num_hidden_neurons = self.neural_image_captioning.num_hidden_neurons
        h_state_input_1 = Input((num_hidden_neurons[0],), name='h_state_input_1')
        c_state_input_1 = Input((num_hidden_neurons[0],), name='c_state_input_1')
        h_state_input_2 = Input((num_hidden_neurons[1],), name='h_state_input_2')
        c_state_input_2 = Input((num_hidden_neurons[1],), name='c_state_input_2')

        # Token Embeddings
        embedded_seq = self.neural_image_captioning.sequence_decoder.embeddings(
            self.neural_image_captioning.sequence_input
        )
        embedded_seq = self.neural_image_captioning.sequence_decoder.embeddings_dropout(embedded_seq)

        # LSTM decoders
        output_tokens, h_state_1, c_state_1 = self.neural_image_captioning.sequence_decoder.lstm_decoder_1(
            embedded_seq, initial_state=[h_state_input_1, c_state_input_1])
        output_tokens, h_state_2, c_state_2 = self.neural_image_captioning.sequence_decoder.lstm_decoder_2(
            output_tokens, initial_state=[h_state_input_2, c_state_input_2])

        # Dense -> Softmax decoder
        output_tokens = self.neural_image_captioning.sequence_decoder.dense_decoder(output_tokens)
        output_tokens = self.neural_image_captioning.sequence_decoder.softmax_decoder(output_tokens)

        # Build The Model
        self.inference_model = Model(
            [self.neural_image_captioning.sequence_input,
            h_state_input_1, c_state_input_1,
            h_state_input_2, c_state_input_2],

            [output_tokens,
            h_state_1, c_state_1,
            h_state_2, c_state_2]
        )

        return self

    def get_image_embedding(self, image):
        """Takes an image as an input and returns the fixed size feature vector"""
        image_embedding = self.neural_image_captioning.inceptionv3_encoder.encode_image(image)
        final_image_embedding = self.neural_image_captioning.top_image_encoder.model.predict(np.expand_dims(image_embedding, 0))
        return final_image_embedding

    def get_initial_lstm_states(self, image):
        """Takes an image as an input and returns the context vectors (hidden and cell states of a LSTM network)"""

        states_model = Model(
            self.neural_image_captioning.sequence_decoder.image_embedding_input,
            self.neural_image_captioning.sequence_decoder.lstm_decoder_1(
                self.neural_image_captioning.sequence_decoder.image_embedding_input)[1:]
        )

        image_emb = self.get_image_embedding(image)
        states = states_model.predict(image_emb)
        return states + [np.zeros((1, self.neural_image_captioning.num_hidden_neurons[1]))]*2

    def greedy_search(self, image):
        """Greedy Search Inference"""

        # Get the context of the image
        states_values = self.get_initial_lstm_states(image)

        # Start token
        target_seq = np.zeros((1, self.neural_image_captioning.maxlen))
        target_seq[0, 0] = self.word2idx['<START>']

        # Init
        stop_condition = False
        decoded_tokens = []

        while not stop_condition:
            [output_tokens,
            h_state_1, c_state_1,
            h_state_2, c_state_2] = self.inference_model.predict(
                [target_seq] + states_values)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, 0, :])
            sampled_word = self.idx2word[sampled_token_index]
            decoded_tokens.append(sampled_word)

            # Exit condition
            if sampled_word == '<END>' or len(decoded_tokens) > self.neural_image_captioning.maxlen*3:
                stop_condition = True

            # Write sampled token
            target_seq = np.zeros((1, self.neural_image_captioning.maxlen))
            target_seq[0, 0] = sampled_token_index

            states_values = [h_state_1, c_state_1, h_state_2, c_state_2]

        if '<END>' in decoded_tokens:
            decoded_tokens.remove('<END>')

        return ' '.join(decoded_tokens)

    def beam_search(self, image, beam_width=3, alpha=0.7):
        """Beam Search Inference"""

        # Get the context of the image
        states_values = self.get_initial_lstm_states(image)
        states_values = [
            states_values for _ in range(beam_width)
        ]

        # Start token
        target_seq = np.zeros((beam_width, 1, self.neural_image_captioning.maxlen))
        target_seq[:, 0, 0] = self.word2idx['<START>']

        # Init
        candidates = [
            [list(), 0] for _ in range(beam_width)
        ]
        skip_idxs = []

        while len(skip_idxs) < beam_width:
            all_candidates = []
            skip_idxs = []

            for i, row in enumerate(candidates):
                seq, score = row
                if len(seq) >= self.neural_image_captioning.maxlen or (len(seq) > 0 and seq[-1] == self.word2idx['<END>']):
                    skip_idxs.append(i)
                    continue
                
                [output_tokens,
                h_state_1, c_state_1,
                h_state_2, c_state_2] = self.inference_model.predict([target_seq[i]]+states_values[i])
                states_values[i] = [h_state_1, c_state_1, h_state_2, c_state_2]
                word_probabilities = output_tokens[0, 0, :]
                
                for j in range(len(word_probabilities)):
                    all_candidates.append([seq + [j], score + np.log(word_probabilities[j] + 1e-18)])
                    
                if len(row[0]) == 0:
                    break
            
            k = 0
            tmp = candidates
            for idx in skip_idxs:
                candidates[k] = tmp[idx]
                k+=1
            
            if k < beam_width:
                candidates[k:] = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width-k]
            
            for i, candidate in enumerate(candidates):
                target_seq[i] = np.zeros((1, self.neural_image_captioning.maxlen))
                target_seq[i, 0, 0] = candidate[0][-1]
        
        
        scores = [score for _, score in candidates]
        sequences = [[self.idx2word[idx] for idx in seq] for seq, _ in candidates]
        sequences = [seq[:seq.index('<END>')] if '<END>' in seq else seq for seq in sequences]
        
        scores = [1/(len(seq) ** alpha) * score for _, score in zip(sequences, scores)]
        sequences = [' '.join(seq) for seq in sequences]

        return sequences[np.argmax(scores)]

    def predict_logprob(self, image, sent):
        sent = [self.word2idx[word] for word in sent.split()]

        output_probs, _, _, _, _ = self.inference_model.predict(
            [pad_sequences([sent], self.neural_image_captioning.maxlen, padding='post')] + self.get_initial_lstm_states(image)
        )
        output_probs = output_probs[0, :len(sent):, :]

        return np.sum([np.log(probs[idx]) for probs, idx in zip(output_probs, sent)])
