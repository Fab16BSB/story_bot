import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, Embedding
from tensorflow.keras.layers import add, dot, concatenate
from helpers import get_hyperparameters, create_word_indexes, affine_answer
from data_processing import get_stories, transform_entry, vectorization


class Chatbot:
    """
    A chatbot model based on memory networks for question answering.

    This class loads and processes training and test data from text files,
    initializes embedding layers and builds a memory-based neural network
    for predicting answers to questions based on short stories.

    Attributes:
        train (list): Preprocessed training data in story-question-answer format.
        test (list): Preprocessed test data in story-question-answer format.
        embedding_dim (int): Dimension of the word embeddings.
        dropout_proportion (float): Dropout rate used in the embedding and LSTM layers.
        cells_nb (int): Number of LSTM cells used in the model.
        network (keras.Model): Compiled Keras model ready for training or evaluation.
    """
    train = None
    test = None
    embedding_dim = None
    dropout_proportion = None
    cells_nb = None
    network = None

    def __init__(self, path_textfiles, embedding_dim=64, dropout_proportion=0.3, cells_nb=32):
        """
       Initializes the Chatbot by loading data, setting hyperparameters,
       creating embeddings, and building the model.

       Args:
           path_textfiles (str): A format string with one placeholder (`{}`),
                                 used to load training and test files by
                                 formatting with 'train' and 'test' respectively.
           embedding_dim (int, optional): Size of the word embedding vectors. Defaults to 64.
           dropout_proportion (float, optional): Dropout rate for regularization. Defaults to 0.3.
           cells_nb (int, optional): Number of units in the LSTM layer. Defaults to 32.
        """
        self.embedding_dim = embedding_dim
        self.dropout_proportion = dropout_proportion
        self.cells_nb = cells_nb

        self.train = get_stories(path_textfiles.format('train'))
        self.test = get_stories(path_textfiles.format('test'))

        vocab, self.story_maxlength, self.query_maxlength = get_hyperparameters(self.train, self.test)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1

        self.word_indexes = create_word_indexes(vocab)

        # Création des embeddings
        self.embedding_u = self.__build_embedding_u(vocab_size)
        self.embedding_m = self.__build_embedding_m(vocab_size)
        self.embedding_c = self.__build_embedding_c(vocab_size)

        # build the final model
        self.network = self.__create_model(vocab_size)

    def __build_embedding_u(self, vocab_size):
        """
        Builds the embedding layer for encoding the question (u vector).

        The embedding transforms each word in the question into a dense vector
        of size `embedding_dim`, followed by dropout for regularization.

        Args:
            vocab_size (int): Size of the vocabulary (including padding).

        Returns:
            keras.Sequential: Embedding model for the question input.
        """
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=self.embedding_dim,
                            input_length=self.query_maxlength))
        model.add(Dropout(self.dropout_proportion))
        return model

    def __build_embedding_m(self, vocab_size):
        """
        Builds the embedding layer for memory input vectors (m vectors).

        This embedding maps story words into dense vectors of size `embedding_dim`,
        with dropout applied for regularization.

        Args:
            vocab_size (int): Size of the vocabulary (including padding).

        Returns:
            keras.Sequential: Embedding model for memory encoding (m).
        """
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=self.embedding_dim))
        model.add(Dropout(self.dropout_proportion))
        return model

    def __build_embedding_c(self, vocab_size):
        """
        Builds the embedding layer for contextual memory vectors (c vectors).

        Unlike the other embeddings, this one maps story words into vectors
        of size `query_maxlength`, so that the response can be aligned with
        the question encoding during attention.

        Args:
            vocab_size (int): Size of the vocabulary (including padding).

        Returns:
            keras.Sequential: Embedding model for contextual memory encoding (c).
        """
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=self.query_maxlength))
        model.add(Dropout(self.dropout_proportion))
        return model

    def __create_model(self, vocab_size):
        """
        Builds the memory network model based on the architecture described in
        "End-To-End Memory Networks" (Weston et al., 2015).

        The model takes a story and a question as input, encodes them using
        embedding layers, applies an attention mechanism to compute relevance
        between the story and the question, and outputs a probability distribution
        over the vocabulary representing the predicted answer.

        Args:
            vocab_size (int): Total size of the vocabulary, including padding.

        Returns:
            keras.Model: A compiled Keras model ready for training or inference.
        """

        # Define the input placeholders for the story and the question
        input_sequence = Input((self.story_maxlength,))
        question = Input((self.query_maxlength,))

        # Encode the story using memory embeddings m and c
        input_encoded_m = self.embedding_m(input_sequence)
        input_encoded_c = self.embedding_c(input_sequence)

        # Encode the question using embedding u
        question_encoded = self.embedding_u(question)

        # Compute attention weights between memory and question embeddings
        probabilities = dot([input_encoded_m, question_encoded], axes=(2, 2))
        probabilities = Activation('softmax')(probabilities)

        # Use attention weights to combine with contextual memory embedding
        response = add([probabilities, input_encoded_c])

        # Permute the dimensions to match the expected shape for concatenation
        response = Permute((2, 1))(response)

        # Concatenate the response with the encoded question
        answer = concatenate([response, question_encoded])

        # Process the combined vector through LSTM to generate a final answer representation
        answer = LSTM(self.cells_nb)(answer)

        # Apply dropout for regularization
        answer = Dropout(self.dropout_proportion)(answer)

        # Final dense layer projecting to the vocabulary size
        answer = Dense(vocab_size)(answer)

        # Softmax activation to produce a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        return Model([input_sequence, question], answer)

    def train_model(self):
        """
        Vectorizes the training and test data, compiles the model, and trains it.

        This method performs the following steps:
        - Converts the stories, questions, and answers into vectorized format using the current vocabulary.
        - Compiles the memory network with RMSprop optimizer and categorical crossentropy loss.
        - Trains the model on the training set for a fixed number of epochs.
        - Evaluates performance using a validation set during training.
        """
        inputs_train, queries_train, answers_train = vectorization(self.train, self.word_indexes,
                                                                   self.story_maxlength, self.query_maxlength)
        inputs_test, queries_test, answers_test = vectorization(self.test, self.word_indexes,
                                                                self.story_maxlength, self.query_maxlength)

        # compile the model
        self.network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # train
        self.network.fit([inputs_train, queries_train], answers_train,
                         batch_size=32, epochs=120,
                         validation_data=([inputs_test, queries_test], answers_test))

    def predict(self, story, question):
        """
        Predict the most likely word (code) and its confidence score from a story-question pair.

        Steps:
            1. Transform and vectorize the input story and question.
            2. Use the trained model to predict a probability distribution over vocabulary.
            3. Identify the index with the highest probability (argmax).
            4. Map the index back to the corresponding word from the vocabulary.
            5. Return the word along with its associated confidence score (in %).

        Args:
            story (str): The context or story text.
            question (str): The question related to the story.

        Returns:
            tuple[str, float]: The predicted word (answer) and its confidence score (0-100).
        """
        entry = transform_entry(story, question)
        inputs_test, queries_test = vectorization(
            entry,
            self.word_indexes,
            self.story_maxlength,
            self.query_maxlength,
            entry=True
        )

        raw_pred = self.network.predict([inputs_test, queries_test])[0]  # shape: (vocab_size,)
        val_max = int(np.argmax(raw_pred))
        accuracy = float(raw_pred[val_max] * 100)

        # Retrieve the word corresponding to val_max using while instead of for
        items = list(self.word_indexes.items())
        i = 0
        prediction = ""
        while i < len(items) and not prediction:
            key, val = items[i]
            if val == val_max:
                prediction = key
            i += 1

        return affine_answer(question, prediction), accuracy




