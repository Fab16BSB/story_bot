import os
from tensorflow.keras.models import model_from_json
from Chatbot import Chatbot
from helpers import get_new_modelname


class Model:
    """
    Wrapper class that manages a Chatbot instance, including loading a pre-trained model
    if available or training a new one otherwise.

    Attributes:
        chatbot (Chatbot): Instance of the Chatbot class used for training and inference.
    """
    chatbot = None

    def __init__(self, path_textfiles, file_name):
        """
        Initializes the Model by creating a Chatbot instance using the given dataset path.
        Checks if a saved model file exists with the given file_name; if yes, loads the model,
        otherwise trains the chatbot model and saves it.

        Args:
           path_textfiles (str): Path pattern to the training and test text files.
           file_name (str): Base file name to load/save the model files (without extension).
        """
        self.chatbot = Chatbot(path_textfiles)

        if os.path.isfile(file_name + '.json'):
            self.load(file_name)
        else:
            self.chatbot.train_model()
            self.save()

    def save(self, file_path="../Network", model_extension=".json", weights_extension=".weights.h5"):
        """
        Save the current chatbot model architecture and weights to disk.

        Args:
            file_path (str, optional): Directory path where the model and weights files will be saved.
                                       Defaults to "../Network".
            model_extension (str, optional): File extension for the model architecture file.
                                             Defaults to ".json".
            weights_extension (str, optional): File extension for the model weights file.
                                               Defaults to ".weights.h5".

        Process:
            - Creates the directory if it does not exist.
            - Generates a new model name to avoid overwriting existing files.
            - Serializes the model architecture to a file with the specified model extension.
            - Saves the model weights to a file with the specified weights extension.
        """

        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        model_name = get_new_modelname(file_path)

        # Serialize the model architecture to JSON format
        model_json = self.chatbot.network.to_json()
        with open(os.path.join(file_path, model_name + model_extension), "w") as json_file:
            json_file.write(model_json)

        # Save the model weights to an HDF5 file
        self.chatbot.network.save_weights(os.path.join(file_path, model_name + weights_extension))

    def load(self, file_path, model_extension=".json", weights_extension=".weights.h5"):
        """
        Load a saved model architecture and its weights from disk into the chatbot's network.

        Args:
            file_path (str): Base file path (without extension) where the model and weights files are stored.
            model_extension (str, optional): Extension of the model architecture file (default is ".json").
            weights_extension (str, optional): Extension of the model weights file (default is ".weights.h5").

        Process:
            - Reads the model architecture from the JSON file.
            - Loads the model architecture into the chatbot's network.
            - Loads the corresponding weights into the model.

        Raises:
            IOError: If the model or weights files cannot be found or opened.
            ValueError: If the loaded model JSON is invalid.
        """
        json_file = open(file_path + model_extension, 'r')
        model_json = json_file.read()
        json_file.close()

        self.chatbot.network = model_from_json(model_json)

        # load weights into new model
        self.chatbot.network.load_weights(file_path + weights_extension)
