import numpy as np

from data_processing import format_story_text
from Model import Model
from View import View


class Controller:
    """
    Controller class managing the interaction between the Model and the View
    in the Story Bot application.

    Responsibilities:
        - Initialize the Model with dataset and model paths.
        - Initialize the View with necessary data from the Model.
        - Handle user interactions by linking view events to controller methods.
        - Manage prediction results and coordinate updates between the model and the view.
    """

    model = None
    vue = None
    pred_results = None

    def __init__(self, path_dataset, path_model):
        """
        Initialize the Controller by loading the Model and View, setting up event handlers,
        and starting the GUI main loop.

        Args:
            path_dataset (str): Path to the dataset used to initialize/train the model.
            path_model (str): Path to the pre-trained model or model storage location.
        """
        self.model = Model(path_dataset, path_model)
        self.vue = View(self.model.chatbot.word_indexes)

        self.vue.set_story_button_command(self.load_from_test)
        self.vue.set_answer_button_command(self.get_answer)
        self.vue.master.mainloop()

    def load_from_test(self):
        """
        Load a random story and question from the test dataset,
        format the story text, and update the view components accordingly.
        """
        self.vue.random_index = np.random.randint(0, len(self.model.chatbot.test))

        # Extract raw story and question
        story_list = self.model.chatbot.test[self.vue.random_index][0]
        question_list = self.model.chatbot.test[self.vue.random_index][1]

        # Format story text outside the view
        clean_story = format_story_text(story_list)
        clean_question = ' '.join(question_list)

        # Update the view
        self.vue.display_story(clean_story)
        self.vue.display_question(clean_question)
        self.vue.clear_answer()

    def get_answer(self):
        """
        Retrieve the model's predicted answer for the current story and question,
        then update the view with the answer and its confidence score.

        Process:
            - Fetch the current story and question from the view.
            - Use the model's `predict` method to get the predicted word and confidence.
            - Format the result as: "<word> : certainty = <score>%"
            - Update the view's answer display with this formatted string.
        """
        formatted = ""
        if len(self.vue.get_story()) > 0 and len(self.vue.get_question()) > 0:
            prediction, score = self.model.chatbot.predict(
                self.vue.get_story(), self.vue.get_question()
            )

            formatted = f"{prediction} : certainty = {score:.2f}%"

        return self.vue.answer.set(formatted)


