from tkinter import Tk, Menu, PhotoImage, Label, Frame, Text, RAISED, StringVar, BOTTOM, END, Button, \
    LEFT, RIGHT, Entry, X


class View:
    """
    A GUI class for interacting with the Story Bot application.

    This class manages the main window, including input fields for stories and questions,
    buttons to trigger actions, and displays answers. It also stores test stories and
    a word-to-index mapping for processing user input.

    Attributes:
        story_text (Text): Text widget for entering the story.
        question (StringVar): Variable storing the current question input.
        story_button (Button): Button to submit the story.
        answer_button (Button): Button to get the answer.
        master (Tk): The main Tkinter window.
        random_index (int): Index used to select a random test story.
        answer (StringVar): Variable storing the chatbot's answer.
    """
    story_text = None
    question = None
    story_button = None
    answer_button = None
    master = None
    random_index = None
    answer = None

    def __init__(self, word_idx):
        """
        Initialize the View with word index mapping and test stories.

        Args:
            word_idx (dict): Dictionary mapping words to their numerical indices.

        Initializes:
            - The main Tkinter window.
            - The random index to 0.
            - Calls init_window() to set up the GUI.
        """
        self.word_idx = word_idx

        self.random_index = 0
        self.master = Tk()

        self.init_window()

    def init_window(self):
        """
        Initialize the main GUI window for the Story Bot application.

        This method sets up the window title, menu bar with informational items,
        background image, and main UI frames including the story display, question input,
        and answer display. Also initializes action buttons.
        """
        self.master.title("===== Story Bot =====")
        self.master.geometry("500x500")  # Set a fixed window size
        self.master.resizable(False, False)  # Make the window non-resizable
        self.__create_menu()
        self.__create_story_frame()
        self.__create_question_frame()
        self.__create_action_buttons()
        self.__create_answer_frame()

    def __create_menu(self):
        """Create the menu bar with an 'About' section and related informational items."""
        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        # Create "A propos" menu
        a_propos_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="About", menu=a_propos_menu)
        a_propos_menu.add_command(label="Project developed by:")
        a_propos_menu.add_command(label="Zeineb Ghrib / Nadia Essfini / Poirier Fabien")
        a_propos_menu.add_command(label="Python / Tkinter")
        a_propos_menu.add_command(label="2018-2019")

    def __create_story_frame(self):
        """Create and pack the frame and widgets to display the story text."""
        story_frame = Frame(self.master, bg='old lace', relief=RAISED, bd=2)
        story_frame.pack(pady=10, padx=20, fill=X)

        story_label_frame = Frame(story_frame, bg='old lace')
        story_label_frame.pack(fill=X)

        story_label = Label(story_label_frame, text='Story :', bg='old lace', font=('TkDefaultFont', 10, 'bold'))
        story_label.pack(side=LEFT, padx=5, pady=5)

        # A text widget for the story
        self.story_text = Text(story_frame, height=10, width=45, wrap='word', bd=2, relief="sunken", padx=5, pady=5)
        self.story_text.pack(expand=True, fill=X, padx=5, pady=5)

    def __create_question_frame(self):
        """Create and pack the frame and variable to input the question."""
        question_frame = Frame(self.master, borderwidth=2, bg='old lace', relief=RAISED)
        # Reduced pady from 10 to 5. Further reduced to 2 for tighter spacing.
        question_frame.pack(pady=2, padx=20, fill=X)

        question_label = Label(question_frame, text='Question :', bg='old lace')
        question_label.pack(side=LEFT, padx=5, pady=5)

        self.question = StringVar()
        question_entry = Entry(question_frame, textvariable=self.question, width=35, relief="sunken", bd=2)
        question_entry.pack(side=LEFT, fill=X, expand=True, padx=5, pady=5)

    def __create_answer_frame(self):
        """Create and pack the frame and variable to display the answer, positioned slightly above the bottom."""
        answer_frame = Frame(self.master, borderwidth=2, bg='old lace', relief=RAISED)
        # Reduced pady from 10 to 5. Further reduced to 2 for tighter spacing.
        answer_frame.pack(side=BOTTOM, pady=2, padx=20, fill=X)

        answer_label_text = Label(answer_frame, text='Answer :', bg='old lace')
        answer_label_text.pack(side=LEFT, padx=5, pady=5)

        self.answer = StringVar()
        answer_display_label = Label(answer_frame, textvariable=self.answer, bg='old lace', width=40, height=2,
                                     anchor='w')
        answer_display_label.pack(side=LEFT, fill=X, expand=True, padx=5, pady=5)

    def __create_action_buttons(self):
        """Create and pack the buttons for loading test data and submitting the question."""
        button_frame = Frame(self.master)
        button_frame.pack(side=BOTTOM, pady=10, padx=20, fill=X)

        self.story_button = Button(button_frame, text="load from test", relief=RAISED, bd=2,
                                   font=('TkDefaultFont', 10), padx=10, pady=5)
        self.story_button.pack(side=LEFT, padx=(0, 5), pady=5)

        self.answer_button = Button(button_frame, text="Answer", relief=RAISED, bd=2,
                                    font=('TkDefaultFont', 10), padx=10, pady=5)
        self.answer_button.pack(side=RIGHT, padx=(5, 0), pady=5)

    def display_story(self, text):
        """
        Display the given story text in the story text widget.

        Args:
            text (str): The story to be shown in the GUI.
        """
        self.story_text.delete('1.0', END)
        self.story_text.insert('1.0', text)

    def display_question(self, text):
        """
        Set the input field for the question with the given text.

        Args:
            text (str): The question to be displayed in the GUI.
        """
        self.question.set(text)

    def clear_answer(self):
        """
        Clear the currently displayed answer from the answer field.
        """
        self.answer.set("")

    def get_story(self):
        """
        Retrieve the full text content from the story input field.

        Returns:
            str: The complete story text entered by the user.
        """
        return self.story_text.get('1.0', END).strip()

    def get_question(self):
        """
        Retrieve the current text from the question input variable.

        Returns:
            str: The question text entered by the user.
        """
        return self.question.get()

    def set_story_button_command(self, command):
        """
        Set the callback function to be executed when the 'load from test' button is clicked.

        Args:
            command (callable): The function to be called on button click.
        """
        self.story_button.config(command=command)

    def set_answer_button_command(self, command):
        """
        Set the callback function to be executed when the 'answer' button is clicked.

        Args:
            command (callable): The function to be called on button click.
        """
        self.answer_button.config(command=command)
