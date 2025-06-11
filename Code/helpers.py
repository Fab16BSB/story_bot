import os
import string
import sys
import nltk

try:
    from nltk.corpus import words

    _ = words.words()
except LookupError:
    print("\n⚠️ The NLTK corpus 'words' is not installed.")
    answer = input("Would you like to install it now? (Y/n): ").strip().lower()

    if answer == 'y' or answer == '':
        print("➡️ Installing the corpus...\n")
        nltk.download('words')
        try:
            from nltk.corpus import words

            _ = words.words()
            print("✅ Corpus successfully installed.\n")
        except LookupError:
            print("\n❌ Installation failed. Please try manually:\n   python -m nltk.downloader words")
            sys.exit(1)
    else:
        print("\n❌ The 'words' corpus is required to proceed.")
        print("Please run the following command manually:")
        print("   python -m nltk.downloader words")
        sys.exit(1)

ENGLISH_WORDS = set(w.lower() for w in words.words())


def get_vocab(train, test):
    """
    Builds the vocabulary set from the training and test datasets.

    This function extracts all unique words from the stories, questions, and answers
    in both the training and test sets. The resulting vocabulary is sorted alphabetically.

    Args:
        train (list): A list of training samples, where each sample is a tuple (story, question, answer).
        test (list): A list of test samples with the same format as the training samples.

    Returns:
        list: A sorted list of unique words from the combined datasets.
    """
    vocab = set()
    for story, q, answer in train + test:
        new_story = story + q + [answer]
        vocab |= set(new_story)
    return sorted(vocab)


def create_word_indexes(vocab):
    """
    Creates a mapping from words to unique integer indices.

    This function assigns each word in the vocabulary a unique index, starting from 1.
    Index 0 is typically reserved for padding purposes in sequence models.

    Args:
        vocab (list): A list of unique words (typically sorted).

    Returns:
        dict: A dictionary mapping each word to a unique integer index.
    """
    word_indexes = {}
    for index, symbole in enumerate(vocab):
        word_indexes[symbole] = index + 1
    return word_indexes


def get_hyperparameters(train, test):
    """
    Computes key hyperparameters from the training and test datasets.

    This function:
    - Builds the global vocabulary from both datasets.
    - Computes the maximum length of the stories.
    - Computes the maximum length of the questions.

    These values are useful for setting input dimensions in the neural network.

    Args:
        train (list): A list of training samples, each as a tuple (story, question, answer).
        test (list): A list of test samples with the same structure.

    Returns:
        tuple: A tuple containing:
            - vocab (list): Sorted list of unique words in the datasets.
            - story_maxlen (int): Maximum number of tokens in any story.
            - query_maxlen (int): Maximum number of tokens in any question.
    """
    vocab = get_vocab(train, test)

    # the overall dataset
    whole_data = train + test

    # apply len to all stories and get the max of it
    story_maxlen = max(map(len, (x for x, _, _ in whole_data)))

    # do the same to queries
    query_maxlen = max(map(len, (x for _, x, _ in whole_data)))

    return vocab, story_maxlen, query_maxlen


def get_new_modelname(working_path):
    """
    Generates a new model name by incrementing the highest existing model number in the directory.

    The function looks for JSON files in the given directory with names like 'model1.json',
    'model2.json', etc., and returns a new model name with the next incremented number.
    If no such files exist, it returns 'model1'.

    Args:
        working_path (str): Path to the directory containing model files.

    Returns:
        str: The new model name without the file extension (e.g., 'model3').
    """
    listdir = os.listdir(working_path)
    model_names = sorted([name for name in listdir if 'json' in name])
    if len(model_names) == 0:
        model_name = "model1"
    else:
        last_name = model_names[-1]
        last_id = int(last_name.split('.')[0][-1])
        model_name = "model" + str(last_id + 1)

    return model_name


def affine_answer(question, prediction):
    """
    Refine the model’s raw prediction by prepending a contextual phrase based on the question type
    and inserting the relevant person’s name.

    - For "where" questions, returns:
          "<optional reversed context>in the <prediction>"
    - For "why" questions, returns:
          "Because <PersonName> is <prediction>"
    - Otherwise, returns the raw prediction capitalized.

    Args:
        question (str): The input question string (e.g. "Where is Mary?").
        prediction (str): The raw answer predicted by the model (e.g. "bathroom").

    Returns:
        str: The refined, human-readable answer.
    """
    # Lowercase prediction for consistent formatting
    reponse = prediction
    q_lower = question.lower()

    # Try to determine person’s name
    person = list(extract_person(question))

    if len(person):
        if q_lower.startswith("where"):
            reponse = f"{person[0]} is in the {prediction.lower()}"

    elif q_lower.startswith("why"):
        reponse = f"Because he/she is {prediction.lower()}"

    elif q_lower.startswith("what"):
        reponse = f"The {prediction.lower()}{question.lower().replace("what", "").replace("?", "")}"

    return reponse


def extract_person(text):
    """
    Extract probable entity names (e.g., people or places) from a sentence by identifying
    capitalized words not commonly found in the English dictionary.

    This function uses two heuristics:
    1. It first tries to extract capitalized words (excluding the first word) which are often proper nouns.
    2. If none are found, it computes the difference between all words (punctuation removed) and a set of known English words.

    Args:
        text (str): The input sentence or question.

    Returns:
        set: A set of candidate entity names not found in the English dictionary.
    """
    # Heuristic 1: Capitalized words, excluding the first word (common in questions like "What is ...")
    unknown_words = {word for i, word in enumerate(text.split()) if i != 0 and word.istitle()}

    # Fallback: Difference with English dictionary
    if not unknown_words:
        clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = set(clean_text.split())
        unknown_words = tokens - ENGLISH_WORDS

    return unknown_words
