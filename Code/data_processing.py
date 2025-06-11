import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import reduce
import numpy as np



def tokenization(sentence):
    """
    Tokenizes a sentence into tokens, preserving punctuation.

    The sentence is split on non-alphanumeric characters, and tokens are stripped of
    leading/trailing whitespace. Punctuation marks are included as separate tokens.

    Args:
        sentence (str): The input sentence to tokenize.

    Returns:
        list: A list of tokens, including punctuation as separate tokens.

    Example:
        sentence = 'Mary is in the bathroom. Where is the Mary?'
        result = ['mary', 'is', 'in', 'the', 'bathroom', '.', 'where', 'is', 'the', 'mary', '?']
    """
    tokens = re.split(r'(\W+)', sentence)

    clean_tokens = [token.strip() for token in tokens if token is not None and len(token.strip()) > 0]

    return [token for token in clean_tokens]


def extract_stories(lines):
    """
    Parses a list of text lines into a structured format of stories,
    questions, and answers, as found in the bAbI tasks dataset.

    Each line in the input is expected to begin with an incrementing numerical ID.
    The story context resets (a new story begins) when the line ID returns to 1.
    Lines containing a question, answer, and supporting fact ID are identified
    by the presence of a tab character ('\t').

    Note: This function assumes that the input `lines` are already decoded
    UTF-8 strings. It relies on a `tokenization` function (assumed to be
    available in the global scope) to process text into tokens.

    Args:
        lines (list of str): A list of strings, where each string represents
                             a single line from the bAbI tasks dataset.

    Returns:
        list of tuple: A list of tuples, where each tuple represents an
                       extracted story-question-answer triplet. Each tuple
                       has the format `(substory, question, answer)`:
                       - `substory` (list of list of str): A list of tokenized
                         sentences that form the narrative context relevant
                         to the question, up to the point the question is posed.
                       - `question` (list of str): The tokenized question.
                       - `answer` (str): The string representing the answer
                         to the question.

    Example:
        raw_lines = [
        ...     '1 Mary moved to the bathroom.'
        ]

        result = ([['mary', 'moved', 'to', 'the', 'bathroom', '.'], ['john', 'went', 'to', 'the', 'hallway', '.']],
        ['where', 'is', 'mary', '?'], 'bathroom')

    """
    # list of (story, question, answer) tuplets that will be returned
    data = []
    story = []
    for line in lines:
        line = line.strip()

        # get number id and line
        counter, line = line.split(' ', 1)
        counter = int(counter)

        if counter == 1:
            # reset counter
            # new story (see babi tasks description)
            story = []

        if '\t' in line:
            # the line would carry the question, the answer and the supporting line id
            q, a, _ = line.split('\t')
            # tokenize the question
            q = tokenization(q)
            # construct the sub_story (current story up to this point)
            sub_story = [w for w in story if w]  # Filter out empty placeholders
            data.append((sub_story, q, a))
            story.append('')  # Add an empty placeholder to the story list for the fact line

        else:
            # tokenize the new line
            sent = tokenization(line)
            # append it to the current story
            story.append(sent)
    return data


def get_stories(url):
    """
    Loads and processes bAbI task stories from a file, returning each story
    as formatted text along with its tokenized question and answer.

    This function reads the bAbI task file, extracts stories by concatenating sentences,
    formats the concatenated story into a clean string (with punctuation and line breaks),
    and pairs it with its corresponding question tokens and answer token.

    Args:
        url (str): Path to the bAbI dataset text file.

    Returns:
        list of tuples: Each tuple contains:
            - story_text (str): The formatted story string.
            - question_tokens (list of str): Tokens of the question.
            - answer (str): The answer token.

    Example:
        Given the input lines:
            "1 Mary moved to the bathroom.\n"
            "2 John went to the hallway.\n"
            "3 Where is Mary?	bathroom	1\n"

        The function returns:
            [("mary moved to the bathroom.
             john went to the hallway.",
              ['where', 'is', 'mary', '?'],
              'bathroom')]
    """
    with open(url, 'r', encoding='utf-8') as f:
        raw_data = extract_stories(f.readlines())

    # flatten each story (list of sentence token lists) into one token list
    flat_story = lambda story_data: reduce(lambda acc, sent: acc + sent, story_data, [])

    processed = []
    for story_tokens, question_tokens, answer in raw_data:
        # flatten and then format the story text
        tokens = flat_story(story_tokens)
        processed.append((tokens, question_tokens, answer))

    return processed


def format_story_text(story_list):
    """
    Format the story list into a clean, readable string.

    Args:
        story_list (list of str): List of words forming the story.

    Returns:
        str: Formatted story text with proper punctuation and line breaks.
    """
    story_sequence = ' '.join(story_list)
    story_sequence = re.sub(r" \. ", ".\n", story_sequence)
    story_sequence = re.sub(r" \.", ".", story_sequence)
    return story_sequence


def transform_entry(story_entry, question_entry):
    """
    Transforms a single story entry and a question entry into a formatted list.

    This function processes a multi-line story string and a single question string,
    tokenizing both. It then flattens the tokenized story into a single list
    of tokens and pairs it with the tokenized question.

    Args:
        story_entry (str): A string containing the story, with sentences separated
                           by newline characters. Each line will be tokenized.
        question_entry (str): A string containing the question to be tokenized.

    Returns:
        list of tuple: A list containing a single tuple. The tuple format is
                       `(flat_story_tokens, question_tokens)`:
                       - `flat_story_tokens` (list of str): All tokens from the
                         `story_entry`, merged into a single list in order.
                       - `question_tokens` (list of str): All tokens from the
                         `question_entry`.
    """
    story = [tokenization(line.strip()) for line in story_entry.split('\n')]
    question = tokenization(question_entry)

    # merge all sentences of a single story in a signle list
    flat_story = lambda story: reduce(lambda x, y: x + y, story)
    data = [(flat_story(story), question)]
    return data


def vectorization(data, word_indexes, story_maxlen, query_maxlen, entry=False):
    """
    Converts textual stories, questions, and answers into numerical vectors suitable for model input.

    Each word in the input data is mapped to its corresponding index using the `word_indexes` dictionary.
    The resulting sequences are padded with zeros to ensure consistent lengths.

    Args:
        data (list):
            - If entry=False: list of tuples (story, question, answer), where story and question are lists of tokens,
              and answer is a single token (string).
            - If entry=True: list of tuples (story, question) without answers, typically user input.
        word_indexes (dict): Dictionary mapping tokens (words) to their integer indices.
        story_maxlen (int): Maximum length for story sequences (used for padding).
        query_maxlen (int): Maximum length for question sequences (used for padding).
        entry (bool, optional):
            - False (default): data includes answers, which are one-hot encoded.
            - True: data does not include answers (e.g., for prediction), so only story and query vectors are returned.

    Returns:
        If entry=False:
            tuple of (padded_story_vectors, padded_query_vectors, one_hot_answer_vectors)
        If entry=True:
            tuple of (padded_story_vectors, padded_query_vectors)

    Notes:
        - Padding is done with zeros to the right length.
        - The answer vectors are one-hot encoded over the vocabulary size plus one (index 0 reserved).
    """
    story_vectors = []
    query_vectors = []
    targets = []

    if not entry:
        for story, query, answer in data:
            # each story is transformed to a numerical vector
            vect = [word_indexes[w] for w in story]
            story_vectors.append(vect)

            # each query is transformed to a numerical vector
            vect = [word_indexes[w] for w in query]
            query_vectors.append(vect)

            # answer vector
            # index 0 is reserved
            y = np.zeros(len(word_indexes) + 1)
            y[word_indexes[answer]] = 1
            targets.append(y)

        # padding of the resulted vectors
        result = (pad_sequences(story_vectors, maxlen=story_maxlen),
                  pad_sequences(query_vectors, maxlen=query_maxlen), np.array(targets))
    else:
        for story, query in data:
            # each story is transformed to a numerical vector
            vect = [word_indexes[w] for w in story]
            story_vectors.append(vect)

            # each query is transformed to a numerical vector
            vect = [word_indexes[w] for w in query]
            query_vectors.append(vect)

        # no answer vector
        # padding of the resulted vectors
        result = (pad_sequences(story_vectors, maxlen=story_maxlen),
                  pad_sequences(query_vectors, maxlen=query_maxlen))
    return result
