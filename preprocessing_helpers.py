from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

import re

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import defaultdict

import numpy as np
import pandas as pd
from build_vocab import *

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import string

# download necessry dictionnary
nltk.download('wordnet')
nltk.download('punkt')


def truncate_small_words(sentences, minimum_size):
    """Remove words with size smaller than minimum_size from a sentence"""
    truncated_sentences = sentences
    for i in range(len(truncated_sentences)):
        l = []
        for j in range(len(truncated_sentences[i])):
            token = truncated_sentences[i][j]
            if len(token) > minimum_size:
                l.append(token)
                truncated_sentences[i] = l
    return truncated_sentences


def removeNumbers(row):
    """ Removes numbers from a row"""
    text = ''.join([i for i in row if not i.isdigit()])
    return text


def replaceMultiExclamationMark(x):
    """ Replaces repetitions of exclamation marks by the word 'multiexclamation' """
    x = re.sub(r"(\!)\1+", ' multiexclamation ', x)
    return x


def replaceMultiQuestionMark(x):
    """ Replaces repetitions of question marks by the word 'multiquestion' """
    x = re.sub(r"(\?)\1+", ' multiquestion ', x)
    return x


def replaceMultiStopMark(x):
    """ Replaces repetitions of stop marks by the word 'multistop' """
    x = re.sub(r"(\.)\1+", ' multistop ', x)
    return x


def replace_every_punctuation(x):
    """ This function uses the three functions above """
    # Remove the assignment with @
    x = re.sub("@[^\s]*", "", x)
    # Keep the hashtag
    x = re.sub(r"#", " # ", x)
    # Replace 2+ dots with space
    x = re.sub(r'\.{2,}', ' ', x)
    # Replace multiple spaces with a single space
    x = re.sub(r'\s+', ' ', x)
    return replaceMultiExclamationMark(
        replaceMultiQuestionMark(replaceMultiStopMark(x)))


def replace_contractions(x):
    """Replaces contractions by the corresponding text in the pattern"""

    # Builded these progressively
    contraction_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'wanna', 'want to'),
        (r'whi', 'why'),
        (r'wa', 'was'),
        (r"there\'s", "there is"),
        (r"that\'s", "that is"),
        (r'ew(\w+)', 'disgusting'),
        (r'argh(\w+)', 'argh'),
        (r'fack(\w+)', 'fuck'),
        (r'sigh(\w+)', 'sigh'),
        (r'fuck(\w+)', 'fuck'),
        (r'omg(\w+)', 'omg'),
        (r'oh my god(\w+)', 'omg'),
        (r'ladi', 'lady'),
        (r'fav', 'favorite'),
        (r'becaus', 'because'),
        (r'i\'ts', 'it is'),
        (r'ain\'t', 'is not'),
        (r'(\w+)n\'', '\g<1>ng'),
        (r'(\w+)n \'', '\g<1>ng'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would'),
        (r'&', 'and'),
        (r'dammit', 'damn it'),
        (r'dont', 'do not'),
        (r'wont', 'will not'),
    ]

    patterns = [(re.compile(regex_exp, re.IGNORECASE), replacement)
                for (regex_exp, replacement) in contraction_patterns]
    for (pattern, replacement) in patterns:
        (x, _) = re.subn(pattern, replacement, x)
    return x


def filter_small_words(text):
    return " ".join([w for w in text.split() if len(w) > 1])


def replace_by_antonym(word):
    """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """

    antonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None


def replaceNegations(text):
    """ Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """
    i = 0
    length = len(text)
    words = []
    while i < length:
        word = text[i]
        if word == 'not' and i + 1 < length:
            ant = replace(text[i + 1])
            if ant:
                words.append(ant)
                i += 2
                continue
        words.append(word)
        i += 1
    return words


def custom_tokenize_and_replace_negation(x):
    """This function uses the two functions above"""

    tokens = nltk.word_tokenize(x)
    tokens = replaceNegations(tokens)
    x = " ".join(tokens)
    return x


def remove_punctuation(x):
    """ This function removes punctuation """
    translator = str.maketrans('', '', string.punctuation)
    return x.translate(translator)


def replace_elongated(word):
    """ If the word is in the dictionary, keep it else reduce the repetition of letters in it """

    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'

    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:
        return replace_elongated(repl_word)
    else:
        return repl_word


def replace_elongated_words(x):
    """ This functions uses the auxiliary function above """
    finalTokens = []
    tokens = nltk.word_tokenize(x)
    for w in tokens:
        finalTokens.append(replace_elongated(w))
    result = " ".join(finalTokens)
    return result


def SteemSentence(sentence):
    """ This functions reduces inflected (or sometimes derived) words to their stem, base or root form—generally a written word form"""
    ps = PorterStemmer()
    result = []
    words = nltk.word_tokenize(sentence)
    for w in words:
        result.append(ps.stem(w))
    return " ".join(result)


def LemmatizeSentence(sentence):
    """ This functions does the same as steeming but base on context"""
    wordnet_lemmatizer = WordNetLemmatizer()
    result = []
    words = nltk.word_tokenize(sentence)
    for w in words:
        result.append(wordnet_lemmatizer.lemmatize(w, 'v'))
    return " ".join(result)


def process(text, text_processor):
    return " ".join(text_processor.pre_process_doc(text))


def create_text_preprocessor():
    return TextPreProcessor(
        # terms that will be normalized
        normalize=[
            'url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url',
            'date', 'number'
        ],
        # terms that will be annotated
        annotate={
            "hashtag", "allcaps", "elongated", "repeated", 'emphasis',
            'censored'
        },
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words
        spell_correction=True,

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons])

def chosen_preprocessing(text, text_processor):
    """ This functions gathers the different preprocessing tasks above, it is the one that gave us the best results in terms of accuracy  """


    text = replace_contractions(text)
    text = replace_every_punctuation(text)
    text = remove_punctuation(text)
    text = process(text, text_processor)

    return text


################ WORD EMBEDDINGS #################


def create_embedding_matrix(X, pretrained=False):
    """
        Creates the embedding matrix for the word2vec model.

        INPUT:
            X : Multidimensional list  - The traning features
            pretrained                 - True if you want to use the glove pretrained model, False otherwise


        OUTPUT:
        Returns the model trained and the history of the training
    """

    embeddings_index = dict()
    vector_size = 50  # Dimension of each tweet vector

    #1. Load the trained model or use the pre-trained glove model
    if (pretrained == False):

        # Load the trained word2vec model, if you want to see the implementation of the training please refer to build_vocab.py
        model = Word2Vec.load("word2vec.model")
        model.wv.save_word2vec_format('word2vec_dic.txt', binary=False)
        vocabulary_size = len(model.wv.vocab)

        f = open('./word2vec_dic.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))

    else:

        vocabulary_size = 20000
        # Download link of the glove pre trained dictionnary :
        # https://storage.googleapis.com/kaggle-datasets/8560/11981/glove.twitter.27B.50d.txt.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1545508577&Signature=hz8oe4Wa%2Fj%2FteWP4go5ttJle8oeVQpcZfy%2BVfrMETizFMlMcS%2F7TT86TAAQSyOg%2BRvXfTMGs0yN%2Bid1k2DtLyRiYKXrGmpxKEChrmsSabyT7YOCzZiy1bdVyE87833AvaQ30DpopUWi%2B4FbizY185fwYHNC07%2BVsDHFTBsI8yqNTtA6RdAcK%2BHLPEMcBRJZeLz%2BRT8mdvw%2FZzJ2uAbsTTnUzwnHYIDZuy27inV%2BX6Rsyv%2BUMvRDBrPjYhvxEBFWrCErLe%2BC15NH3nXk9vT3R8rXck1YZBhjcExFBPHADScAWgE6%2FESIGLBZ77HQ4WOiusqLg5fOi%2FHg1tOHY7sr2jg%3D%3D
        f = open('./glove.twitter.27B.50d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))

    # 2. Tokenization and padding for the vectors of words
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    data = pad_sequences(sequences, vector_size)

    # 3. Create a weight matrix for words in training docs to feed the neural net
    embedding_matrix = np.zeros((vocabulary_size, vector_size))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return vector_size, vocabulary_size, embedding_matrix, data, tokenizer
