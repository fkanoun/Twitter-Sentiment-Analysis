from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


import re

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import string

# download necessry dictionnary
nltk.download('wordnet')
nltk.download('punkt')




def truncate_small_words(sentences, minimum_size):
    "Remove words with size smaller than one from a sentence"

    truncated_sentences = sentences
    for i in range(len(truncated_sentences)):
        l = []
        for j in range(len(truncated_sentences[i])):
            token = truncated_sentences[i][j]
            if len(token) > minimum_size:
                l.append(token)
                truncated_sentences[i] = l
    return truncated_sentences


def filter_small_words(text):
    return " ".join([w for w in text.split() if len(w) > 1])


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
    " This function uses the three functions above"
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


def replace(word):
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

    "This function uses the two above"
    tokens = nltk.word_tokenize(x)
    tokens = replaceNegations(tokens)
    x = " ".join(tokens)
    return x


def remove_punctuation(x):
    "This function removes punctuation"
    translator = str.maketrans('', '', string.punctuation)
    return x.translate(translator)


def replace_elongated(word):
    """ If the word is in the dictionary, keep it
        Else reduces the repetition of letters in it """

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
    'This functions uses the auxiliary function above'
    finalTokens = []
    tokens = nltk.word_tokenize(x)
    for w in tokens:
        finalTokens.append(replace_elongated(w))
    result = " ".join(finalTokens)
    return result


def SteemSentce(sentence):
    'Stemming is the process for reducing inflected (or sometimes derived) words to their stem, base or root form—generally a written word form'
    ps = PorterStemmer()
    result = []
    words = nltk.word_tokenize(sentence)
    for w in words:
        result.append(ps.stem(w))
    return " ".join(result)


def LemmatizeSentence(sentence):
    wordnet_lemmatizer = WordNetLemmatizer()
    result = []
    words = nltk.word_tokenize(sentence)
    for w in words:
        result.append(wordnet_lemmatizer.lemmatize(w, 'v'))
    return " ".join(result)


def process(text, text_processor):
    return " ".join(text_processor.pre_process_doc(text))


def chosen_preprocessing(text):

    text_processor = TextPreProcessor(
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
    
    text = replace_contractions(text)
    text = replace_every_punctuation(text)
    text = remove_punctuation(text)
    text = process(text, text_processor)
    
    return text
