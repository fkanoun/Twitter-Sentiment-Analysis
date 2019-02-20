################# Relevant imports #################
from helpers import *

# For neural networks
from neural_nets import *

# For submission creation
from submission_helper import *

# For ML algorithms
from ML_algorithms import *

import pandas as pd

import sys
################## Get the name of the algorithm to use #################

algo_name = get_chosen_algo_name(sys.argv)

################# Read Data #################

# Define folder path
data_folder = './twitter-datasets'

# Define files' names
train_pos_file_name = 'train_pos_full.txt'
train_neg_file_name = 'train_neg_full.txt'
test_file_name = 'test_data.txt'

# Checks if the files exist and arrange the data in seperate data frames for training and testing
data, data_sub = create_dfs(data_folder, train_pos_file_name,
                            train_neg_file_name, test_file_name)

################# Machine Learning algorithms #################

if(algo_is_ML(algo_name)):
    # 1. Create features and outcomes
    X = data.tweets
    y = data.sentiment

    # 3. Build and train the model
    model = get_ML_model_by_name(algo_name, X, y)

    # 4. Predict the outcomes for test_data
    predictions = model.predict(data_sub.tweets)

################# Neural Nets #################
else:
    # 1. Tokenize the tweets to prepare for neural nets
    max_length = 32
    vocabulary_size = 100000
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(data['tweets'])
    sequences = tokenizer.texts_to_sequences(data['tweets'])

    # 2. Create features and outcomes
    X = pad_sequences(sequences, maxlen=max_length)
    y = list(data.sentiment)

    # 3. Build, train the model and get the history
    model, history = get_model_by_name(
        algo_name, X, y, data.tweets, vocabulary_size=vocabulary_size, max_length=max_length)

    # 4. Predict the outcomes for test_data
    predictions = predict(model, data_sub, tokenizer, max_length, data.tweets, algo_name)


################# Create the submission #################

create_submission(predictions)
