import os
import sys
from preprocessing_helpers import *

def get_chosen_algo_name(argv):
    if (len(sys.argv) > 1):
        algo_name = sys.argv[1]
    else:
        print('You did not provide an algo to use, we will use CNN + GRU by default')
        algo_name = 'cnn_gru'


    # Define implemented algorithms
    machine_learning = ['svm_l2', 'svm_l1', 'logistic', 'ridge', 'multinomial', 'bernoulli', 'voting']
    neural_nets = ['cnn_gru', 'lstm', 'cnn_lstm', 'bidir_gru', 'embeddings']
    all_algorithms = machine_learning + neural_nets

    if (algo_name not in all_algorithms):
        print('You have provided a wrong algorithm name')
        print('Please choose one of these')
        print('For Machine Learning:', machine_learning)
        print('For Neural Networks:', neural_nets)
        print('Shutting down')
        sys.exit()

    return algo_name


def algo_is_ML(algo_name):
    machine_learning = ['svm_l2', 'svm_l1', 'logistic', 'ridge', 'multinomial', 'bernoulli', 'voting']
    return algo_name in machine_learning

def algo_is_NN(algo_name):
    neural_nets = ['cnn_gru', 'lstm', 'cnn_lstm', 'bidir_gru', 'embeddings']
    return algo_name in neural_nets


def data_exists(folder_path, train_name_pos, train_name_neg, test_name):
    """
    Checks if data exists in the corresponding folder

    INPUT:
        folder_path             - The path to the data_folder
        train_name_pos :        - The name of the positive tweets file
        train_name_neg :        - The name of the negative tweets file
        test_name :             - The name of the testing tweets file for submission

    """
    # Using os.path.join to avoid being OS dependent
    pos_file_path = os.path.join(folder_path, train_name_pos)
    neg_file_path = os.path.join(folder_path, train_name_neg)
    test_path = os.path.join(folder_path, test_name)

    if os.path.exists(pos_file_path) and os.path.exists(
            neg_file_path) and os.path.exists(test_path):
        return pos_file_path, neg_file_path, test_path

    else:
        print(
            "You don't have the files in the correct folder please check and run again\nfolder name : twitter-datasets\ntrain_pos_file_name = train_pos_full.txt\ntrain_neg_file_name = train_neg_full.txt\ntest_file_name = test_data.txt\nProgram exiting..."
        )
        sys.exit()
        return False


def create_dfs(data_folder, train_pos_file_name, train_neg_file_name, test_file_name):
    """
    Create the data frame for training and testing

    INPUT:
        data_folder             - The path to the data_folder
        train_pos_file_name :   - The name of the positive tweets file
        train_neg_file_name :   - The name of the negative tweets file
        test_file_name :        - The name of the testing tweets file for submission


    OUTPUT:
        Returns (dataframe_train, dataframe_test)
    """
    # Check if all the files exist
    if (data_exists(data_folder, train_pos_file_name, train_neg_file_name,
                    test_file_name)):

        positive_path, negative_path, test_path = data_exists(
            data_folder, train_pos_file_name, train_neg_file_name, test_file_name)

    lines_positive = [line.rstrip('\n') for line in open(positive_path)]
    lines_negative = [line.rstrip('\n') for line in open(negative_path)]
    lines_test = [line.rstrip('\n') for line in open(test_path)]

    #### Preprocessing tool ####
    text_preprocessor = create_text_preprocessor()

    #### TEST DATA ####
    # Create dataFrame from test tweets
    data_sub = pd.DataFrame({"tweets": lines_test})

    print('Preprocessing on the testing Data-Set ')
    # Remove leading digits from tweets
    data_sub['tweets'] = data_sub['tweets'].apply(lambda x: x.split(',', 1)[1])

    # Apply preprocessing on the test data
    data_sub['tweets'] = data_sub['tweets'].apply(
        lambda x: chosen_preprocessing(x, text_preprocessor))

    #### TRAIN DATA ####

    # Create dataFrame from positive tweets and give them value 1 as a sentiment
    data_pos = pd.DataFrame({
        "tweets": lines_positive,
        "sentiment": np.ones(len(lines_positive))
    })

    # Create dataFrame from negative tweets and give them value 0 as a sentiment
    data_neg = pd.DataFrame({
        "tweets": lines_negative,
        "sentiment": np.zeros(len(lines_negative))
    })

    # Concat both of them
    data = pd.concat([data_pos, data_neg],
                     axis=0).reset_index().drop(columns=['index'])

    # Shuffle everything so that we don't have all the positives in one cluster and all the negatives in another
    data = data.sample(frac=1).reset_index(drop=True).head(1000)

    print('Preprocessing on the training Data-Set ')
    # Preprocess the training data
    data['tweets'] = data['tweets'].apply(
        lambda x: chosen_preprocessing(x, text_preprocessor))

    print('Preprocessing Done')
    return data, data_sub
