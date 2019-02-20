from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd


def build_clean_file_vocab():
    """
    Auxilary methode to create a clean csv file of pre processed tweets for training the word2vec model


    OUTPUT:
        Creates the csv file 'full_data.csv' in the repository
    """

    data_folder = './'
    positive_path = os.path.join(data_folder, 'train_pos_full.txt')
    negative_path = os.path.join(data_folder, 'train_neg_full.txt')
    lines_positive = [line.rstrip('\n') for line in open(positive_path)]
    lines_negative = [line.rstrip('\n') for line in open(negative_path)]
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


	#### Preprocessing tool ####

    text_preprocessor = create_text_preprocessor()

    # Shuffle everything so that we don't have all the positives in one cluster and all the negatives in another
    data = data.sample(frac=1).reset_index(drop=True)
    data['tweets'] = data['tweets'].apply(lambda x: chosen_preprocessing(x,text_preprocessor))
    data.to_csv('full_data.csv')


def build_vocab():
    """
        Trains word2vec model from our corpus of tweets

    INPUT:

    OUTPUT:
        Creates the file 'word2vec.model' in the repository which will be loaded to calculate the embedding (weight) matrix of the Neural Nets model
    """

    # The file we use to build the dictionnary is a pre processed file
    # It takes a lot of time to process so we saved it in a csv file and provided a download link
    # Download link = https://uploadfiles.io/sm25c?fbclid=IwAR0bMSuP_c_JDGWgSLITDDpDt_C9EdJOFZOV6HjeaPWPHlrIFJ-yWURXOfE

    data = pd.read_csv('./full_data.csv')

    # If you want to recompue this file, just uncomment the two lines below
    #build_clean_file_vocab()
    #data = pd.read_csv('./full_data.csv')

    data['tweets'].dropna(
        inplace=True)  #Drop the NAN values caused by errors in reading the csv
    X = data['tweets']

    #Tokenize tweets into words
    X = [word_tokenize(i) for i in X]

    #Remove tokens with length < 1 for the training
    X = truncate_small_words(X, minimum_size=1)

    ## Building and training our Word2Vec Model ##
    window_size = 5  #  The context is given by the terms that occur within a window-neighbourhood of a term
    size = 50  # number of dimensions of the vector for a word
    epochs = 100  # number of iterations on the tweets
    min_count = 2  #Terms that appear less than min_count are ignored (We chose to only ignore the terms that occur once for more precision)
    cores = multiprocessing.cpu_count()  #number of cpus used
    sg = 1  #if 1 than skipgram is used else cbow

    model = Word2Vec(
        X,
        sg=1,
        window=window_size,
        size=size,
        workers=cores,
        iter=epochs,
        sample=0.00001)
    model.save("word2vec.model")
