# Twitter Sentiment Analysis

With faster and more powerful processors, Machine learning has become very important and offers useful tools and techniques to deal with a lot of problems in almost every scientific field.
Classification is perhaps one of the most useful tools in machine learning in order to resolve a tremendous number of real world applications. From classifying skin cancers to Amazon reviews, computer scientists are aiming to improve the different algorithms put in place in order to have better prediction while reducing computational costs.
Our aim in this paper is to present the different paths one can explore in order to predict the sentiment from a corpus of text.


## Dependencies

In order to run the project you will need the following dependencies installed:

### Libraries

* [Scikit-Learn] - Download scikit-learn library with conda

    ```sh
    $ conda install scikit-learn
    ```

* [Gensim] - Install Gensim library

    ```sh
    $ conda install gensim
    ```

* [NLTK] - Download all packages of NLTK

    ```sh
    $ python
    $ >>> import nltk
    $ >>> nltk.download()
    ```

* [GloVe] - Install Glove python
    ```sh
    $ pip install glove_python
    ```
* [keras] - Install keras library

    ```sh
    $ pip install keras
    ```
* [ekphrasis] - Install ekphrasis library

    ```sh
    $ pip3 install ekphrasis
    ```



### Files
* Data
	Link from our professors: https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files.



* Preprocessed Tweet Files used for word2vec training.

    In case you want to avoid preprocessing execution time, you can download the [preprocessed train tweets](https://down.uploadfiles.io/get/sm25c)

* Pretrained Word Embeddings

    In case you want to avoid training from scratch the whole word embeddings matrix, you can download the [glove_python](https://storage.googleapis.com/kaggle-datasets/8560/11981/glove.twitter.27B.50d.txt.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1545508577&Signature=hz8oe4Wa%2Fj%2FteWP4go5ttJle8oeVQpcZfy%2BVfrMETizFMlMcS%2F7TT86TAAQSyOg%2BRvXfTMGs0yN%2Bid1k2DtLyRiYKXrGmpxKEChrmsSabyT7YOCzZiy1bdVyE87833AvaQ30DpopUWi%2B4FbizY185fwYHNC07%2BVsDHFTBsI8yqNTtA6RdAcK%2BHLPEMcBRJZeLz%2BRT8mdvw%2FZzJ2uAbsTTnUzwnHYIDZuy27inV%2BX6Rsyv%2BUMvRDBrPjYhvxEBFWrCErLe%2BC15NH3nXk9vT3R8rXck1YZBhjcExFBPHADScAWgE6%2FESIGLBZ77HQ4WOiusqLg5fOi%2FHg1tOHY7sr2jg%3D%3D)

* Our trainded Word2Vec Embeddings model
	https://drive.google.com/file/d/18W-hB4ko0f7rfQ29uMs9h1HEzleNotok/view?fbclid=IwAR1yl0vqgy9X3Z41QigwyMKGkaqrIE7C8GG9bxyhET0kR8gY5adIiFUf7cs
## Demo


Unzip and open the main folder with terminal. Depending on the algorithm you want to test. Run (The preprocessing part takes a long time since we're using 2.5M tweets):

```sh
    $ python3 run.py 'algo_name'
```

Where 'algo_name'  may be :

* ML :
  * svm_l2 -->linear SVM with parameter l2.
  * svm_l1 -->linear SVM with parameter l1.
  * logistic --> Logistic regression
  * ridge --> Ridge regression
  * multinomial --> Multinomial Naive Bayes
  * bernoulli --> Bernouli Naive Bayes
  * voting  --> Voting classfier(SVM+logistic+ridge).


* For Neural network algorithms 'algo_name' may be:
  * cnn_gru --> Convolutional Neural Network with Gated Recurrent Unit
  * lstm --> Long Short-Term Memory Network
  * cnn_lstm --> Convolutional Neural Network with a Long Short-Term Memory Network
  * bidir_gru --> Neural Network with Bidirectional Gated Recurrent Unit
  * embeddings --> Neural Network based on the word embeddings of a word2vec model


 WHen the command is executed a sumbission file will be created under the name `submission.csv`.

* To reprocuce our best score run
```sh
    $ python3 run.py cnn_gru
```

## CrowdAI Info :
* Username: Sami
* Submission ID: 24186

## Contributors

- Sami Ben Hassen
- Firas Kanoun
- Ali Fessi
