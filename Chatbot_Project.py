# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2023
# Project Part 3
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# =========================================================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk

#-----------------------------------CODE FROM PART 1--------------------------------------------------

# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from Blackboard
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info(user_input)
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = ""
    name_match = re.search(r"(^|\s)([A-Z][A-Za-z-&'\.]*(\s|$)){2,4}", user_input)
    if name_match is not None:
        name = name_match.group(0).strip()
    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)
    tfidf_train = vectorizer.fit_transform(training_documents)
    return vectorizer, tfidf_train


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300, )
    if token in word2vec:
        word_vector = word2vec[token]
    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300, )
    tokens = get_tokens(user_input)
    word_embedding = []
    for token in tokens:
        word_vector = w2v(word2vec, token)
        word_embedding.append(word_vector)
    embedding = np.mean(word_embedding, axis=0)
    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Four instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    nb = None
    logistic = None
    svm = None
    mlp = None

    nb = GaussianNB()
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)
    return nb, logistic, svm, mlp


# Function: train_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    tfidf_array = tfidf_train.toarray()
    return model.fit(tfidf_array, training_labels)


# Function: train_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_w2v(model, word2vec, training_documents, training_labels):
    doc_vectors = []
    for doc in training_documents:
        doc_vector = string2vec(word2vec, doc)
        doc_vectors.append(doc_vector)
    doc_array = np.array(doc_vectors)
    return model.fit(doc_array, training_labels)


# Function: test_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    tfidf_matrix = vectorizer.transform(test_documents)
    predictions = model.predict(tfidf_matrix.toarray())
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    return precision, recall, f1, accuracy


# Function: test_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_w2v(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    doc_vectors = []
    for doc in test_documents:
        doc_vector = string2vec(word2vec, doc)
        doc_vectors.append(doc_vector)
    predictions = model.predict(np.array(doc_vectors))
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    return precision, recall, f1, accuracy

#-----------------------------------CODE FROM PART 2--------------------------------------------------

# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0
    token = [token for token in get_tokens(user_input) if token not in string.punctuation]
    num_words = len(token)
    return num_words

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps, word_counts = 0.0, []
    gtokens = nltk.tokenize.sent_tokenize(user_input)
    for gtoken in gtokens:
        word_counts.append(count_words(gtoken))
    wps = np.average(np.array(word_counts))
    return wps


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []
    word = get_tokens(user_input)
    tagged_input = nltk.pos_tag(word)
    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    pronouns = ['PRP', 'PRP$', 'WP', 'WP$']
    personal_pronouns = ['PRP']
    articles = ['DT']
    past_tense_verbs = ['VBD', 'VBN']
    future_tense_verbs = ['MD']
    prepositions = ['IN']

    for tinput in tagged_input:
        if tinput[1] in pronouns:
            num_pronouns += 1
        if tinput[1] in personal_pronouns:
            num_prp += 1
        if tinput[1] in articles:
            num_articles += 1
        if tinput[1] in past_tense_verbs:
            num_past += 1
        if tinput[1] in future_tense_verbs:
            num_future += 1
        if tinput[1] in prepositions:
            num_prep += 1
    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0
    for token in get_tokens(user_input):
        if token in ['no', 'not', 'never', 'n\'t']:
            num_negations += 1
    return num_negations


# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    # DO NOT change these values for autograder to work correctly
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20
    features = {
        "num_pronouns": num_pronouns,
        "num_prp": num_prp,
        "num_articles": num_articles,
        "num_past": num_past,
        "num_future": num_future,
        "num_prep": num_prep,
        "num_negations": num_negations
    }
    sorted_features = {key: value for key, value in sorted(features.items(), key=lambda item: item[1], reverse=True)}
    if num_words > num_words_threshold:
        informative_correlates.append(psychological_correlates["num_words"])
    if wps > wps_threshold:
        informative_correlates.append(psychological_correlates["wps"])

    count = len(informative_correlates)
    for i, j in sorted_features.items():
        if j > 0 and count < 3:
            informative_correlates.append(psychological_correlates[i])
            count += 1
    return informative_correlates


#-----------------------------------NEW IN PART 3--------------------------------------------------

# Function: welcome_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    user_input = print("Welcome to the CS 421 chatbot!  ")

    return "get_user_info"


# Function: get_info_state()
# This function does not take any input
# Returns: A string indicating the next state and a string indicating the
#          user's name
#
# This function implements a state that requests the user's name and then processes
# the user's response to extract that information.
def get_info_state():
    # Request the user's name, and accept a user response of
    # arbitrary length.  Feel free to customize this!
    user_input = input("What is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    return "sentiment_analysis", name


# Function: sentiment_analysis_state(name, model, vectorizer, word2vec)
# name: A string indicating the user's name
# model: The trained classification model used for predicting sentiment
# vectorizer: OPTIONAL; The trained vectorizer, if using TFIDF (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using Word2Vec (leave empty otherwise)
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what they want to talk about,
# and then processes their response to predict their current sentiment.
def sentiment_analysis_state(name, model, vectorizer=None, word2vec=None):
    # Check the user's sentiment
    user_input = input("Thanks {0}!  What do you want to talk about today?\n".format(name))

    # Predict the user's sentiment
    #test = vectorizer.transform([user_input])  # Use if you selected a TFIDF model
    test = string2vec(load_w2v(EMBEDDING_FILE), user_input)  # Use if you selected a w2v model

    label = None
    label = nb_w2v.predict(test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    return "stylistic_analysis"


# Function: stylistic_analysis_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative psycholinguistic correlates.
def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # Uncomment the code below to view your output from each individual function
    # print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
    #      "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
    #    num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print(
        "Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))

    return "check_next_action"


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the sentiment analysis
# ("sentiment_analysis"), or redo the stylistic analysis
# ("stylistic_analysis").
def check_next_state():
    next_state = ""
    valid_inputs = {'terminate': 'quit', 'redo sentiment': 'sentiment_analysis', 'redo stylistic': 'stylistic_analysis'}
    while True:
        user_input = input(
            "Which state would you like to enter next? (Type 'terminate', 'redo sentiment', or 'redo stylistic'): ").strip().lower()
        if user_input in valid_inputs:
            return valid_inputs[user_input]
        print("Invalid input. Defaulting to 'quit'.")
        return valid_inputs['terminate']

# Function: run_chatbot(model, vectorizer=None):
# model: A trained classification model
# vectorizer: OPTIONAL; The trained vectorizer, if using Naive Bayes (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using other classification options (leave empty otherwise)
# Returns: This function does not return any values
#
# This function implements the main chatbot system---it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, vectorizer=None, word2vec=None):
    dialogue_state = 'welcome_state'
    while dialogue_state != 'quit':
        if dialogue_state == 'welcome_state':
            dialogue_state, _ = 'name_state', welcome_state()
        elif dialogue_state == 'name_state':
            dialogue_state, _ = 'sentiment_analysis', get_info_state()
        elif dialogue_state == 'sentiment_analysis':
            dialogue_state, _ = 'stylistic_analysis', sentiment_analysis_state(name,model, word2vec)
        elif dialogue_state == 'stylistic_analysis':
            dialogue_state, _ = 'next_state', stylistic_analysis_state()
        elif dialogue_state == 'next_state':
            dialogue_state = check_next_state()
# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.
if __name__ == "__main__":
    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)  # Use if you selected a Word2Vec model

    # Compute TFIDF representations so that you can make use of them later
    # vectorizer, tfidf_train = vectorize_train(documents)  # Use if you selected a TFIDF model

    # Instantiate and train the machine learning models
    # To save time, only uncomment the lines corresponding to the sentiment
    # analysis model you chose for your chatbot!

    # nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models() # Uncomment to instantiate a TFIDF model
    nb_w2v, logistic_w2v, svm_w2v, mlp_w2v = instantiate_models()  # Uncomment to instantiate a w2v model
    # nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    nb_w2v = train_model_w2v(nb_w2v, word2vec, documents, labels)
    # logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    # logistic_w2v = train_model_w2v(logistic_w2v, word2vec, documents, labels)
    # svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)
    svm_w2v = train_model_w2v(svm_w2v, word2vec, documents, labels)
    # mlp_tfidf = train_model_tfidf(mlp_tfidf, tfidf_train, labels)
    mlp_w2v = train_model_w2v(mlp_w2v, word2vec, documents, labels)

    # ***** New in Project Part 3! *****
    #next_state = welcome_state() # Uncomment to check how this works
    next_state, name = get_info_state() # Uncomment to check how this works
    # next_state = sentiment_analysis_state(name, svm_w2v, word2vec=word2v) # Uncomment to check how this works---note that you'll need to update parameters to use different sentiment analysis models!
    # next_state = stylistic_analysis_state() # Uncomment to check how this works
    # next_state = check_next_state() # Uncomment to check how this works

    # run_chatbot(mlp, word2vec=word2vec) # Example for running the chatbot with
                                        # MLP (make sure to comment/uncomment
                                        # properties of other functions as needed)
    run_chatbot(svm_w2v, name, word2vec=word2vec) # Example for running the chatbot with SVM and Word2Vec---make sure your earlier functions are copied over for this to work correctly!
