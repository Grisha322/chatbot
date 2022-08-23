import csv
import pickle
import os.path
from useful_functions import stemmedWords
from enum import Enum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from config import imCleanBuild
from config import dataFolderPath

# Enum of intents recognized for chatbot so far
class Intent(Enum):
    SmallTalk = 0
    Question = 1

# This class handles Intent matching of the chatbot
class IntentMatcher:
    pathToPickle = f"{dataFolderPath}/intent_matcher.pickle"
    pathToTrainData = f"{dataFolderPath}/categories_for_training.csv"

    def __init__(self):
        # Load all the data
        self.initTrainData()
        
        # Initialize count vectorizes and tfidf transformer for loaded data
        self.count_vect = CountVectorizer(ngram_range=(1, 1), analyzer = stemmedWords)
        X_train_counts = self.count_vect.fit_transform(self.data)
        self.tf_transformer = TfidfTransformer(sublinear_tf = True).fit(X_train_counts)

        # If pickle file is absent, or intent matcher clean build option in the config file is set to true - retrain classifier
        if os.path.isfile(self.pathToPickle) and not imCleanBuild:
            with open(self.pathToPickle, "rb") as read_file:
                self.classifier = pickle.load(read_file)
        else:
            self.trainIntentMatcher()
      
    def initTrainData(self):
        self.data = []
        self.labels = []
        with open(self.pathToTrainData, encoding="utf8", mode = "r", errors = 'ignore') as read_file:
            csv_reader = csv.reader(read_file, delimiter = ",")
            for row in csv_reader:
                self.data.append(row[0])
                self.labels.append(row[1])

    # Train classifier on the loaded data for intent matching purpuses. Evaluation of the given classifier was tested using
    # cross validation, however the code for it is absent in the given package, due to being not relevant to the scope of the module
    def trainIntentMatcher(self):
        X_train_counts = self.count_vect.fit_transform(self.data)
        X_train_tf = self.tf_transformer.transform(X_train_counts)
        clf = LogisticRegression(random_state=0).fit(X_train_tf, self.labels)
        with open(self.pathToPickle, "wb") as write_file:
            pickle.dump(clf, write_file)
        self.classifier = clf

    # This method transforms and predicts the class label of the given query.
    # The label is then converted to Enum element 
    def predict(self, X):
        X_new_counts = self.count_vect.transform(X if isinstance(X, list) else [X])
        X_new_tfidf = self.tf_transformer.transform(X_new_counts)
        predicted = self.classifier.predict(X_new_tfidf)
        return Intent(int(predicted))

