from sklearn.feature_extraction.text import CountVectorizer
from math import log10
from enum import  Enum
import numpy as np
from sys import float_info
from useful_functions import consineSimilarity
from useful_functions import preprocessToCanonForm
from vocabulary import *

# Enum of term weight options for bag of word model
class WeightOption(Enum):
    none = 0
    TF_IDF = 1
    TF = 2

# This class is used as a bow factory
class BagOfWordFactory:
    def getBagOfWord(documents, idLetterAnnotation = "D", weightOption = WeightOption.TF_IDF, removeStopWord = True, acceptWords=[]):
        if weightOption == WeightOption.TF_IDF:
            return BagOfWordWithTF_IDF(documents, idLetterAnnotation, removeStopWord, acceptWords)
        elif weightOption == WeightOption.TF:
            return BagOfWordWithTF(documents, idLetterAnnotation, removeStopWord, acceptWords)
        elif weightOption == WeightOption.none:
            return BagOfWord(documents, idLetterAnnotation, removeStopWord, acceptWords)

# This class represents implementation of a bag of word model without any weightning applied
class BagOfWord:

    def __init__(self, documents, idLetterAnnotation = "D", removeStopWord = True, acceptWords=[]):
        # id letter annotation parameter indicates which letter will be used for ids of documents in
        # the list of canon form documents. For example if D is chosen then ids will be in the form of D{numeric id of the element} - D123

        # Remove stop words and accept words are parameters for preprocessing.
        self.removeStopWord = removeStopWord
        self.acceptWords = acceptWords

        # initialize list of documents in the canon form and vocabulary
        self.canonDocuments, self.vocabulary = self.canonFormAndVocab(documents, idLetterAnnotation)

        # Compute bag of word
        self.bow = self.initBagOfWord()

    # This method computes similarity of the given vector to every document in the canon form document list
    def computeSimilarities(self, vector):
        return [consineSimilarity(vector, self.bow[docId]) for docId in self.bow]

    # This method preprocesses and transforms the document
    def transformDocument(self, document):
        canonForm = preprocessToCanonForm(document, self.removeStopWord, self.acceptWords)
        termFrequency = np.zeros(self.vocabulary.length())
        for stem in canonForm:
            try:
                index = self.vocabulary.index(canonForm=stem)
                termFrequency[index] += 1
            except ValueError:
                continue
        return termFrequency

    # This method preprocesses given documents and computes vocabulary
    def canonFormAndVocab(self, documents, idLetter):
        counter = 1
        canonDocuments = {}
        vocabulary = Vocabulary()
        for document in documents:
                docId = f"{idLetter}{counter}"
                canonDocuments[docId] = preprocessToCanonForm(document, self.removeStopWord, self.acceptWords)
                counter += 1
                for stem in canonDocuments[docId]:
                    if not vocabulary.contains(canonForm=stem):
                        term = Term(f"t{vocabulary.getNextId()}", stem)
                        vocabulary.add(term)
        return canonDocuments, vocabulary

    # This method computes bag of word model
    def initBagOfWord(self):
        bow = {}
        for docId in self.canonDocuments:
            bow[docId] = np.zeros(self.vocabulary.length())
            for stem in self.canonDocuments[docId]:
                index = self.vocabulary.index(canonForm=stem)
                bow[docId][index] += 1
        return bow

# This class represents a bow model with tfidf weightning 
class BagOfWordWithTF_IDF(BagOfWord):

    def __init__(self, documents, idLetterAnnotation = "D", removeStopWord = True, acceptWords=[]):
        super().__init__(documents, idLetterAnnotation, removeStopWord, acceptWords)
        for term in self.vocabulary.terms:
            term.idf = self.calculateIdf(term)
        for docId in self.bow:
            self.bow[docId] = self.applyTF_IDF(self.bow[docId])

    def calculateIdf(self, term):
        documentFrequency = 0
        index = self.vocabulary.index(term=term)
        for docId in self.bow:
            if self.bow[docId][index] > 0:
                documentFrequency += 1
        return log10(self.vocabulary.length() / documentFrequency + float_info.epsilon)

    def getTf(self, frequency):
        return log10(1 + frequency)

    def transformDocument(self, document):
        termFrequncy = super().transformDocument(document)
        return self.applyTF_IDF(termFrequncy)

    def applyTF_IDF(self, document):
        return [self.getTf(document[i]) * self.vocabulary.get(index=i).idf for i in range(0, self.vocabulary.length(), 1)]

# This class represents a bow model with tf weightning 
class BagOfWordWithTF(BagOfWord):
    def __init__(self, documents, idLetterAnnotation = "D", removeStopWord = True, acceptWords=[]):
        super().__init__(documents, idLetterAnnotation, removeStopWord, acceptWords)
        
        for docId in self.bow:
            self.bow[docId] = self.applyTF(self.bow[docId])    

    def getTf(self, frequency):
        return log10(1 + frequency)

    def transformDocument(self, document):
        termFrequncy = super().transformDocument(document)
        return self.applyTF(termFrequncy)

    def applyTF(self, document):
        return [self.getTf(document[i]) 
        for i in range(0, len(document), 1)]