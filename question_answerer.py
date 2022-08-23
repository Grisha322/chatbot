import random
import csv
import pickle
import os.path
import bag_of_word as bow
from config import qaCleanBuild
from enum import Enum
from datetime import datetime
from config import chatBotName
from identity_manager import identityManager
from intent_matcher import Intent
from config import dataFolderPath

# Enum of all different question query types
class QueryType(Enum):
    GeneralQuery = 0
    TimeQuery = 1
    UserNameQuery = 2
    BotNameQuery = 3
    NotIdentified = 100

    def StringToEnum(name):
        if name == "time":
            return QueryType.TimeQuery
        elif name == "userName":
            return QueryType.UserNameQuery
        elif name == "botsName":
            return QueryType.BotNameQuery
        else:
            return QueryType.GeneralQuery

# This class handles question answering feature of the bot
class QuestionAnswerer:
    pathToTrainData = f"{dataFolderPath}/qa_dataset.csv"
    pathToPickle = f"{dataFolderPath}/question_answerer.pickle"

    def __init__(self):
        self.initTrainData()
        # If pickle file is absent, or question answerer clean build option in the config file is set to true - rebuild bow model
        if os.path.isfile(self.pathToPickle) and not qaCleanBuild:
            with open(self.pathToPickle, "rb") as read_file:
                self.bow = pickle.load(read_file)
        else:
            # Initialize bag of word model. Include stopword removal in preprocessing, though accept my you me your words, 
            # in order to distuinguish between user asking their name and user asking bot's name queries
            self.bow = bow.BagOfWordFactory.getBagOfWord(self.combineQuestions(), "Q", removeStopWord=True, acceptWords=["my", "you", "me", "your"])
            with open(self.pathToPickle, "wb") as write_file:
                pickle.dump(self.bow, write_file)

    # This method finds the most appropriate and relevant answer to the given query
    def answerQuetion(self, query):
        # Preprocess and transform the given query
        transformedQuery = self.bow.transformDocument(query)
        # Compute similarities between query and documents in the bow model
        similarities = self.bow.computeSimilarities(transformedQuery)
        bestMatch = max(similarities)
        # Identify which query type does the most similar document in the bow belongs to
        queryType = self.matchQueryType(similarities, bestMatch)
        answers = []
        if queryType == QueryType.NotIdentified:
            pass
        elif queryType == QueryType.GeneralQuery:
            # For general query, randomly choose a response option from a list of responses of the most similar questions.
            # Given qa dataset contains duplicates of the same questions, but with different responses. Those duplicates will have 
            # same similarity, yet different responses. Choose randomly from those options
            answers.append(random.choice([self.documents[queryType][i] for i in range(0, len(self.documents[queryType])) if similarities[i] == bestMatch]))
            # Get name outro from identity manager
            answers.append(identityManager.getNameOutro(Intent.Question))
        elif queryType != QueryType.UserNameQuery:
            answers.append(random.choice(self.documents[queryType])) 
        else:
            # In case if user requested their name, ask identity manager to handle this
            answers.append(identityManager.handleUserRequestedName())
        # Replace identifiers of the type <*> with a predefined value based on the identifier for each of the answers
        answers = [self.fillGaps(answer) for answer in answers]
        return answers

    # This method combines queries of all the query types into 1 list
    def combineQuestions(self):
        return [question for queryType in self.questions for question in self.questions[queryType]]

    # This function repllaces identifiers of the type <*> with a predefined value based on the identifier
    def fillGaps(self, document):
        return document.replace("<time>", datetime.now().strftime("%H:%M:%S")).replace("<bot>", chatBotName)

    # Identify query type based on list of similarities and best similarity identified 
    def matchQueryType(self, similarities, bestMatch):
        # Inform about a failure, if best similarity is less than 50 percent
        if bestMatch < 0.5:
            return QueryType.NotIdentified
        questions = self.combineQuestions()
        # Find query type group associated with the best similarity
        matchedQuestion = questions[similarities.index(bestMatch)]
        for queryType in self.questions:
            if matchedQuestion in self.questions[queryType]:
                return queryType

    # This method initializes train datta, by fetching it from specific datafile
    def initTrainData(self):
        documents = {}
        questions = {}
        for queryType in QueryType:
            if queryType == QueryType.NotIdentified:
                continue
            documents[queryType] = []
            questions[queryType] = []

        # Assign questions and responses based on the Document name column
        with open(self.pathToTrainData, encoding="utf8", mode = "r", errors = 'ignore') as read_file:
            csv_reader = csv.reader(read_file, delimiter = ",")
            first = True
            for row in csv_reader:
                if first:
                    first = not first
                    continue
                queryType = QueryType.StringToEnum(row[3])
                questions[queryType].append(row[1])
                if row[2]:
                    documents[queryType].append(row[2])
        self.documents = documents
        self.questions = questions
