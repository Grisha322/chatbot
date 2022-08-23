import json
import pickle
import os.path
import random
import bag_of_word as bow
from enum import Enum
from config import stCleanBuild
from identity_manager import identityManager
from intent_matcher import Intent
from config import dataFolderPath

# Enum of different small talk intents
class SmalltalkIntents(Enum):
    Greetings = 0
    CourtesyGreeting = 1
    HowAreYou = 2
    Thanks = 3
    Shutup = 4
    Goodbye = 5
    CourtesyGoodbye = 6
    Clever = 7
    GreetingsAnswer = 8
    NotIdentified = 100

    def StringToEnum(name):
        if name == "Greetings":
            return SmalltalkIntents.Greetings
        elif name == "CourtesyGreeting":
            return SmalltalkIntents.CourtesyGreeting
        elif name == "HowAreYou":
            return SmalltalkIntents.HowAreYou
        elif name == "Thanks":
            return SmalltalkIntents.Thanks
        elif name == "Shutup":
            return SmalltalkIntents.Shutup
        elif name == "Goodbye":
            return SmalltalkIntents.Goodbye
        elif name == "CourtesyGoodbye":
            return SmalltalkIntents.CourtesyGoodbye
        elif name == "Clever":
            return SmalltalkIntents.Clever
        elif name == "GreetingsAnswer":
            return SmalltalkIntents.GreetingsAnswer
        else:
            raise Exception(f'{name} is not associated with SmallTalkIntents')

# This class handles small talk interactions
class SmallTalkHandler:
    pathToData = f"{dataFolderPath}/small_talk.json"
    pathToPickle = f"{dataFolderPath}/small_talk.pickle"

    def __init__(self):
        self.initData()

        # If pickle file is absent, or small talk clean build option in the config file is set to true - rebuild bow model
        if os.path.isfile(self.pathToPickle) and not stCleanBuild:
            with open(self.pathToPickle, "rb") as read_file:
                self.bow = pickle.load(read_file)
        else:
            # Obtian a list of all queries from different intents
            queries = self.combineQueries()
           
            # Create a bag of word model for the queries. Exclude removing stopwords for preprocessing and pick Term Frequency as a weightning option
            self.bow = bow.BagOfWordFactory.getBagOfWord(documents=queries, idLetterAnnotation="Q", removeStopWord=False, weightOption= bow.WeightOption.TF)
            #Save created bow into a pickle
            with open(self.pathToPickle, "wb") as write_file:
                pickle.dump(self.bow, write_file)

    # This method matches intent of a given query and calls for response generation and returns it
    def handleQuery(self, query):
        smallTalkIntent = self.matchIntent(query)
        response = ""
        if(not smallTalkIntent == SmalltalkIntents.NotIdentified):
            response = self.generateResponse(smallTalkIntent)
            
        return response

    # This method generates response for a given intent
    def generateResponse(self, smallTalkIntent):
        response = ""
        if smallTalkIntent == SmalltalkIntents.CourtesyGreeting:
            # Add Greeting at the beginning of a response
            response = f"{self.generateResponseForIntent(SmalltalkIntents.Greetings)}, {self.generateResponseForIntent(smallTalkIntent)}"
        elif smallTalkIntent == SmalltalkIntents.Greetings or smallTalkIntent == SmalltalkIntents.GreetingsAnswer:
            # Add a name to a response
            response = f"{self.generateResponseForIntent(smallTalkIntent)}! {identityManager.getNameOutro(Intent.SmallTalk)}"               
        else:
            response += self.generateResponseForIntent(smallTalkIntent)
        return response

    # Choose randomly from a collection of responses
    def generateResponseForIntent(self, smallTalkIntent):
        return random.choice(self.responses[smallTalkIntent])

    # This method matches intent of a query
    def matchIntent(self, query):
        # Preprocess and transform the query based on the bow model saved
        transformedQuery = self.bow.transformDocument(query)
        # Compute similarities
        similarities = self.bow.computeSimilarities(transformedQuery)
        bestMatch = max(similarities)
        # If best similarity is less than 50 percent, then notify the failure of intent matching
        if bestMatch < 0.5:
            return SmalltalkIntents.NotIdentified
        # Find to which intent group the best similarity belongs
        queries = self.combineQueries()
        matchedQuestion = queries[similarities.index(bestMatch)]
        for smallTalkIntents in self.queries:
            if matchedQuestion in self.queries[smallTalkIntents]:
                return smallTalkIntents
            

    # This method combines queries of all the intent groups into 1 list
    def combineQueries(self):
        return [query for smallTalkIntent in self.queries for query in self.queries[smallTalkIntent]]

    # This method fethces queries and responses from the datafile.
    def initData(self):
        self.queries = {}
        self.responses = {}
        with open(self.pathToData, "r") as read_file:
            data = json.load(read_file)

        for intent in data["intents"]:
            try:
                smallTalkIntent = SmalltalkIntents.StringToEnum(intent)
            except Exception:
                continue
            self.queries[smallTalkIntent] = []
            self.responses[smallTalkIntent] = []
            for query in data["intents"][intent]["query"]:
                self.queries[smallTalkIntent].append(query)
            for response in data["intents"][intent]["response"]:
                self.responses[smallTalkIntent].append(response)
