import json
import random
from enum import Enum
from useful_functions import preprocessToCanonForm
from config import chatBotName
from intent_matcher import Intent
from config import dataFolderPath

# Enum of different Identity Manager queries
class IdentityManagementQueries(Enum):
    UserRequestedName = 0
    UserNameRequest = 1
    NameExtractedAnswer = 2
    RequestCommandAfterQuestion = 3
    RequestCommandAfterSmallTalk = 4

    def StringToEnum(name):
        if name == "UserRequestedName":
            return IdentityManagementQueries.UserRequestedName
        elif name == "UserNameRequest":
            return IdentityManagementQueries.UserNameRequest
        elif name == "NameExtractedAnswer":
            return IdentityManagementQueries.NameExtractedAnswer
        elif name == "RequestCommandAfterQuestion":
            return IdentityManagementQueries.RequestCommandAfterQuestion
        elif name == "RequestCommandAfterSmallTalk":
            return IdentityManagementQueries.RequestCommandAfterSmallTalk
        else:
            raise Exception(f'{name} is not associated with IdentityManagementQueries')

    def IntentToIMQuery(intent):
        if intent == Intent.Question:
            return IdentityManagementQueries.RequestCommandAfterQuestion
        elif intent == Intent.SmallTalk:
            return IdentityManagementQueries.RequestCommandAfterSmallTalk
        else:
            raise Exception(f'{intent} is not associated with IdentityManagementQueries')

# This class controls identity management of the system
class IdentityManager:
    userName = ""
    nameQueryRaised = False
    pathToData = f"{dataFolderPath}/identity_management.json"

    def __init__(self):
        self.initData()

    def isUserSet(self):
        return self.userName != ""

    # Answer approprietly if user requested their name
    def handleUserRequestedName(self):
        response = self.generateResponse(IdentityManagementQueries.UserRequestedName)
        # If user name is yet not known, it will be requested in generate response, so set the name query raised to true
        if not self.isUserSet():
            self.nameQueryRaised = True
        
        return response

    # This method generates outro with a name for smalltalk and question intents
    def getNameOutro(self, intent):
        try: 
            response = self.generateResponse(IdentityManagementQueries.IntentToIMQuery(intent))
        except Exception:
            response = ""
        return response

    # This method returns the name of the user is set or generates request for name
    def getName(self):
        if not self.isUserSet():
            self.nameQueryRaised = True
            return self.generateResponseForQuery(IdentityManagementQueries.UserNameRequest)
        return self.userName

    # Choose a random response from the collection of responses
    def generateResponseForQuery(self, imQuery):
        # in case if the user name is not set, and there are separate no user responses for this query then chose from them
        if not self.isUserSet() and self.no_user_responses[imQuery]:
            responses = self.no_user_responses[imQuery]
        else:
            responses = self.responses[imQuery]
        return random.choice(responses)

    # General function for response generation
    def generateResponse(self, imQuery):
        response = self.generateResponseForQuery(imQuery)
        # Replace identifiers of the type <*> with a predefined value based on the identifier
        return self.fillGaps(response)

    # This function repllaces identifiers of the type <*> with a predefined value based on the identifier
    def fillGaps(self, document):
        return document.replace("<nameQuery>", self.getName()).replace("<bot>", chatBotName)

    # Handle query that assumigly contains name. Generate appropriate response for both success and not
    def handleNameIntroduction(self, query):
        self.extractName(query)
        if self.isUserSet():
            self.nameQueryRaised = False
        return self.generateResponse(IdentityManagementQueries.NameExtractedAnswer)

    # Remove all the tokens that are in the respective name contained query vocabulary. The remaining tokens are assumed to be the name.
    def extractName(self, query):
        query = query.strip()
        self.userName = ' '.join([word for word in query.split(" ") if preprocessToCanonForm(word, False)[0] not in self.nameIntroVocab])

    # This method initializes parameters that contain reponses for every query type
    def initData(self):
        self.responses = {}
        self.no_user_responses = {}
        with open(self.pathToData, "r") as read_file:
            data = json.load(read_file)

        for headers in data["queries"]:
            try:
                imQuery = IdentityManagementQueries.StringToEnum(headers)
            except Exception:
                continue
            self.responses[imQuery] = []
            self.no_user_responses[imQuery] = []
            for response in data["queries"][headers]["response"]:
                self.responses[imQuery].append(response)
            try:
                # No user response migth be not provided for specific queries
                for no_user_response in data["queries"][headers]["no_user_response"]:
                    self.no_user_responses[imQuery].append(no_user_response)
            except KeyError:
                continue
        self.nameIntroVocab = data["nameIntroductionVocab"]

# Create a shared instance
identityManager = IdentityManager()