import intent_matcher as im
import question_answerer as qa
import small_talk_handler as st
import config
from identity_manager import identityManager

class Main:

    # Initialize and train (if needed) small talk handler, intent matcher and question answerer
    def __init__(self):
        print(config.onStartGreeting)
        self.smallTalkHandler = st.SmallTalkHandler() 
        self.answerer = qa.QuestionAnswerer()
        self.matcher = im.IntentMatcher()
        print(config.onTrainingFinished)

    # This method prints out bots answer to a query
    def answer(self, answer):
        print(f"{config.chatBotName}: {answer}")

    # This method executes main loop of the program.
    def run(self):
        stop = False
        print(config.briefing)
        while(not stop):
            # Prompt the user for input
            query = input(config.inputSymbol)

            # Check for stopword
            if self.matchWord(query, config.stopWord):
                stop = not stop
                continue
            if not query.strip(): # Ensure string has some characters
                continue

            # Check if identity manager is awaiting for users name
            if identityManager.nameQueryRaised:
                self.answer(identityManager.handleNameIntroduction(query))
                continue

            # Match intent
            intent = self.matcher.predict(query)

            # Answer according to intent
            if intent == im.Intent.Question:
                if(answers := self.answerer.answerQuetion(query)):
                    for answer in answers:
                        self.answer(answer)
                else:
                    self.answer(config.qestionAnswererFallback)
            elif intent == im.Intent.SmallTalk:
                if(answer := self.smallTalkHandler.handleQuery(query)):
                    self.answer(answer)
                else:
                    self.answer(config.smallTalkHandlerFallback)

    # This checks if stopword was typed
    def matchWord(self, query, word):
        return ''.join(c.lower() for c in query if c.isalnum()) == word

Main().run()

"""
import json
import sys
import random

with open("Intent.json", "r") as read_file:
    data = json.load(read_file)

for intents in data["intents"]:
    for question in intents["text"]:
        print(question.lower())

print("\n")

questions = []
with open("qa_dataset.csv", encoding="utf8", mode = "r", errors = 'ignore') as read_file:
    csv_reader = csv.reader(read_file)
    i = 0
    for row in csv_reader:
        if i == 0:
            i = 1
            continue
        questions.append(row[1])
        
sample = random.choices(questions, k = 150)

for question in sample: 
    sys.stdout.buffer.write((question.lower() + '\n').encode('utf8'))
       
"""