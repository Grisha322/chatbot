stCleanBuild = False # If set to True, small talk handler will be forced to rebuild it's pickle file
qaCleanBuild = False # If set to True, question answerer will be forced to rebuild it's pickle file
imCleanBuild = False # If set to True, intent manager will be forced to rebuild it's pickle file

chatBotName = "Elfie"

inputSymbol = ">>> " # Terminal input symbol

stopWord = "stop" 

onStartGreeting = "Welcome on board. Waking up the last brain cell..."
onTrainingFinished = "Done!"
briefing = "You may type any query now. If you wish to stop type `stop`"

qestionAnswererFallback = "Unable to find appropriate response in the database. Please rephrase the query."
smallTalkHandlerFallback = "I'm sorry, I don't know how to answer you."

dataFolderPath = "data"