from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import PorterStemmer
from sys import float_info

# This function returns a list of stemmed words from a given document
def stemmedWords(doc):
        analyzer = CountVectorizer().build_analyzer()
        p_stemmer = PorterStemmer()
        return (p_stemmer.stem(w) for w in analyzer(doc))

# This function computes cosine similarity of given vectors
def consineSimilarity(vector1, vector2):
        # epsilon is added to each element of the vectors in order to avoid strange Python bug, that occurs when the vector consists of only zeros
        similarity = 1 - spatial.distance.cosine([x + float_info.epsilon for x in vector1], [x + float_info.epsilon for x in vector2] )
        return similarity

# This function preprocesses to canon form by lowering the case, performing the stemming and stop words removal.
# If removeStopWords is set to False, then stop words won't be removed when pre processing. Additionally
# All the words provided in the acceptWords parameter will be accepted regardless of stop word removal
def preprocessToCanonForm(document, removeStopWord, acceptWords = []):
        p_stemmer = PorterStemmer()
        return [p_stemmer.stem(word.lower()) for word in word_tokenize(document) if not (removeStopWord and word.lower() in stopwords.words('english')) or word.lower() in acceptWords]