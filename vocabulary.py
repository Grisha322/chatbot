from numpy import NAN

# This claa describes the term of the vocabulary
# Each term has it's name, canon form and idf.
class Term:
    def __init__(self):
        self.name = ""
        self.canonForm = ""
        self.idf = NAN

    def __init__(self, name, canonForm, idf = NAN):
        self.name = name
        self.canonForm = canonForm
        self.idf = idf

# This class describes vocabulary of terms. 
class Vocabulary:
    def __init__(self):
        self.terms = []

    # This method add the given term to a vocabulary
    def add(self, term):
        self.terms.append(term)

    # This method generates id for terms.
    def getNextId(self):
        return self.length() + 1

    # This method returns an index of a term, based on one of the given parameters
    def index(self, index = NAN, name = "", canonForm = "", term = None):
        if not term:
            term = self.get(index, name, canonForm)
        return self.terms.index(term)

    # This method return current size of a vocabulary        
    def length(self):
        return len(self.terms)

    # This method returns a term based on given parameters
    def get(self, index = NAN, name = "", canonForm = ""):
        if index is not NAN:
            return self.terms[index]
        elif name:
            for term in self.terms:
                if term.name == name:
                    return term
        elif canonForm:
            for term in self.terms:
                if term.canonForm == canonForm:
                    return term
        
        raise ValueError()

    # This method checks if there exist a term with some given attributes in the vocabulary
    def contains(self, index = NAN, name = "", canonForm = ""):
        try:
            term = self.get(index, name, canonForm)
            return term
        except ValueError:
            return None