"""
Preprocessing text 
"""

#import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from gensim.parsing.preprocessing import remove_stopwords 



class TextPreprocessing:

    #nltk.download('punkt')

    def __init__(self, tokenizer="nltk", language="english", stopwords=False, add_stopwords=None, 
                 stemming=False, lemmatization=False, sentence=False):
        
        self.tokenizer = tokenizer
        self.language = language #choose language for pre_trained tokenizer
        self.stopwords = stopwords #remove stopwords from text
        self.add_stopwords = add_stopwords #add additional stopwords if needed
        self.stemming = stemming #reduce words to their root or base form by removing suffixes
        self.lemmatization = lemmatization
        self.sentence = sentence #tokenize by word or sentence

        self.tokens = None
        self.tokens_ids = None
        # self.tokens_stem = None
        # self.tokens_lem = None


    def stopwords_step(self, text):
        list_stopwords = list(set(stopwords.words(self.language)))
        
        if self.add_stopwords is not None:
            list_stopwords.extend(self.add_stopwords)
        
        return remove_stopwords(text,list_stopwords)


    def tokenization_step(self, text): 
        if self.tokenizer == "nltk":
            return word_tokenize(text, self.language) #, preserve_line=False)
        
        elif self.tokenizer == "spacy":
            pass

        elif self.tokenizer == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokens = tokenizer.tokenize(text)
            self.tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            return tokenizer.tokenize(text)
        
        else:
            raise(NotImplementedError("This tokenizer isn't implemented yet"))

    
    def stemming_step(self, tokens):
        stemmer = SnowballStemmer(language=self.language)
        tokens_stem = []
        for t in tokens:
            tokens_stem.append(stemmer.stem(t))
        
        return tokens_stem
    

    def lemmatizer_step(self, tokens):
        if self.language == "english":
            lemma_funs = WordNetLemmatizer()
        else:
            lemma_funs = FrenchLefffLemmatizer()
            
        tokens_lem = []
        for t in tokens:
            tokens_lem.append(lemma_funs.lemmatize(t))
        
        return tokens_lem
    

    def preprocess_pipeline(self, text):

        if isinstance(text,str) is False:
            raise(TypeError())
        
        # Remove stopwords
        if stopwords is True:
            tokens = self.stopwords_step(text)

        # Tokenization
        tokens = self.tokenization_step(text)
        
        # Stemming
        if self.stemming is True:
            tokens = self.stemming_step(tokens)
        
        # Lemmatization
        if self.lemmatization is True:
            tokens = self.lemmatizer_step(tokens)
        
        return tokens
        
        
    def fit_transform(self, text):

        if self.language not in ["english","french"]:
            raise (NotImplementedError("This language isn't implemented"))

        if isinstance(text,str) is True:
            self.tokens = self.preprocess_pipeline(text)
        
        if isinstance(text,list) is True:
            self.tokens = [self.preprocess_pipeline(t) for t in text]
        
        return self.tokens



# class TextEmbedding:
#     def __init__(self,):
#         pass



    

        


####### Example ########           

#print(stopwords.words("english"))

from text_extraction import TextExtraction
from text_cleaning import TextCleaning 

extraction = TextExtraction(path="examples/text_collection_ex")
text = extraction.extract_pdf()

cleaning = TextCleaning(digits=True)
text_clean = cleaning.fit_transform(text)

# from nltk.tokenize import sent_tokenize
# sent_tokenizer = sent_tokenize(text_clean[1], language='english')
# print(sent_tokenizer)


preprocessing = TextPreprocessing(stopwords=True, tokenizer="bert")
text_tokens = preprocessing.fit_transform(text_clean[1])
ids_tokens = preprocessing.tokens_ids

print(text_tokens)
print(ids_tokens)