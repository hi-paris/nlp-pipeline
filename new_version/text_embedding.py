"""
Text embeddings 

Author: Laurène DAVID, Machine Learning engineer at Hi! PARIS
"""

# sklearn, gensim, spacy, nltk
# CountVectorizer
# TF-IDF 
# BagofWords (same CountVectorizer ?)
# sentence transformers (BERT)


#pip install gensim nltk spacy sentence-transformers
#import nltk
#nltk.download('punkt')
#python -m spacy download en_core_web_sm


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import spacy
nltk.download('punkt')
#from sentence_transformers import SentenceTransformer

class TextEmbedding:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        #sentence transformer model(BERT-based)
        #self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    def count_vectorizer(self, texts):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        return X.toarray(), vectorizer.get_feature_names_out()

    def tfidf_vectorizer(self, texts):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        return X.toarray(), vectorizer.get_feature_names_out()

    def doc2vec_embeddings(self, texts):
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(texts)]
        model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, workers=4, epochs=100)
        return [model.dv[str(i)] for i in range(len(texts))]

    def spacy_word_vectors(self, texts):
        return [self.nlp(document).vector for document in texts]

    def bert_sentence_embeddings(self, texts):
        """
        Generate BERT sentence embeddings for the texts
        """
        return self.sentence_model.encode(texts)

"""
# BERT Sentence Embeddings
bert_vectors = text_embedding.bert_sentence_embeddings(corpus)
print("BERT Sentence Embeddings:", bert_vectors)
"""
#concret 

texts = ["Hi!Paris", "What a beautiful artsy city"]
text_embedding = TextEmbedding()

# Bag of Words
bow_vectors, bow_features = text_embedding.count_vectorizer(texts)
print("Bag of Words Vectors:", bow_vectors)

# TF-IDF
tfidf_vectors, tfidf_features = text_embedding.tfidf_vectorizer(texts)
print("TF-IDF Vectors:", tfidf_vectors)

# Gensim Doc2Vec
doc2vec_vectors = text_embedding.doc2vec_embeddings(texts)
print("Doc2Vec Vectors:", doc2vec_vectors)

# spaCy Word Vectors
spacy_vectors = text_embedding.spacy_word_vectors(texts)
print("spaCy Word Vectors:", spacy_vectors)

"""
# BERT Sentence Embeddings
bert_vectors = text_embedding.bert_sentence_embeddings(texts)
print("BERT Sentence Embeddings:", bert_vectors)
"""