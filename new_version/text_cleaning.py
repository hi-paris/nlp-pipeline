"""
Text cleaning script for NLP Toolbox 

Authors:
    Shreshta Shaurya, Machine Learning Engineer Intern at Hi! PARIS
    Laurène DAVID, Machine Learning Enginner at Hi! PARIS

To add: ability to give custom pattern 

"""

import re
import emoji
import string 

class TextCleaning:

    '''Clean text data

    This class does following tasks:
        - Allows you to choose whether to replace html. By default it is False.
        - Allows you to choose whether to replace digital. By default it is False.
        - Allows you to choose whether to change emoji 2 words. By default it is False.

    
    Parameters 
    ----------
    lower: boolean, default True 
        If True, the text is converted to lower case

    punctuation: boolean, default True 
        If True, 

    emoji2word: boolean, default False
        If True, lets you convert emojis into words 

    digits: boolean, default False
        If True, lets you remove digits from the text 

    html: boolean, default False 
        If True, lets you remove html from the text by default it is True 
    

    '''

    def __init__(self, lower=True, punctuation=True, emoji2word=False, digits=False, html=False):
        self.lower = lower
        self.punctuation = punctuation 
        self.emoji2word = emoji2word
        self.digits = digits
        self.html = html 


    def clean_text(self, text):
        '''
        Clean text data
        '''

        # if isinstance(text,str) == False :
        #     raise(TypeError("only string objects can be cleaned"))
        
        if self.lower:
            text = text.lower()

        if self.emoji2word:
            text = emoji.demojize(text, delimiters=("", ""))

        if self.digits: 
            text = re.sub(r'[0-9]', "", text)
    
        if self.punctuation:
            pattern = re.compile('[%s]' % re.escape(string.punctuation))
            text = pattern.sub('', text)
        
        if self.html:
            text = re.sub('http://\S+|https://\S+|www.\S+', '', text)
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text = re.sub("<[a][^>]*>(.+?)</[a]>", '', text)
            text = re.sub('<p>|</p>|<i>|</i>|<br />', ' ', text)
            text = re.sub('&gt;', " ", text)
            text = re.sub('&#x27;', " ", text)
            text = re.sub('&quot;', '', text)
            text = re.sub('&#x2F;', ' ', text)
            text = re.sub('&#62;', '', text)
            text = re.sub(r'&amp;', '', text)
        
        return text.strip()


    def fit_transform(self, text):
        '''
        Fit text cleaning class to a string or list of strings 

        Parameters
        ----------
        text: str or list 
            Input text to be cleaned. 
        
        Returns
        ---------
        clean_text: str or list 
            If text is str, clean_text will be the cleaned string input 
            If text is list, clean_text will be a list with 

        '''
        if isinstance(text,str):
            self.clean_text = self.clean_text(text)
            return self.clean_text
        
        elif isinstance(text,list):
            self.clean_text = [self.clean_text(t) for t in text]
            return self.clean_text
        
        else:
            raise(TypeError("text should be a string or list type"))
    


# text = "Hello World, 789!"
# cleaning = TextCleaning(digits=False)
# clean_text = cleaning.fit_transform(text)

# print(clean_text)