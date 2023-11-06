"""
Text extraction methods for text, pdf and markdown files for NLP Toolbox

Authors:
    Shreshta Shaurya, Machine Learning Engineer Intern at Hi! PARIS
    Laurène DAVID, Machine Learning Enginner @ Hi! PARIS

"""

import fitz
import glob
from unidecode import unidecode


class TextExtraction:
    '''
    Text collection class with methods to get text from .txt files, .pdf files and .md (markdown) files.

    Parameters
    ----------
    path: str 
        Path to the folder that contains the pdf files

    Attributes
    ----------
    files: list
        List of file names that were used to extract text 
    
    '''
    
    def __init__(self, path):
        self.path = path
    
    def extract_txt(self):
        '''
        Extract text from .txt files

        Returns 
        ----------
        list_text: list
            Extracted text of files with .txt extension in seperate lists
        
        '''

        files = glob.glob(f"{self.path}/*.txt")

        if len(files) == 0:
            raise(ValueError("There are no .txt files in the given directory"))

        files.sort()
        self.files_ = files

        list_text = []
        for file in files:
            try:
                with open(file) as f:
                    lines = f.read()
                    list_text.append([lines])
            except:
                list_text.append('NaN')

        return list_text


    def extract_pdf(self, unidecode=False):
        '''
        Extract text from pdf files

        Parameters 
        ----------
        unidecode: bool, default False
            Decode unicode test to ASCII 

        Returns 
        ----------
        list_text: list
            Extracted text of files with .pdf extension in seperate lists
        '''

        files = glob.glob(f"{self.path}/*.pdf")

        if len(files) == 0:
            raise(ValueError("There are no .pdf files in the given directory"))
        
        files.sort()
        self.files_ = files 

        list_text = []
        for file in files:
            whole_text = ''
            try:
                with fitz.open(file) as pdf:
                    for page in pdf:
                        text = page.get_text()
                        
                        if unidecode is True:
                            text = unidecode(text)

                        whole_text = " ".join([whole_text, text])
                list_text.append([whole_text])
            except:
                list_text.append('NaN')

        return list_text
    

    def extract_md(self):
        '''
        Extract text from markdown files

        Returns 
        ----------
        list_text: list of lists
        Extracted text of files with .md extension in seperate lists
        '''

        files = glob.glob(f"{self.path}/*.md")

        if len(files) == 0:
            raise(ValueError("There are no .md files in the given directory"))
  
        files.sort()
        self.files_ = files 

        list_text = []
        for file in files:
            try:
                with open(file,'r') as md:
                    file_split = [line for line in md]
                    file_split = " ".join(file_split)
                list_text.append([file_split])
            except:
                list_text.append('NaN')
        
        return list_text
    
    
    def extract_multiple(self, extensions):
        '''
        Extract text from different types of files (.txt, .pdf and/or .md)

        Parameters
        ---------
        extensions: tuple
            Tuple with the extensions to include in the text extraction with the format "*."
        
        Example
        ---------
        path = "examples/text_collection_ex"
        extractor = TextExtraction(path)

        extensions = ("*.pdf", "*.txt")
        text = extractor.extract_multiple(extensions)
        
        ''' 

        pass 



path_test = "examples/text_collection_ex"
extraction = TextExtraction(path=path_test)

print("TEXT:",extraction.extract_txt())
print("\n")
print("PDF:",extraction.extract_pdf())
print("\n")
print("MARKDOWN:",extraction.extract_md())

#print(extraction.files_)