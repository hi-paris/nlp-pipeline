
This article was published as a part of the Data Science Blogathon.

Introduction
Hello friends, In this article, we will discuss End to End NLP pipeline in an easy way. If we have to build any NLP-based software using Machine Learning or Deep Learning then we can use this pipeline. Natural Language Processing (NLP) is one of the fastest-growing fields in the world. Natural language processing (NLP) is a field of artificial intelligence in which computers analyze, understand, and derive meaningful information from human language in a smart and useful way. The set of ordered stages one should go through from a labeled dataset to creating a classifier that can be applied to new samples is called the NLP pipeline.

NLP Pipeline
NLP Pipeline
NLP Pipeline is a set of steps followed to build an end to end NLP software.

Before we started we have to remember this things pipeline is not universal, Deep Learning Pipelines are slightly different, and Pipeline is non-linear.

1. Data Acquisition
In the data acquisition step, these three possible situations happen.

1. Data Available Already


DataHour: Era of AI-Assisted Innovation
🗓️ Date: 6 Nov 2023 🕖 Time: 8:00 PM – 9:00 PM IST

A. Data available on local Machine – If data is available on the local machine then we can directly go to the next step i.e. Data Preprocessing.

B. Data available in Database – If data is available in the database then we have to communicate to the data engineering team. Then Data Engineering team gives data from the database. data engineers create a data warehouse.

C. Less Data Available – If data is available but it is not enough. Then we can do data Augmentation. Data augmentation is to making fake data using existing data. here we use Synonyms, Bigram flip, Back translate, or adding additional noise.

2. Data is not available in our company but is available outside. Then we can use this approach.

        A. Public Dataset – If a public dataset is available for our problem statement.
B. Web Scrapping –  Scrapping competitor data using beautiful soup or other libraries
C. API – Using different APIs. eg. Rapid API

3. Data Not Available – Here we have to survey to collect data. and then manually give a label to the data.

2. Text Preprocessing
So Our data collection step is done but we can not use this data for model building. we have to do text preprocessing.

This text preprocessing I have already explained in my previous blog. Click here.
Steps –
1. Text Cleaning – In-text cleaning we do HTML tag removing, emoji handling, Spelling checker, etc.
2. Basic Preprocessing — In basic preprocessing we do tokenization(word or sent tokenization, stop word removal, removing digit, lower casing.
3. Advance Preprocessing — In this step we do POS tagging, Parsing, and Coreference resolution.