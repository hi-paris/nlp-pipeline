Data Acquisition
In the data acquisition step, these three possible situations happen.

1. Data Available Already

A. Data available on local Machine – If data is available on the local machine then we can directly go to the next step i.e. Data Preprocessing.

B. Data available in Database – If data is available in the database then we have to communicate to the data engineering team. Then Data Engineering team gives data from the database. data engineers create a data warehouse.

C. Less Data Available – If data is available but it is not enough. Then we can do data Augmentation. Data augmentation is to making fake data using existing data. here we use Synonyms, Bigram flip, Back translate, or adding additional noise.

2. Data is not available in our company but is available outside. Then we can use this approach.

        A. Public Dataset – If a public dataset is available for our problem statement.
B. Web Scrapping –  Scrapping competitor data using beautiful soup or other libraries
C. API – Using different APIs. eg. Rapid API

3. Data Not Available – Here we have to survey to collect data. and then manually give a label to the data.



