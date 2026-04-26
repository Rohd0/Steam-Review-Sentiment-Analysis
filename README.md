# Steam-Review-Sentiment-Analysis
This is a project as a part of my Advanced Database Management course, where I was tasked with using BIg Data to perform some sort of work. I opted for analyzing steam reviews using PySpark.

To run the sentiment analysis, the folllowing dependencies/enviornmental factors must be met:
- An installation of Python 3.11.9 (or above)
- All external dependencies such as PySpark, Textblob installed via pip
- An active installation of Java with the JAVA_HOME enviornmental variable setup for PySpark to function
- A copy of the dataset we are working with, named specifically as "steam_reviews.csv" in the same root folder of the Python script. The dataset in my case was downloaded from this Kaggle link: https://www.kaggle.com/datasets/najzeko/steam-reviews-2021?select=steam_reviews.csv
- HADOOP_HOME variable must be configured in windows to prevent errors.

Once these variables are fulfileld, it is simply a matter of running the Python script and waiting for the big data sentiment analysis to finish. In my case, I was running a Ryzen 7840U laptop with 32GB of RAM; some settings may need to be tuned to run on your system.
