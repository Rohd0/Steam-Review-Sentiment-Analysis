# http://localhost:4040/jobs/
#
#
#
# here i am importing all the modules i will be using, pyspark and textblob are the main ones here, i need the functions like avg and count so i can do math on the columns later
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, avg, count
from pyspark.sql.types import FloatType, StringType
from textblob import TextBlob

# i had to implement this to make the code function when running, otherwise it would crash on my windows machine; spark needs to know where python is
python_path= r"C:\Users\anir\AppData\Local\Programs\Python\Python311\python.exe"
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

# i added this so spark knows where my hadoop folder is, otherwise it wont save the csv on windows. I had to download hadoop binaries manually and point to them
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] = os.environ['PATH'] + ";" + r"C:\hadoop\bin"

# this just performs a time calculation when called, i record the start time here so i can subtract it from the end time later
timer =time.time()

# here i am configuring the master node and 3 worker nodes simulation
# local-cluster[3, 1, 4096] tells spark to spawn 3 separate slave nodes, using 1 core each, with 4096MB (4GB) of ram each
# i am keeping the driver (master) at 8gb
# this setup creates a total of 4 separate java processes to act like a real distributed server cluster
pysparkobject = SparkSession.builder \
    .appName("SteamBurnoutAnalysis") \
    .master("local-cluster[3, 1, 4096]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.python.worker.timeout", "3600") \
    .getOrCreate()

# i implemented this so irrelevant log messages would go away
pysparkobject.sparkContext.setLogLevel("ERROR")

# here I am ingesting the data and loading it into the system
# inferschema makes spark guess if a column is a number or text and multiline lets it read reviews that have enter keys in them
print("Status: Loading the file from disk...")
csvfilepath ="steam_reviews.csv"
# i set inferschema to false because spark was freezing while trying to read the whole 8gb file to guess the data types
steamdata = pysparkobject.read.csv(csvfilepath, header=True, inferSchema=False, multiLine=True, escape='"')

# any rows with bugs are deleted using this, this is the cleaning part. empty / broken rows are gotten rid of so program loads
cleandata = steamdata.filter(col("review").isNotNull() & col("`author.playtime_forever`").isNotNull())

# this is a debugging utility, by setting it to 1000, i can make a nearly hour long process take seconds
# cleandata = cleandata.limit(1000)

# here, I am mapping the data and using textblob to retrieve the sentiment. a value of -1 implies negative review, and 1 implies positive review;
# i use a try and except block because if a review has weird symbols textblob might get confused and i want it to just return 0
def retrievesentiment(text):
    if text:
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    return 0.0

# here i am splitting the players by their playtimes to categorize them
# i divide the minutes by 60 to get hours for my categories
def categorybracket(minutes):
    if minutes != None:
        try:
            hours =float(minutes)/60.0
            if hours < 10: return "Category 1: 0-10 hrs"
            elif hours < 100: return "Category 2: 10-100 hrs"
            elif hours < 500: return "Category 3: 100-500 hrs"
            else: return "Category 4: 500+ hrs (Burnout)"
        except:
            return "Unknown"
    return "Unknown"

# these functions were needed for spark to communicate with the 3 slave nodes, udf helps to apply the textblob stuff
sentimentcoms=udf(retrievesentiment, FloatType())
bracketcoms=udf(categorybracket, StringType())

# asking3 slave nodes to begin work on processing the 21 million reviews parallel; i added repartition(12) so the data is split into 12 pieces, allowing the 3 slaves to take 4 pieces each
print("Mapping started.")
pysparkobject.sparkContext.setJobDescription("Step 1: Mapping Sentiment on Slaves")
mappingdata = cleandata.repartition(12).select(col("review"), col("`author.playtime_forever`").alias("playtime")).withColumn("sentiment", sentimentcoms(col("review"))).withColumn("bracket", bracketcoms(col("playtime")))

# this is where the reducing is happen, players are group by bracket and the calculations on their sentiment are done; allowing 21 million rows of data to become 4 categories to read
print("Reducing data across nodes.")
pysparkobject.sparkContext.setJobDescription("Step 2: Grouping and Reducing Data")

# the table is created here
finalstats= mappingdata.groupBy("bracket").agg(avg("sentiment").alias("avg_sentiment"),count("*").alias("total_reviews")).orderBy("bracket")

# i am outputting and printing the results here
print("Final results below:")
finalstats.show()

# the timer started near the end is stopped, total time is printed
endtimer =time.time()
executetimer =(endtimer - timer) / 60
print("Run time was" + str(executetimer) + " minutes!")

# i save the file here, i use overwrite mode so that if i run the code again it just replaces the old folder instead of erroring out
print("Status: Saving results to CSV using Hadoop.")
finalstats.write.mode("overwrite").csv("burnout_analysis_output")

# i added this so the window doesnt close instantly before i can read the time results
print("All tasks finished successfully.")
input("Exit with any key...")
pysparkobject.stop()
