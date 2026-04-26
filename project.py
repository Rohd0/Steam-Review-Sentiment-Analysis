# here i am importing all the modules i will be using, pyspark and textblob are the main ones here
# i need the functions like avg and count so i can do math on the columns later
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, avg, count
from pyspark.sql.types import FloatType, StringType
from textblob import TextBlob

# i had to implement this to make the code function when running, otherwise it would crash on my windows machine
# this tells the spark system exactly where my python exe is so it doesnt get lost looking for it
python_path= r"C:\Users\anir\AppData\Local\Programs\Python\Python311\python.exe"
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

# this just performs a time calculation when called
# i record the start time here so i can subtract it from the end time later
timer =time.time()

# here, I am creating 1 master node and 3 slave nodes with 4g memory each. this is the creation phase of the nodes
# the appname is just a label for the project and local[4] tells it to use 4 cores of my computer cpu
# i gave the driver 8g of ram and the executors 4g so they have enough power to handle the big csv file
# the timeout is set to 3600 seconds so the program doesnt give up if the 21 million rows take too long
pysparkobject = SparkSession.builder.appName("SteamBurnoutAnalysis").master("local[4]").config("spark.driver.memory", "8g").config("spark.executor.memory", "4g").config("spark.python.worker.timeout", "3600").getOrCreate()
# i implemented this so irrelevant log messages would go away
# i only want to see errors because the normal info messages fill up the whole terminal window
pysparkobject.sparkContext.setLogLevel("ERROR")

# here I am ingesting the data and loading it into the system
# inferschema makes spark guess if a column is a number or text and multiline lets it read reviews that have enter keys in them
print("Loading the file!")
csvfilepath ="steam_reviews.csv"
steamdata = pysparkobject.read.csv(csvfilepath, header=True, inferSchema=True, multiLine=True, escape='"')

# any rows with bugs are deleted using this, this is the cleaning part. empty / broken rows are gotten rid of so program loads
# if a review is totally empty or the playtime is missing it will make the sentiment math crash so i drop those rows
cleandata = steamdata.dropna(subset=["review", "`author.playtime_forever`"])

# here, I am mapping the data and using textblob to retrieve the sentiment. a value of -1 implies negative review, and 1 implies positive review. 
# i use a try and except block because if a review has weird symbols textblob might get confused and i want it to just return 0
def retrievesentiment(text):
    if text:
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    return 0.0

# here i am splitting the players by their playtimes to categorize them
# i divide the minutes by 60 to get hours because the original dataset uses minutes and that is hard to read
def categorybracket(minutes):
    if minutes != None:
        hours =float(minutes)/60.0
        if hours < 10: return "Category 1: 0-10 hrs"
        elif hours < 100: return "Category 2: 10-100 hrs"
        elif hours < 500: return "Category 3: 100-500 hrs"
        else: return "Category 4: 500+ hrs (Burnout)"
    return "Unknown"

# these functions were needed for spark to communicate with the 3 slave nodes
# i have to wrap my python functions in a udf so spark knows how to send the logic to the worker cores
sentimentcoms=udf(retrievesentiment, FloatType())
bracketcoms=udf(categorybracket, StringType())

# here i am asking the 3 slave nodes to begin work on processing the 21 million reviews parallel
# withcolumn creates the new columns for sentiment and brackets while the alias makes the playtime name shorter
print("Mapping started!")
mappingdata = cleandata.select(col("review"), col("`author.playtime_forever`").alias("playtime")).withColumn("sentiment", sentimentcoms(col("review"))).withColumn("bracket", bracketcoms(col("playtime")))

# this is where the reducing is happen, players are group by bracket and the calculations on their sentiment are done; allowing 21 million rows of data to become 4 categories to read
# the groupby puts everyone in the same bracket together and then the agg calculates the average sentiment for that group
print("Reducing data..")
finalstats= mappingdata.groupBy("bracket").agg(avg("sentiment").alias("avg_sentiment"),count("*").alias("total_reviews")).orderBy("bracket")

# i am outputting and printing the results here
# the show function prints the little table to the screen so i can see the results immediately
print("Output:")
finalstats.show()

# the timer started near the end is stopped, total time is printed
# i calculate the final time by taking the current time and subtracting the start time then dividing by 60
endtimer =time.time()
executetimer =(endtimer - timer) / 60
print("The total time of running was: " + str(executetimer) + " minutes!")

# i save the file (this failed to save last time)
# i use overwrite mode so that if i run the code again it just replaces the old folder instead of erroring out
finalstats.write.mode("overwrite").csv("burnout_analysis_output")

# pressing any key exits the software
# i added this so the window doesnt close instantly before i can read the time results
input("Exit with any key...")
pysparkobject.stop()