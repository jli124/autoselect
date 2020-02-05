import re
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,VectorAssembler,FeatureHasher,Normalizer,StandardScaler,OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

## start a spark session
spark = SparkSession \
    .builder \
    .appName("combindfiles") \
    .getOrCreate()## read in all the data file from s3

flight_all = spark.read.format("csv").option("header", "true").load("s3a://insightflightpred/flightcsv/*.csv")
schema = ["YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","FL_DATE","OP_UNIQUE_CARRIER","OP_CARRIER_FL_NUM","ORIGIN","DEST","WEATHER_DELAY","ARR_DEL15","CANCELLED"]
#flight_2019.select("OP_UNIQUE_CARRIER").take(5)
flight_allsel = flight_all.select(schema)
#for weather delay
flight_weasel=flight_allsel.filter(flight_allsel.WEATHER_DELAY.isNotNull())

for feature in feature_num:
    flight_weasel =flight_weasel.withColumn(feature, flight_weasel[feature].cast(IntegerType()))

##FeatureHasher to feature vector
hasher = FeatureHasher(inputCols=schema,outputCol="features")
flight_wea_featurized = hasher.transform(flight_weasel)

##Splitting the dataset
flight_wea_train, flight_wea_test = flight_wea_featurized.randomSplit([0.8, 0.2], seed=12345)
df_size = float(flight_wea_train.select("ARR_DEL15").count())
num_positives = flight_wea_train.select("ARR_DEL15").where('ARR_DEL15==1').count()
num_negatives = flight_wea_train.select("ARR_DEL15").where('ARR_DEL15==0').count()
balance_ratio = 1- num_positives/df_size

balance_ratio 