import re
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from settings import s3_path, feature_all, feature_num, feature_cat, feature_response

spark = SparkSession \
    .builder \
    .appName("readdata") \
    .getOrCreate()

def column_typechange(df,feature,column_type):
'''
change the column of the dataframe into desired type
'''
	df.withColumn(feature, df[feature].cast(column_type()))

def read_and_clean(s3_path,feature_all,feature_response):
'''
read the all csv file from S3, select features of interest and delete rows without a response feature
'''

	df = spark.read.format("csv").option("header", "true").load(s3_path + "*.csv"). \
	select(feature_all)
	for feature in feature_response:
		df.na.drop(subset=[feature])
	for feature in feature_num:
		column_typechange(df,feature,IntegerType())
return df