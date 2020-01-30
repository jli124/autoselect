from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
from pyspark.sql import DataFrameWriter 

if (__name__ == '__main__'):
	spark = SparkSession.builder.appName("jdbc_test").getOrCreate()
	flight_2019 = spark.read.csv("2019.csv", header = True)
	schema = ["YEAR", "MONTH", "DAY_OF_MONTH"]
	flight_2019_sel = flight_2019.select(schema)
	flight_2019_result = flight_2019_sel.head(6)
	flight_2019_resultdf = spark.sparkContext.parallelize(flight_2019_result).toDF(["YEAR", "MONTH", "DAY_OF_MONTH"])
	url = "jdbc:postgresql://host:portnum/databasename"
	properties = {"user": "username", "password": "yourpassword","driver": "org.postgresql.Driver"}
	flight_2019_resultdf.write.jdbc(url=url,table='test_flight',mode='overwrite',properties=properties
	spark.stop()