import re
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf,when

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,VectorAssembler,FeatureHasher,Normalizer,StandardScaler,OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator,BinaryClassificationMetrics
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark2pmml import PMMLBuilder

## start a spark session
spark = SparkSession \
    .builder \
    .appName("combindfiles") \
    .getOrCreate()

# os.chdir(dir_name) # change directory from working dir to dir with files
# for item in os.listdir(dir_name): \# loop through items in dir
# 	file_name = os.path.abspath(item) \ # get full path of files
# 	zip_ref = zipfile.ZipFile(file_name) \# create zipfile object
# 	zip_ref.extractall(dir_name) \ # extract file to dir
# 	zip_ref.close() \# close file
# 	os.remove(file_name)

## read in all the data file from s3
flight_all = spark.read.format("csv").option("header", "true").load("s3a://insightflightpred/flightcsv/*.csv")
## read only 2019 file
flight_2019 = spark.read.csv("2019.csv", header = True)

#flight = spark.read.option("header", "true").csv("s3n://insightflightpred/flightcsv/*.csv")

##select columns of interest
schema = ["YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","OP_UNIQUE_CARRIER","FLIGHTS","ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID","WEATHER_DELAY","ARR_DEL15"]
#flight_2019.select("OP_UNIQUE_CARRIER").take(5)
flight_2019sel = flight_2019.select(schema)
flight_2019.select("ARR_DEL15").describe()

#flight_2019sel = flight_2019sel.na.drop(subset=["ARR_DEL15"])
#flight_2019sel = flight_2019sel.na.drop(subset=["WEATHER_DELAY"])
flight_2019sel=flight_2019sel.filter(flight_2019sel.ARR_DEL15.isNotNull())
flight_2019sel.describe().select("ARR_DEL15").show()
##change all numeric features into integer type
feature_num = ["YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","WEATHER_DELAY","ARR_DEL15"]
feature_cat = ["OP_UNIQUE_CARRIER","FLIGHTS","ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID"]

for feature in feature_num:
    flight_2019sel =flight_2019sel.withColumn(feature, flight_2019sel[feature].cast(IntegerType()))


##Transforming the categorical variables
# def transform_cat_feature(df, cat_feature, cat_feature_vec):
# 	indexer = StringIndexer(inputCol=cat_feature, outputCol=cat_feature_vec)
# 	model_index = indexer.fit(df)
# 	indexed = model_index.transform(df)
# 	encoder_cat = OneHotEncoder(inputCol=cat_feature, outputCol=cat_feature_vec)
# 	model_encode = encoder_cat.fit(df)
# 	encoded = model_encode.transform(df)
# 	encoded.show()

##FeatureHasher to feature vector
hasher = FeatureHasher(inputCols=schema,outputCol="features")
flight_2019_featurized = hasher.transform(flight_2019sel)

##Splitting the dataset
flight_2019_train, flight_2019_test = flight_2019_featurized.randomSplit([0.8, 0.2], seed=12345)

## Unbalaned handling
df_size = float(flight_2019_train.select("ARR_DEL15").count())
num_positives = flight_2019_train.select("ARR_DEL15").where('ARR_DEL15==1').count()
num_negatives = flight_2019_train.select("ARR_DEL15").where('ARR_DEL15==0').count()

balance_ratio = 1- num_positives/df_size
balance_ratio = 0.8101951264352605
flight_2019_train=flight_2019_train.withColumn("classWeights", when(flight_2019_train.ARR_DEL15 == 1,balance_ratio).otherwise(1-balance_ratio))
##Logistic Regression Model
lr = LogisticRegression(labelCol="ARR_DEL15", featuresCol="features",weightCol="classWeights", maxIter=5)
## Building the ML pipeline
#pipeline = Pipeline(stages=[FeatureHasher,lr])
lrModel = lr.fit(flight_2019_train)
flight_2019_pred = lrModel.transform(flight_2019_test)
##model evaluation
evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="ARR_DEL15")

correct = flight_2019_pred.filter(flight_2019_pred.ARR_DEL15 == flight_2019_pred.prediction).count()
total = flight_2019_pred.count(）
summary_stat = lrModel.summary
accuracy_2019 = summary_stat.accuracy
roc_2019 = evaluator.evaluate(flight_2019_pred)
##model evaluation
evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="ARR_DEL15")
#flight_2019_pred.select("Outcome","rawPrediction","prediction","probability").show(5)

##The ML pipeline and save
from pyspark.ml.feature import RFormula
formula = RFormula(formula = "ARR_DEL15 ~ .")
pipeline = Pipeline(stages = [formula,lr])
flight_2019_train = flight_2019_train.drop("features")
pipelineModel = pipeline.fit(flight_2019_train)
sc = SparkContext('test')
pmmlBuilder = PMMLBuilder(sc, flight_2019_train, pipelineModel) \
 .putOption(lr, "compact", True)
pmmlBuilder.buildFile("logReg.pmml")
 





##getting the coefficient matrix
coefficients = lrModel.coefficientMatrix
intercept = lrModel.intercept
weights = best_lr.weights
coefficients = beslrModelt_lr.coefficients
intercept = lrModel.intercept







##crossvalidation
paramGrid = ParamGridBuilder() \
	.addGrid(lr.aggregationDepth,[2,5,10])\
    .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
    .addGrid(lr.fitIntercept,[False, True])\
    .addGrid(lr.maxIter,[10, 100, 1000])\
    .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
    .build()
cv = CrossValidator(estimator = lr,estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=3)
cvModel = cv.fit(flight_2019_train)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing
predict_train=cvModel.transform(flight_2019_train)
predict_test=cvModel.transform(flight_2019_test)
print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))

schema2 = ["YEAR","MONTH","DAY","DAY_OF_WEEK", "AIRLINE","FLIGHTS","ORIGIN_AIRPORT","DESTINATION_AIRPORT",
"SCHEDULED_DEPARTURE","DEPARTURE_TIME","DEPARTURE_DELAY","TAXI_OUT","SCHEDULED_TIME","ELAPSED_TIME","AIR_TIME","DISTANCE"
,"TAXI_IN","SCHEDULED_ARRIVAL","ARRIVAL_TIME","CANCELLED","CANCELLATION_REASON","DELAYED"]

flight_2019_resultdf = spark.sparkContext.parallelize(flight_2019sel).toDF(schema2)
flight_2019_resultdf = sc.parallelize(flight_2019sel).toDF(schema2)
