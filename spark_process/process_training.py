import re
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf,when
import pyspark.sql.functions as f

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,VectorAssembler,FeatureHasher,Normalizer,StandardScaler,OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator,BinaryClassificationMetrics
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

## start a spark session
spark = SparkSession \
    .builder \
    .appName("combindfiles") \
    .getOrCreate()

def column_typechange(df,feature,column_type):
'''
change the column of the dataframe into desired type
'''
    df.withColumn(feature, df[feature].cast(column_type()))
    return df

def read_and_clean(s3_path,feature_all,feature_response):
'''
read the all csv file from S3, select features of interest and delete rows without a response feature
'''
    df = spark.read.format("csv").option("header", "true").load(s3_path + "*.csv"). \
    select(feature_all)
    for feature in feature_response:
        df = df.filter(df.feature.isNotNull())
    for feature in feature_num:
        df = column_typechange(df,feature,IntegerType())
    return df

def ml_transformer(df,feature_all,response_feature):
    '''
    preprocess the data ready for the logistic regression model
    '''
    feature_only = feature_all.remove(response_feature)
    hasher = FeatureHasher(inputCols=feature_only,outputCol="features")
    df_featurized = hasher.transform(df)
    df_train, df_test = df_featurized.randomSplit([0.8, 0.2], seed=12345)
    df_size = float(df_train.select(response_feature).count())
    num_positives = df_train.select(response_feature).where('{}==1'.format(response_feature)).count()
    num_negatives = df_train.select(response_feature).where('{}==0'.format(response_feature)).count()
    balance_ratio = 1- num_positives/df_size
    df_train=df_train.withColumn("classWeights", when(df_train.response_feature == 1,balance_ratio).otherwise(1-balance_ratio))
    return df_train,df_test

def ml_trainer(df_train,df_test,response_feature,df_2020):
    '''
    training and cross-validation for the model
    '''
    lr = LogisticRegression(labelCol=response_feature, featuresCol="features",weightCol="classWeights", maxIter=5)
    lrModel = lr.fit(df_train)
    df_pred = lrModel.transform(df_test)
    evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol=response_feature)
    summary_stat = lrModel.summary
    accuracy = summary_stat.accuracy
    roc = evaluator.evaluate(df_pred)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.aggregationDepth,[2,5,10])\
        .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
        .addGrid(lr.fitIntercept,[False, True])\
        .addGrid(lr.maxIter,[10, 100, 1000])\
        .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
        .build()
    cv = CrossValidator(estimator = lr,estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=3)
    cvModel = cv.fit(df_train)
    predict_train=cvModel.transform(df_train)
    predict_test=cvModel.transform(df_test)
    roc_before=evaluator.evaluate(predict_train)
    roc_after=evaluator.evaluate(predict_test)
    if roc_before > roc_before:
        return lrModel.transform(df_2020)
    else:
        return cvModel.transform(df_2020)

def extract_prob(v):
    try:
        return float(v[1])  
    except ValueError:
        return None

def post_process(df_target,columns):
    '''
    clean the result table for front-end use
    '''
    df_result = lrModel.transform(df_target)
    df_result = df_result.toDF(*columns)
    df_result = df_result.drop("features")
    df_result = df_result.drop("rawPrediction")
    df_result = udf(extract_prob, DoubleType())
    df_result = df_result.withColumn("prob_flag", extract_prob_udf(col("probability")))
    df_result = df_result.drop("probability")
    df_result = df_result.withColumn("date", F.to_date(F.concat_ws("-", "YEAR", "MONTH", "DAY")))
    return df_result

def write_result_jdbc(df_result,host,portnum,username,password,databasename):
    '''
    save the result table in postgres 
    '''
    url = "jdbc:postgresql://{}:{}/{}".format(host,portnum,databasename)
    properties = {"user": username, "password": password,"driver": "org.postgresql.Driver"}
    df_result.write.jdbc(url=url,table='test_flight',mode='overwrite',properties=properties)
    return df_result

if (__name__ == '__main__'):
    df1 = read_and_clean(s3_path,feature_all,feature_response)
    df_result_1 = ml_trainer(ml_transformer,response_feature,df_2020)
    df_result_2 = post_process(df_result_1,columns)
    write_result_jdbc(df_result_2,host,portnum,username,password,databasename)
spark.stop()