from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,VectorAssembler,FeatureHasher,Normalizer,StandardScaler,OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from settings import s3_path, feature_all, feature_num, feature_cat, feature_response

def ml_pipeline(df,feature_all,feature_to_predict):
'''
build a logistic regression model with parameter tuning through cross-validation
'''
	hasher = FeatureHasher(inputCols=feature_all,outputCol="features")
	df_featurized = hasher.transform(df)
	df_train, df_test = df_featurized.randomSplit([0.8, 0.2], seed=12345)

	lr = LogisticRegression(labelCol=feature_to_predict, featuresCol="features",maxIter=10)
	evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", \
		labelCol="Outcome")
	paramGrid = ParamGridBuilder() \
		.addGrid(lr.aggregationDepth,[2,5,10])\
	    .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
	    .addGrid(lr.fitIntercept,[False, True])\
	    .addGrid(lr.maxIter,[10, 100, 1000])\
	    .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
	    .build()
	cv = CrossValidator(estimator = lr,estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=3)
	cvModel = cv.fit(df_train)
	# this will likely take a fair amount of time because of the amount of models that we're creating and testing
	predict_train=cvModel.transform(df_train)
	predict_test=cvModel.transform(df_test)
print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))