s3_path = "s3a://insightflightpred/flightcsv/"
s3_path_2019 = "2019.csv"
feature_all = ["YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","FL_DATE","OP_UNIQUE_CARRIER","OP_CARRIER_FL_NUM","ORIGIN","DEST","WEATHER_DELAY","ARR_DEL15","CANCELLED"]
feature = ["YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","OP_UNIQUE_CARRIER","FLIGHTS", \
"ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID","WEATHER_DELAY","ARR_DEL15"]
feature_num = ["YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","WEATHER_DELAY","ARR_DEL15","CANCELLED"]
feature_cat = ["FL_DATE","OP_UNIQUE_CARRIER","OP_CARRIER_FL_NUM","ORIGIN","DEST"]
feature_response = ["WEATHER_DELAY","ARR_DEL15","CANCELLED"]
columns = ["YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK","OP_UNIQUE_CARRIER","FLIGHTS","ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID", "WEATHER_DELAY", "ARR_DEL15", "features", "rawPrediction", "probability", "prediction"]
