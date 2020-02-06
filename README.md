# Easy Prediction
## Motivation
Within United States, there are more than 3 million travelers taking flights every day, and among which more than 5% are experiencing delayed, cancellation and some other performances that could make the trip unpleasant. It would be useful to be able to predict the delayed/cancellation risk before travelling so that the frustration for the travelers could be reduced and they could also have the chance to plan their trip accordingly. 
## Dataset
The dataset is obtained through Department of Transportation, [Report Carrier On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.asp).The dataset contains information like time period, origin information, departure information, delay information etc. The dataset is about 70 GB covering the above information from 1987 to 2019. 
## Preliminary pipeline
![pipeline image](https://github.com/jli124/autoselect/blob/master/pipeline1.png)
Spark is set up through Pegasus, and the PostgreSQL database is connected through Amazon RDS. The results of the predicted probability were save in PostgreSQL and visualized through Dash. 

