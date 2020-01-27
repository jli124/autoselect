cd Downloads/pegasus
export AWS_ACCESS_KEY_ID=XXXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXX
export AWS_DEFAULT_REGION=us-west-2
export REM_USER=ubuntu
export PEGASUS_HOME=/Users/name/Downloads/pegasus
export PATH=$PEGASUS_HOME:$PATH
source ~/.bash_profile
peg config

#peg up mymaster.yml
#peg up myworkers.yml

eval `ssh-agent -s`
peg fetch spark_cluster
peg install spark_cluster ssh
peg install spark_cluster aws
peg install spark_cluster hadoop
peg install spark_cluster environment ##
peg service spark_cluster hadoop start
peg install spark_cluster spark
peg service spark_cluster spark start

##run locale config if error seen
locale.setlocale(locale.LC_ALL,'en_US.UTF-8')