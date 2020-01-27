wget "https://archive.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz"
tar xvf hadoop-3.2.0.tar.gz
cd hadoop-3.2.0
sudo apt install maven
mvn package -Pdist,native -DskipTests -Dtar
export HADOOP_COMMON_LIB_NATIVE_DIR="~/hadoop/lib/"
export HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=~/hadoop/lib/"