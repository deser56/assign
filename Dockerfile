FROM openjdk:8-jre-slim

ARG SPARK_VERSION=3.3.0

LABEL maintainer="sm2983"
LABEL version="v1"

ENV DAEMON_RUN=true
ENV SPARK_VERSION=${SPARK_VERSION}
ENV HADOOP_VERSION=2.4.1
ENV HADOOP_HOME = /usr/local/hadoop
ENV SCALA_VERSION=2.11
ENV SCALA_HOME=/usr/share/scala


RUN set -ex && \
        apt-get update && \
		apt-get install wget --yes && \
		wget https://dlcdn.apache.org/hadoop/common/hadoop-2.10.2/hadoop-2.10.2.tar.gz && \
		tar -xzf hadoop-2.10.2.tar.gz && \
		mv hadoop-2.10.2/ hadoop/

RUN	apt-get install python3-pip --yes &&\
	pip install --upgrade pip && \
    pip install --no-cache pyspark==${SPARK_VERSION} && \
    pip install numpy && \
    pip install pandas && \
	pip install py4j && \
	pip install quinn && \
    pip install findspark

WORKDIR /opt/app

COPY . .


ENV PROG_NAME sm2983-PredictMLSparkdocker.py 
ADD ${PROG_NAME} .
EXPOSE 4040

