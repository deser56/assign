#Sharukya Smitesh Marneni 
#CS-643 Programming Assignment - 2 
#Prediction
import argparse
import os
import quinn
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    def clean_up_quotes(dataframe):
        return dataframe.replace('"', '')
    inputCols=["fixed acidity", \
                   "volatile acidity", \
                   "citric acid", \
                   "residual sugar", \
                   "chlorides", \
                   "free sulfur dioxide", \
                   "total sulfur dioxide", \
                   "density", \
                   "pH", \
                   "sulphates", \
                   "alcohol"]
    assembleroutputCol="num_features"
    scaleroutputCol="features" 
    spark = SparkSession \
        .builder \
        .appName("cloudwinequality_p") \
        .getOrCreate()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, \
        help="Path to file for testing")
    parser.add_argument('-m', default= os.getcwd() + "/model", \
        help="Path to file for training")
    args = parser.parse_args()    
    ifile = args.i
    modelDir = args.m
    model = PipelineModel.load( modelDir )
    testDF = spark.read.load(ifile, format="csv", sep=";", inferSchema="true", header="true")
    testDF= quinn.with_columns_renamed(clean_up_quotes)(testDF)
    testDF = testDF.withColumnRenamed('quality', 'label')
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1score = evaluator.evaluate(model.transform(testDF))
    print("F1 Score for the Model: ", f1score )
    spark.stop()