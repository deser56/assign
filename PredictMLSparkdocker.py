#Sharukya Smitesh Marneni 
#CS-643 Programming Assignment - 2 
#Prediction
import argparse
import os
import quinn
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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

    ifile = "ValidationDataset.csv"
    trainingDF = spark.read.load("TrainingDataset.csv",format="csv",header='true', inferSchema='true', sep=';')
    testDF = spark.read.load(ifile, format="csv", sep=";", inferSchema="true", header="true")
    
    trainingDF = quinn.with_columns_renamed(clean_up_quotes)(trainingDF)
    trainingDF = trainingDF.withColumnRenamed('quality', 'label')

    testDF= quinn.with_columns_renamed(clean_up_quotes)(testDF)
    testDF = testDF.withColumnRenamed('quality', 'label')

    rf = RandomForestClassifier()
    assembler = VectorAssembler( \
        inputCols=inputCols, \
        outputCol=assembleroutputCol)
    scaler = StandardScaler(inputCol=assembleroutputCol, outputCol=scaleroutputCol, withStd=True)
    
    pipeline = Pipeline(stages=[assembler, scaler, rf])

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 100, 500]) \
        .build()
        
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName='f1'),
                          numFolds=3)
    model = crossval.fit(trainingDF)

    evaluator = MulticlassClassificationEvaluator(metricName="f1")

    f1score = evaluator.evaluate(model.transform(testDF))

    print("F1 Score for the Model: ", f1score )
    
    spark.stop()