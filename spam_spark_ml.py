from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover,HashingTF, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def main():
    # Create a SparkSession with increased memory
    spark = SparkSession.builder \
        .appName("SpamClassification") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    # Load the dataset
    # Make sure the CSV file has a header and the schema is inferred
    # df = spark.read.csv("cleaned_final_dataset.csv", header=True, inferSchema=True)
    df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("quote", "\"")
    .option("escape", "\"")
    .option("multiLine", True)   # important for long emails
    .option("encoding", "UTF-8")
    .csv("cleaned_final_dataset.csv")
    )
    
    # Filter out rows with NULL values in label or text columns
    # df = df.filter((df.label.isNotNull()) & (df.text.isNotNull()))
    
    # df.select("label").distinct().show()

    # Sample the data to reduce memory usage (use 10% of the data)
    # df = df.sample(fraction=0.1, seed=42)

    # It's a good practice to see the schema
    print("Schema of the DataFrame:")
    df.printSchema()
    print("Sample data:")
    df.show(5)

    # Convert string labels to numeric labels
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
    df = label_indexer.transform(df)

    # Prepare Data and Split into training and testing sets
    (train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

    print(f"Training data count: {train_data.count()}")
    print(f"Test data count: {test_data.count()}")

    # Configure an ML pipeline, which consists of multiple stages:
    # 1. Tokenizer: Split text into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    # 2. StopWordsRemover: Remove common English stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # 3. CountVectorizer: Convert words to a feature vector using term frequency
    # vocabSize reduced to avoid memory issues
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=275103)

    # countVectorizer = CountVectorizer(inputCol="filtered_words", outputCol="rawFeatures", vocabSize=10000)

    # 4. IDF: Apply Inverse Document Frequency to the feature vector
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    # 5. NaiveBayes: The classification algorithm
    nb = NaiveBayes(labelCol="indexedLabel", featuresCol="features", modelType="multinomial")

    # Create the pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])

    # Train the model
    print("Training the model...")
    model = pipeline.fit(train_data)
    print("Model training completed.")

    # Make predictions on the test data
    print("Evaluating the model...")
    predictions = model.transform(test_data)

    # Evaluate Model Performance
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")

    accuracy = evaluator_accuracy.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()