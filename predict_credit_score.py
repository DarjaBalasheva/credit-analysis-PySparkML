from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.sql.types import IntegerType


def model_params(rf):
    '''
    Create a parameter grid for hyperparameter tuning of the Random Forest model.
    :param rf: RandomForestRegressor instance
    :return: List of hyperparameter combinations
    '''
    return ParamGridBuilder() \
        .addGrid(rf.maxDepth, [2, 3, 4, 5]) \
        .addGrid(rf.maxBins, [2, 3, 4]) \
        .build()


def prepare_data(df: DataFrame, assembler: VectorAssembler) -> DataFrame:
    '''
    Prepare the data for training by transforming the necessary columns.
    :param df: Input Spark DataFrame
    :param assembler: Configured VectorAssembler
    :return: DataFrame with transformed features
    '''
    df = df.withColumn("is_married", df.married.cast(IntegerType()))
    sex_index = StringIndexer(inputCol='sex', outputCol="sex_index")
    df = sex_index.fit(df).transform(df)
    df = assembler.transform(df)
    return df


def vector_assembler() -> VectorAssembler:
    '''
    Create a feature vector for the model.
    :return: VectorAssembler instance
    '''
    features = ["age", "sex_index", "is_married",
                "salary", "successfully_credit_completed",
                "credit_completed_amount", "active_credits",
                "active_credits_amount", "credit_amount"]
    return VectorAssembler(inputCols=features, outputCol="features")


def build_random_forest() -> RandomForestClassifier:
    '''
    Create a Random Forest Classifier.
    :return: RandomForestClassifier instance
    '''
    return RandomForestClassifier(labelCol="is_credit_closed", featuresCol="features")


def build_evaluator() -> MulticlassClassificationEvaluator:
    '''
    Create an evaluator for the model.
    :return: MulticlassClassificationEvaluator instance
    '''
    return MulticlassClassificationEvaluator(
        labelCol="is_credit_closed",
        predictionCol="prediction",
        metricName="accuracy"
)


def build_tvs(rand_forest, evaluator, model_params) -> TrainValidationSplit:
    '''
    Create a TrainValidationSplit for hyperparameter tuning.
    :param rand_forest: Random Forest Classifier
    :param evaluator: Evaluator for model performance
    :param model_params: Parameter grid for tuning
    :return: TrainValidationSplit instance
    '''
    return TrainValidationSplit(
        estimator=rand_forest,
        estimatorParamMaps=model_params,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2
    )


def train_model(train_df, test_df) -> (RandomForestClassificationModel, float):
    '''
    Train a Random Forest model on the training data and evaluate it on the test data.
    :param train_df: Training DataFrame
    :param test_df: Test DataFrame
    :return: Best model and accuracy
    '''

    # Create a feature vector for the model
    assembler = vector_assembler()

    # Prepare the training and test data
    train_pdf = prepare_data(train_df, assembler)
    test_pdf = prepare_data(test_df, assembler)

    # Build the Random Forest model
    rf = build_random_forest()

    # Build the evaluator for model performance
    evaluator = build_evaluator()

    # Create a TrainValidationSplit for hyperparameter tuning
    tvs = build_tvs(rf, evaluator, model_params(rf))

    # Fit the model on the training data
    models = tvs.fit(train_pdf)

    # Get the best model and evaluate it on the test data
    best = models.bestModel

    # Transform the test data and evaluate the model
    predictions = best.transform(test_pdf)
    accuracy = evaluator.evaluate(predictions)

    print(f"Accuracy: {accuracy}")
    print(f'Model maxDepth: {best._java_obj.getMaxDepth()}')
    print(f'Model maxBins: {best._java_obj.getMaxBins()}')
    return best, accuracy


if __name__ == "__main__":
    """
    Training a model that predicts whether a client should be granted a loan.
    """
    # Initialize Spark session
    spark = SparkSession.builder.appName('PySparkMLJob').getOrCreate()
    # Load training and test datasets
    train_df = spark.read.parquet("data/predict_credit_score/train.parquet")
    test_df = spark.read.parquet("data/predict_credit_score/test.parquet")
    # Train the model
    train_model(train_df, test_df)
