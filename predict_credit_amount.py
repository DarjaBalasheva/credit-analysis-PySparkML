from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.sql.types import IntegerType


def model_params(rf):
    '''
    Create a parameter grid for hyperparameter tuning of the Random Forest model.
    :param rf: RandomForestRegressor instance
    :return: List of hyperparameter combinations
    '''
    return ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 6, 7, 8, 9]) \
        .addGrid(rf.maxBins, [4, 5, 6, 7]) \
        .build()


def prepare_data(df: DataFrame, assembler) -> DataFrame:
    '''
    Prepare the data for training by transforming the necessary columns.
    :param df: Input Spark DataFrame
    :param assembler: Configured VectorAssembler
    :return: Transformed DataFrame ready for modeling
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
    features = ["age", "sex_index", "is_married", "salary", "successfully_credit_completed",
                "credit_completed_amount", "active_credits", "active_credits_amount"]
    return VectorAssembler(inputCols=features, outputCol="features")


def build_random_forest() -> RandomForestRegressor:
    '''
    Create a Random Forest Regressor for predicting credit amount.
    :return: RandomForestRegressor instance
    '''
    return RandomForestRegressor(featuresCol="features", labelCol="credit_amount")


def build_evaluator() -> RegressionEvaluator:
    '''
    Create an evaluator for the Random Forest model.
    :return: RegressionEvaluator instance
    '''
    return RegressionEvaluator(predictionCol="prediction",
                               labelCol="credit_amount",
                               metricName="rmse")


def build_tvs(rand_forest, evaluator, model_params) -> TrainValidationSplit:
    '''
    Build a TrainValidationSplit for hyperparameter tuning of the Random Forest model.
    :param rand_forest: RandomForestRegressor instance
    :param evaluator: Evaluator for model performance
    :param model_params: List of hyperparameter combinations
    :return: TrainValidationSplit instance
    '''
    return TrainValidationSplit(
        estimator=rand_forest,
        estimatorParamMaps=model_params,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2
    )


def train_model(train_df, test_df) -> (RandomForestRegressionModel, float):
    '''
    Train the Random Forest model and evaluate it on the test set.
    :param train_df: Training DataFrame
    :param test_df: Test DataFrame
    :return: Best model and RMSE
    '''

    # Prepare the data
    assembler = vector_assembler()

    # Assemble features for training and testing data
    train_pdf = prepare_data(train_df, assembler)
    test_pdf = prepare_data(test_df, assembler)

    # # Build the Random Forest model
    rf = build_random_forest()

    # Build the evaluator
    evaluator = build_evaluator()

    # Build the TrainValidationSplit for hyperparameter tuning
    tvs = build_tvs(rf, evaluator, model_params(rf))

    # Fit the model
    models = tvs.fit(train_pdf)

    # Evaluate the best model on the test set
    best = models.bestModel

    # Get best predictions and evaluate RMSE
    predictions = best.transform(test_pdf)
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE: {rmse}")
    print(f'Model maxDepth: {best._java_obj.getMaxDepth()}')
    print(f'Model maxBins: {best._java_obj.getMaxBins()}')
    return best, rmse


if __name__ == "__main__":
    '''
    Training a model that predicts a credit amount.
    '''

    # Initialize Spark session
    spark = SparkSession.builder.appName('PySparkMLJob').getOrCreate()

    # Load training and test data
    train_df = spark.read.parquet("data/predict_credit_amount/train.parquet")
    test_df = spark.read.parquet("data/predict_credit_amount/test.parquet")

    # Train the model
    train_model(train_df, test_df)
