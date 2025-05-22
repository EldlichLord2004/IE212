import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, regexp_replace, lower, explode

# Create Spark session
spark = SparkSession.builder.appName("CollaborativeFiltering") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load data
user_reviews = spark.read.parquet("Dataset/user_reviews.parquet")
metadata = spark.read.parquet("Dataset/metadata.parquet")

def prepare_data(user_reviews_df, metadata_df, sample_size=None):
    if sample_size:
        reviews_df = user_reviews_df.limit(sample_size)
    else:
        reviews_df = user_reviews_df
    complete_df = reviews_df.join(metadata_df, reviews_df.parent_asin == metadata_df.parent_asin, "inner")
    hybrid_df = complete_df.select(
        reviews_df["review_id"],
        reviews_df["title"].alias("review_title"),
        metadata_df["title"].alias("product_title"),
        reviews_df["rating"],
        metadata_df["average_rating"],
        metadata_df["rating_number"],
        reviews_df["asin"].alias("item_id"),
        metadata_df["parent_asin"],
        reviews_df["user_id"],
        reviews_df["helpful_vote"],
        metadata_df["categories"],
        reviews_df["text"].alias("review_text"),
        reviews_df["rating_label"]
    )
    hybrid_df = hybrid_df.withColumn(
        "clean_review_text", 
        regexp_replace(lower(col("review_text")), "[^a-zA-Z0-9 ]", " ")
    )
    return hybrid_df

def build_collaborative_filtering(df):
    print("Building collaborative filtering model...")
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="item_id", outputCol="item_index", handleInvalid="skip")
    indexed_df = user_indexer.fit(df).transform(df)
    indexed_df = item_indexer.fit(indexed_df).transform(indexed_df)
    (training, test) = indexed_df.randomSplit([0.8, 0.2], seed=42)
    als = ALS(
        maxIter=10,
        regParam=0.05,
        rank=50,
        userCol="user_index",
        itemCol="item_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=False
    )
    model = als.fit(training)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Collaborative Filtering RMSE = {rmse}")
    # Save models
    model.save("als_model")
    user_indexer.write().overwrite().save("user_indexer_model")
    item_indexer.write().overwrite().save("item_indexer_model")
    print("Models saved.")
    return model, user_indexer, item_indexer

if __name__ == "__main__":
    sample_size = 10000
    hybrid_df = prepare_data(user_reviews, metadata, sample_size)
    model, user_indexer, item_indexer = build_collaborative_filtering(hybrid_df)

    # Test: lấy user ngẫu nhiên và show top 5 sản phẩm được recommend
    test_user = hybrid_df.select("user_id").distinct().limit(1).collect()[0].user_id
    print(f"\nGenerating recommendations for user: {test_user}")

    # Chuẩn hóa user_index
    user_index_model = user_indexer.fit(hybrid_df)
    user_data = user_index_model.transform(hybrid_df.filter(col("user_id") == test_user)).limit(1)
    user_idx = user_data.select("user_index").first()[0]

    # Recommend top 5 sản phẩm cho user này
    recs = model.recommendForUserSubset(spark.createDataFrame([(user_idx,)], ["user_index"]), 5)
    recs = recs.select(explode("recommendations").alias("rec")).select(
        col("rec.item_index").alias("item_index"),
        col("rec.rating").alias("predicted_rating")
    )
    recs.show(truncate=False)
    print("Collaborative filtering test completed!")