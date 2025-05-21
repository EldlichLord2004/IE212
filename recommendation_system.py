import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, CountVectorizer, HashingTF, IDF, Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode, collect_list, udf, array_contains, regexp_replace, lower
from pyspark.sql.types import FloatType, ArrayType, StringType
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
import numpy as np

# Create a Spark session
spark = SparkSession.builder.appName("HybridRecommendationSystem") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

# Load the datasets
user_reviews = spark.read.parquet("Dataset/user_reviews.parquet")
metadata = spark.read.parquet("Dataset/metadata.parquet")



# Define a function to load and prepare data
def prepare_data(user_reviews_df, metadata_df, sample_size=None):
    # Sample data if needed
    if sample_size:
        reviews_df = user_reviews_df.limit(sample_size)
    else:
        reviews_df = user_reviews_df
    
    # Join with metadata
    complete_df = reviews_df.join(metadata_df, reviews_df.parent_asin == metadata_df.parent_asin, "inner")
    
    # Create hybrid dataframe with selected columns
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
    
    # Clean review text
    hybrid_df = hybrid_df.withColumn(
        "clean_review_text", 
        regexp_replace(lower(col("review_text")), "[^a-zA-Z0-9 ]", " ")
    )
    
    return hybrid_df

# COLLABORATIVE FILTERING COMPONENT
def build_collaborative_filtering(df):
    print("Building collaborative filtering model...")
    
    # Create user and item indices
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="item_id", outputCol="item_index", handleInvalid="skip")
    
    # Prepare data
    indexed_df = user_indexer.fit(df).transform(df)
    indexed_df = item_indexer.fit(indexed_df).transform(indexed_df)
    
    # Split data
    (training, test) = indexed_df.randomSplit([0.8, 0.2], seed=42)
    
    # Build ALS model
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
    
    # Train model
    model = als.fit(training)
    
    # Make predictions
    predictions = model.transform(test)
    
    # Evaluate
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Collaborative Filtering RMSE = {rmse}")
    
    return model, user_indexer, item_indexer

# CONTENT-BASED FILTERING COMPONENT
def build_content_based_filtering(df):
    print("Building content-based filtering model...")
    
    # Process categories
    categories_df = df.select("item_id", "categories").distinct()
    
    # Flatten categories array
    exploded_df = categories_df.select("item_id", explode("categories").alias("category"))
    
    # Create category features
    category_features = exploded_df.groupBy("item_id").agg(collect_list("category").alias("category_features"))
    
    # Process review text
    text_df = df.select("item_id", "clean_review_text").distinct()
    
    # Tokenize text
    tokenizer = Tokenizer(inputCol="clean_review_text", outputCol="words")
    wordsData = tokenizer.transform(text_df)
    
    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    filtered_data = remover.transform(wordsData)
    
    # Create TF-IDF features
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=1000)
    featurized_data = hashingTF.transform(filtered_data)
    
    idf = IDF(inputCol="rawFeatures", outputCol="text_features")
    idf_model = idf.fit(featurized_data)
    text_features = idf_model.transform(featurized_data)
    
    # Combine text features with category features
    content_features = text_features.join(category_features, "item_id", "inner")
    
    return content_features, idf_model, hashingTF, tokenizer, remover

# HYBRID RECOMMENDATION SYSTEM
def build_hybrid_recommendation_system(user_reviews_df, metadata_df, sample_size=50000):
    # Prepare data
    hybrid_df = prepare_data(user_reviews_df, metadata_df, sample_size)
    
    # Set checkpoint directory to help with memory issues
    sc = spark.sparkContext
    sc.setCheckpointDir("/tmp/checkpoint")
    
    # Build collaborative filtering model
    cf_model, user_indexer, item_indexer = build_collaborative_filtering(hybrid_df)
    
    # Build content-based filtering model
    content_features, idf_model, hashingTF, tokenizer, remover = build_content_based_filtering(hybrid_df)
    
    # Pre-compute models and mappings to avoid serialization issues
    user_index_model = user_indexer.fit(hybrid_df)
    item_index_model = item_indexer.fit(hybrid_df)
    
    # Pre-compute item mapping
    item_mapping = item_index_model.transform(hybrid_df).select("item_id", "item_index").distinct()
    
    # Create a function to combine recommendations
    def get_recommendations(user_id, n=10):
        # Get user index
        user_data = hybrid_df.filter(col("user_id") == user_id).limit(1)
        if user_data.count() == 0:
            print(f"User {user_id} not found")
            return None
        
        # Use pre-fitted model instead of fitting inside function
        user_data = user_index_model.transform(user_data)
        user_idx = user_data.select("user_index").first()[0]
        
        # Get collaborative filtering recommendations
        cf_recs = cf_model.recommendForUserSubset(spark.createDataFrame([(user_idx,)], ["user_index"]), n)
        if cf_recs.count() == 0:
            print("No collaborative filtering recommendations found")
            return None
        
        cf_recs = cf_recs.select(
            col("user_index"),
            explode("recommendations").alias("recommendation")
        ).select(
            col("user_index"),
            col("recommendation.item_index").alias("item_index"),
            col("recommendation.rating").alias("cf_score")
        )
        
        # Use pre-computed mapping instead of transforming again
        cf_recs = cf_recs.join(item_mapping, "item_index", "inner")
        
        # Add content-based features for diversity
        # Get user's category preferences
        user_categories = hybrid_df.filter(col("user_id") == user_id) \
            .select(explode("categories").alias("category")) \
            .groupBy("category") \
            .count() \
            .orderBy(col("count").desc())
        
        # Convert to list of categories
        if user_categories.count() > 0:
            preferred_categories = [row.category for row in user_categories.collect()]
        else:
            preferred_categories = []
        
        # Collect content features for local processing to avoid serialization issues
        content_features_collected = content_features.select("item_id", "category_features").collect()
        content_features_dict = {row["item_id"]: row["category_features"] for row in content_features_collected}
        
        # Define a simpler UDF that doesn't depend on SparkContext
        def calculate_content_score(item_id):
            if item_id in content_features_dict and preferred_categories:
                item_cat_list = content_features_dict[item_id]
                overlap = sum(1 for cat in preferred_categories[:3] if cat in item_cat_list)
                return float(overlap) / max(len(preferred_categories[:3]), 1)
            return 0.0
        
        calculate_content_score_udf = udf(calculate_content_score, FloatType())
        
        # Add content score to recommendations
        recommendations = cf_recs.withColumn("content_score", calculate_content_score_udf(col("item_id")))
        
        # Calculate hybrid score (weighted combination)
        recommendations = recommendations.withColumn(
            "hybrid_score", 
            col("cf_score") * 0.7 + col("content_score") * 0.3
        )
        
        # Get final recommendations
        final_recs = recommendations.join(
            hybrid_df.select("item_id", "product_title", "average_rating").distinct(), 
            "item_id", 
            "inner"
        ).orderBy(col("hybrid_score").desc())
        
        return final_recs.select(
            "item_id", 
            "product_title", 
            "average_rating",
            "hybrid_score"
        ).limit(n)
    
    return get_recommendations

# Execute the hybrid recommendation system
def main():
    print("Building hybrid recommendation system...")
    
    # Set a sample size to control memory usage
    sample_size = 1000000
    
    # Build the recommendation function
    get_recommendations = build_hybrid_recommendation_system(user_reviews, metadata, sample_size)
    
    # Test with a random user
    test_user = user_reviews.select("user_id").distinct().limit(1).collect()[0].user_id
    print(f"\nGenerating recommendations for user: {test_user}")
    
    recommendations = get_recommendations(test_user, n=10)
    if recommendations:
        print("\nTop 10 Recommendations:")
        recommendations.show(10, False)
    
    print("Recommendation system built successfully!")

if __name__ == "__main__":
    main() 