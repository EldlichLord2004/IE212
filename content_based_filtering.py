import findspark
findspark.init()
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.sql.functions import col, explode, collect_list, regexp_replace, lower

# Create Spark session
spark = SparkSession.builder.appName("ContentBasedFiltering") \
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

def build_content_based_filtering(df):
    print("Building content-based filtering model...")
    categories_df = df.select("item_id", "categories").distinct()
    exploded_df = categories_df.select("item_id", explode("categories").alias("category"))
    category_features = exploded_df.groupBy("item_id").agg(collect_list("category").alias("category_features"))
    text_df = df.select("item_id", "clean_review_text").distinct()
    tokenizer = Tokenizer(inputCol="clean_review_text", outputCol="words")
    wordsData = tokenizer.transform(text_df)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    filtered_data = remover.transform(wordsData)
    word2vec = Word2Vec(vectorSize=100, minCount=1, inputCol="filtered_words", outputCol="text_features")
    w2v_model = word2vec.fit(filtered_data)
    text_features = w2v_model.transform(filtered_data)
    content_features = text_features.join(category_features, "item_id", "inner")
    w2v_model.write().overwrite().save("w2v_model")
    print("Word2Vec model saved.")
    return content_features, w2v_model, tokenizer, remover

if __name__ == "__main__":
    sample_size = 10000
    hybrid_df = prepare_data(user_reviews, metadata, sample_size)
    content_features, w2v_model, tokenizer, remover = build_content_based_filtering(hybrid_df)

    # Kiểm tra schema và show một vài dòng kết quả
    print("\nSchema of content_features:")
    content_features.printSchema()
    print("\nSample content-based features:")
    content_features.select("item_id", "category_features", "text_features").show(5, truncate=False)

    # Thống kê số lượng sản phẩm đã tạo đặc trưng
    count = content_features.count()
    print(f"\nTotal items with content-based features: {count}")

    # Nếu muốn, tính thử độ tương đồng giữa 2 sản phẩm bất kỳ
    from pyspark.ml.linalg import Vectors
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType

    def cosine_similarity(v1, v2):
        arr1 = v1.toArray()
        arr2 = v2.toArray()
        return float(np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2) + 1e-8))

    cosine_udf = udf(cosine_similarity, DoubleType())

    # Lấy 2 sản phẩm bất kỳ
    items = content_features.select("item_id", "text_features").limit(2).collect()
    if len(items) == 2:
        v1 = items[0]["text_features"]
        v2 = items[1]["text_features"]
        sim = cosine_similarity(v1, v2)
        print(f"\nCosine similarity between item {items[0]['item_id']} and {items[1]['item_id']}: {sim:.4f}")

    print("Content-based filtering test completed!")