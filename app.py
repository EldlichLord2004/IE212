from flask import Flask, request, render_template
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Khởi tạo Spark session (dùng lại config như trong model của bạn)
spark = SparkSession.builder.appName("HybridRecommendationSystemWeb") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load dữ liệu (hoặc load từ pickle nếu đã lưu mô hình)
user_reviews = spark.read.parquet("Dataset/user_reviews.parquet")
metadata = spark.read.parquet("Dataset/metadata.parquet")

# Giả sử bạn đã có hàm build_hybrid_recommendation_system như trên
from recommendation_system import build_hybrid_recommendation_system

# Khởi tạo hàm lấy dự đoán
get_recommendations = build_hybrid_recommendation_system(user_reviews, metadata, sample_size=100000)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    asin = None
    predicted_rating = None
    history = []
    product_title = None

    if request.method == 'POST':
        asin = request.form['asin'].strip()
        # Lấy lịch sử rating
        history_df = user_reviews.filter(col("asin") == asin).select("user_id", "rating", "text")
        history = history_df.limit(20).toPandas().to_dict(orient='records')

        # Lấy thông tin sản phẩm
        product = metadata.filter(col("parent_asin") == asin).select("title").limit(1).collect()
        product_title = product[0].title if product else "Không tìm thấy"

        # Dự đoán rating trung bình (nếu muốn dùng mô hình collaborative filtering)
        # Hoặc lấy rating trung bình thực tế
        avg_rating_row = user_reviews.filter(col("asin") == asin).agg({"rating": "avg"}).collect()
        predicted_rating = round(avg_rating_row[0][0], 2) if avg_rating_row and avg_rating_row[0][0] else "Không có dữ liệu"

    return render_template('index.html', asin=asin, predicted_rating=predicted_rating, history=history, product_title=product_title)

if __name__ == '__main__':
    app.run(debug=True)