# %%
# 写一个代码实例，模拟一个dataframe，进行label编码的过程

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor
from pyspark.sql import Row
from tools import find_project_root

rootpath = find_project_root()

# 创建spark session
spark = SparkSession.builder.appName("Label Encoding").getOrCreate()

# 读取data/install_app_impressions文件夹下面的所有.parquet文件并汇总到一个dataframe中

df = spark.read.parquet(str(rootpath) + "/" + "data/raw2/*.parquet")

# %%
# 显示数据长度
df.show(10)
print(df.count())

# %%
# 创建一个train数据集，筛选df中req_time小于2025-07-15 00:00:00.000的数据
train_df = df.filter(df.req_time < "2025-07-15 00:00:00.000")
test_df = df.filter(df.req_time >= "2025-07-15 00:00:00.000")
print(train_df.count(), test_df.count())
# 清空df，节省内存
df = None

# %%
## 对train_df以及test_df分批次存储为.parquet文件,5万条每一个文件
def save_in_batches(df, batch_size, output_dir):
    # 加行号（注意：这一行才会触发真正计算）
    rdd_with_index = df.rdd.zipWithIndex().map(lambda x: Row(index=x[1], **x[0].asDict()))
    indexed_df = spark.createDataFrame(rdd_with_index)
    
    # 按照行号分 batch_id
    indexed_df = indexed_df.withColumn("batch_id", floor(col("index") / batch_size))

    # 写出，每个 batch_id 是一个 parquet 分区
    indexed_df.write \
        .partitionBy("batch_id") \
        .mode("overwrite") \
        .parquet(output_dir)

    print("写入完成，每个分区约含 {} 行".format(batch_size))
# %%
## 对train和test数据分批保存
save_in_batches(train_df, batch_size=30000, output_dir=str(rootpath) + "/data/processed/train_parquet")
save_in_batches(test_df, batch_size=30000, output_dir=str(rootpath) + "/data/processed/test_parquet")

# %%
