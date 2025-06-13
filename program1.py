from pyspark.sql import SparkSession
import numpy as np
import time
import sys

def matrix_multiply_spark(spark, matrix_a, matrix_b, block_size=100):
    """
    Melakukan perkalian matriks secara paralel dengan Spark
    Parameters:
        spark: SparkSession
        matrix_a: Matriks pertama (2D numpy array)
        matrix_b: Matriks kedua (2D numpy array)
        block_size: Ukuran blok untuk partisi (optimasi cache)
    """
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Dimensi matriks tidak sesuai untuk perkalian")
    
    rdd_a = spark.sparkContext.parallelize(matrix_a.tolist()).zipWithIndex()  # (row, row_index)
    rdd_b = spark.sparkContext.parallelize(matrix_b.T.tolist()).zipWithIndex()  # (col, col_index)
    
    result = rdd_a.cartesian(rdd_b) \
        .map(lambda x: ((x[0][1], x[1][1]), sum(a*b for a,b in zip(x[0][0], x[1][0])))) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: (x[0][0], (x[0][1], x[1]))) \
        .groupByKey() \
        .map(lambda x: (x[0], sorted(list(x[1]), key=lambda y: y[0]))) \
        .sortByKey() \
        .map(lambda x: [v for (i,v) in x[1]])
        
    return np.array(result.collect())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit matrix_multiply.py <matrix_size> <num_partitions>")
        sys.exit(1)
        
    matrix_size = int(sys.argv[1])
    num_partitions = int(sys.argv[2])
    
    spark = SparkSession.builder \
        .appName(f"MatrixMultiplication-{matrix_size}-{num_partitions}") \
        .config("spark.default.parallelism", num_partitions) \
        .getOrCreate()
    
    print(f"Menjalankan perkalian matriks {matrix_size}x{matrix_size} dengan {num_partitions} partisi")
    
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)
    
    start_time = time.time()
    result = matrix_multiply_spark(spark, matrix_a, matrix_b)
    elapsed_time = time.time() - start_time
    
    print(f"Dimensi matriks: {matrix_size}x{matrix_size}")
    print(f"Jumlah partisi/workers: {num_partitions}")
    print(f"Waktu eksekusi: {elapsed_time:.4f} detik")

    if matrix_size <= 10:
        print("Hasil perkalian matriks:")
        print(result)
    else:
        print("Hasil perkalian matriks (5x5 pertama):")
        print(result[:5, :5])
    
    spark.stop()
