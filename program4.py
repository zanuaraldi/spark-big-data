from pyspark.sql import SparkSession
import numpy as np
def matrix_multiply_spark(spark, matrix_a, matrix_b, block_size=100):
 """
 Melakukan perkalian matriks secara paralel dengan Spark

 Parameters:
 spark: SparkSession
 matrix_a: Matriks pertama (2D numpy array)
 matrix_b: Matriks kedua (2D numpy array)
 block_size: Ukuran blok untuk partisi (optimasi cache)
 """
 # Validasi dimensi matriks
 if matrix_a.shape[1] != matrix_b.shape[0]: 
  raise ValueError("Dimensi matriks tidak sesuai untuk perkalian")

 # Konversi matriks ke RDD
 rdd_a = spark.sparkContext.parallelize(matrix_a.tolist()).zipWithIndex() # (row, row_index)
 rdd_b = spark.sparkContext.parallelize(matrix_b.T.tolist()).zipWithIndex() # (col, col_index)

 # Buat produk kartesian dan hitung perkalian per elemen
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
 # Inisialisasi Spark
 spark = SparkSession.builder \
 .appName("MatrixMultiplication") \
 .getOrCreate()

 # Contoh matriks
 matrix_a = np.random.rand(100, 100) # Matriks 100x100
 matrix_b = np.random.rand(100, 100) # Matriks 100x100

 # Hitung perkalian matriks
 result = matrix_multiply_spark(spark, matrix_a, matrix_b)

 print("Hasil perkalian matriks (5x5 pertama):")
 print(result[:5, :5])

 spark.stop()