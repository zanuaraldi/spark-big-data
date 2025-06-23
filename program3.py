from pyspark.sql import SparkSession
import numpy as np
import time
import matplotlib.pyplot as plt
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
    # Validasi dimensi matriks
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Dimensi matriks tidak sesuai untuk perkalian")

    # Konversi matriks ke RDD
    rdd_a = spark.sparkContext.parallelize(matrix_a.tolist()).zipWithIndex()  # (row, row_index)
    rdd_b = spark.sparkContext.parallelize(matrix_b.T.tolist()).zipWithIndex()  # (col, col_index)

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

def run_experiment(dimensions, worker_counts):
    results = []
    
    for dim in dimensions:
        dim_results = []
        for workers in worker_counts:
            # Configure Spark with specified number of workers
            spark = SparkSession.builder \
                .appName(f"MatrixMultiplication-{dim}x{dim}-{workers}workers") \
                .master(f"local[{workers}]") \
                .getOrCreate()
            
            print(f"Running experiment with {dim}x{dim} matrices using {workers} workers...")
            
            # Generate random matrices
            matrix_a = np.random.rand(dim, dim)
            matrix_b = np.random.rand(dim, dim)
            
            # Time the multiplication
            start_time = time.time()
            result = matrix_multiply_spark(spark, matrix_a, matrix_b)
            end_time = time.time()
            
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.2f} seconds")
            
            dim_results.append((workers, execution_time))
            spark.stop()
            
        results.append((dim, dim_results))
    
    return results

def plot_results(results):
    plt.figure(figsize=(12, 8))
    
    for dim, dim_results in results:
        workers, times = zip(*dim_results)
        plt.plot(workers, times, marker='o', label=f"{dim}x{dim}")
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Matrix Multiplication Performance: PySpark')
    plt.legend()
    plt.grid(True)
    plt.savefig('pyspark_performance.png')
    plt.close()
    
    # Plot speedup relative to 5 workers
    plt.figure(figsize=(12, 8))
    
    for dim, dim_results in results:
        workers, times = zip(*dim_results)
        base_time = times[0]  # Time with 5 workers
        speedups = [base_time/time for time in times]
        plt.plot(workers, speedups, marker='o', label=f"{dim}x{dim}")
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup (relative to 5 workers)')
    plt.title('Matrix Multiplication Speedup: PySpark')
    plt.legend()
    plt.grid(True)
    plt.savefig('pyspark_speedup.png')
    plt.close()

if __name__ == "__main__":
    # Define dimensions and worker counts
    dimensions = [1000, 10000, 20000, 40000]
    worker_counts = [5, 10, 20]
    
    # For testing with smaller values
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        dimensions = [100, 200]
        worker_counts = [2, 4]
    
    # Run experiments
    results = run_experiment(dimensions, worker_counts)
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\nPerformance Summary:")
    print("===================")
    
    for dim, dim_results in results:
        print(f"\nMatrix Dimension: {dim}x{dim}")
        print("--------------------------")
        for workers, time in dim_results:
            print(f"Workers: {workers}, Time: {time:.2f} seconds")
        
        # Calculate speedup
        base_time = dim_results[0][1]  # Time with minimum workers
        print("\nSpeedup (relative to minimum workers):")
        for workers, time in dim_results:
            speedup = base_time / time
            efficiency = speedup / workers * 5  # Efficiency relative to 5 workers
            print(f"Workers: {workers}, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}")