from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import numpy as np
import time
import sys

def create_spark_session():
    """Membuat Spark session dengan konfigurasi yang optimal"""
    try:
        # Hentikan session yang ada jika ada
        existing_spark = SparkSession.getActiveSession()
        if existing_spark:
            existing_spark.stop()
    except:
        pass

    conf = SparkConf() \
        .setAppName("MatrixMultiplication") \
        .setMaster("local[*]") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "2g") \
        .set("spark.executor.cores", "4") \
        .set("spark.sql.adaptive.enabled", "true") \
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .set("spark.network.timeout", "600s") \
        .set("spark.executor.heartbeatInterval", "60s") \
        .set("spark.dynamicAllocation.enabled", "false")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark

def matrix_multiply_spark_optimized(spark, matrix_a, matrix_b, num_partitions=None):
    """
    Perkalian matriks menggunakan Spark dengan optimasi
    """
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Dimensi matriks tidak sesuai untuk perkalian")

    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape

    # Tentukan jumlah partisi optimal
    if num_partitions is None:
        num_partitions = min(spark.sparkContext.defaultParallelism * 2, rows_a // 10)
        num_partitions = max(num_partitions, 1)

    print(f"Menggunakan {num_partitions} partisi")

    # Broadcast matrix B untuk efisiensi
    broadcast_b = spark.sparkContext.broadcast(matrix_b)

    def multiply_row(row_data):
        """Mengalikan satu baris matrix A dengan seluruh matrix B"""
        row_idx, row_values = row_data
        matrix_b_local = broadcast_b.value
        result_row = []

        for col_idx in range(matrix_b_local.shape[1]):
            dot_product = sum(row_values[k] * matrix_b_local[k, col_idx]
                            for k in range(len(row_values)))
            result_row.append(dot_product)

        return (row_idx, result_row)

    # Konversi matrix A ke RDD dengan indeks
    rdd_a = spark.sparkContext.parallelize(
        [(i, matrix_a[i].tolist()) for i in range(rows_a)],
        num_partitions
    )

    # Lakukan perkalian
    result_rdd = rdd_a.map(multiply_row)

    # Kumpulkan hasil dan urutkan berdasar indeks baris
    result_list = result_rdd.collect()
    result_list.sort(key=lambda x: x[0])

    # Konversi ke numpy array
    result_matrix = np.array([row[1] for row in result_list])

    # Bersihkan broadcast variable
    broadcast_b.unpersist()

    return result_matrix

def matrix_multiply_block_wise(spark, matrix_a, matrix_b, block_size=100):
    """
    Implementasi perkalian matriks dengan pembagian blok untuk matriks besar
    """
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Dimensi matriks tidak sesuai untuk perkalian")

    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape

    # Broadcast matrix B
    broadcast_b = spark.sparkContext.broadcast(matrix_b)

    def process_block(block_data):
        """Memproses blok dari matrix A"""
        start_row, end_row = block_data
        matrix_b_local = broadcast_b.value

        block_result = []
        for i in range(start_row, min(end_row, rows_a)):
            row_result = []
            for j in range(cols_b):
                dot_product = sum(matrix_a[i, k] * matrix_b_local[k, j]
                                for k in range(cols_a))
                row_result.append(dot_product)
            block_result.append((i, row_result))

        return block_result

    # Buat blok-blok
    blocks = [(i, min(i + block_size, rows_a))
              for i in range(0, rows_a, block_size)]

    # Proses blok secara paralel
    rdd_blocks = spark.sparkContext.parallelize(blocks)
    result_rdd = rdd_blocks.flatMap(process_block)

    # Kumpulkan dan urutkan hasil
    result_list = result_rdd.collect()
    result_list.sort(key=lambda x: x[0])

    result_matrix = np.array([row[1] for row in result_list])

    broadcast_b.unpersist()
    return result_matrix

def simple_test_1000x1000(spark):
    """Test sederhana khusus untuk matriks 1000x1000"""
    print("\n" + "="*60)
    print("TEST SEDERHANA MATRIKS 1000x1000")
    print("="*60)

    # Buat matriks 1000x1000
    print("Membuat matriks random 1000x1000...")
    matrix_a = np.random.rand(1000, 1000).astype(np.float32)
    matrix_b = np.random.rand(1000, 1000).astype(np.float32)

    results = {}

    try:
        # Test 1: NumPy (baseline)
        print("\n1. NumPy Baseline:")
        start_time = time.time()
        np_result = np.dot(matrix_a, matrix_b)
        np_time = time.time() - start_time
        results['numpy'] = np_time
        print(f"   ✓ Selesai dalam {np_time:.4f} detik")

        # Test 2: Spark Optimized
        print("\n2. Spark Optimized Method:")
        start_time = time.time()
        spark_result = matrix_multiply_spark_optimized(spark, matrix_a, matrix_b)
        spark_time = time.time() - start_time
        results['spark_optimized'] = spark_time
        print(f"   ✓ Selesai dalam {spark_time:.4f} detik")

        # Verifikasi kebenaran hasil
        diff = np.max(np.abs(np_result - spark_result))
        print(f"   ✓ Perbedaan maksimum: {diff:.8f}")

        if diff < 1e-4:
            print("   ✓ Hasil akurat!")
        else:
            print("   ⚠ Hasil kurang akurat")

        # Test 3: Spark Block-wise
        print("\n3. Spark Block-wise Method:")
        start_time = time.time()
        block_result = matrix_multiply_block_wise(spark, matrix_a, matrix_b, block_size=200)
        block_time = time.time() - start_time
        results['spark_blocks'] = block_time
        print(f"   ✓ Selesai dalam {block_time:.4f} detik")

        # Verifikasi block-wise
        diff_block = np.max(np.abs(np_result - block_result))
        print(f"   ✓ Perbedaan maksimum: {diff_block:.8f}")

        if diff_block < 1e-4:
            print("   ✓ Hasil akurat!")
        else:
            print("   ⚠ Hasil kurang akurat")

        # Tampilkan hasil
        print(f"\n4. Hasil Perkalian (sampel 5x5 pertama):")
        print(spark_result[:5, :5])

        # Analisis performa
        print(f"\n5. Analisis Performa:")
        print(f"   NumPy time:           {results['numpy']:.4f} detik")
        print(f"   Spark Optimized:      {results['spark_optimized']:.4f} detik")
        print(f"   Spark Block-wise:     {results['spark_blocks']:.4f} detik")

        if results['spark_optimized'] < results['numpy']:
            speedup = results['numpy'] / results['spark_optimized']
            print(f"   ✓ Spark Optimized {speedup:.2f}x lebih cepat dari NumPy")
        else:
            slowdown = results['spark_optimized'] / results['numpy']
            print(f"   ✗ Spark Optimized {slowdown:.2f}x lebih lambat dari NumPy")

        if results['spark_blocks'] < results['numpy']:
            speedup = results['numpy'] / results['spark_blocks']
            print(f"   ✓ Spark Block-wise {speedup:.2f}x lebih cepat dari NumPy")
        else:
            slowdown = results['spark_blocks'] / results['numpy']
            print(f"   ✗ Spark Block-wise {slowdown:.2f}x lebih lambat dari NumPy")

        # Pilih metode terbaik
        best_method = min(results.items(), key=lambda x: x[1])
        print(f"\n6. Metode Terbaik: {best_method[0]} ({best_method[1]:.4f} detik)")

        return True, results

    except Exception as e:
        print(f"\n✗ Error dalam test 1000x1000: {e}")
        return False, {}

def benchmark_methods(spark, size=500):
    """Benchmark berbagai metode perkalian matriks"""
    print(f"Membuat matriks random {size}x{size}")
    matrix_a = np.random.rand(size, size).astype(np.float32)
    matrix_b = np.random.rand(size, size).astype(np.float32)

    results = {}

    # NumPy baseline
    print("\n1. NumPy (baseline):")
    start = time.time()
    np_result = np.dot(matrix_a, matrix_b)
    np_time = time.time() - start
    results['numpy'] = np_time
    print(f"   Waktu: {np_time:.4f} detik")

    # Spark optimized
    print("\n2. Spark Optimized:")
    start = time.time()
    spark_result = matrix_multiply_spark_optimized(spark, matrix_a, matrix_b)
    spark_time = time.time() - start
    results['spark_optimized'] = spark_time
    print(f"   Waktu: {spark_time:.4f} detik")

    # Verifikasi hasil
    diff = np.max(np.abs(np_result - spark_result))
    print(f"   Perbedaan maksimum dengan NumPy: {diff:.8f}")

    # Spark block-wise (untuk matriks besar)
    if size <= 1000:
        print("\n3. Spark Block-wise:")
        start = time.time()
        block_result = matrix_multiply_block_wise(spark, matrix_a, matrix_b, block_size=100)
        block_time = time.time() - start
        results['spark_blocks'] = block_time
        print(f"   Waktu: {block_time:.4f} detik")

        diff_block = np.max(np.abs(np_result - block_result))
        print(f"   Perbedaan maksimum dengan NumPy: {diff_block:.8f}")

    return results, spark_result

def main():
    spark = None
    try:
        print("Memulai aplikasi perkalian matriks dengan Spark...")
        spark = create_spark_session()

        print(f"Spark Context: {spark.sparkContext.appName}")
        print(f"Master: {spark.sparkContext.master}")
        print(f"Default Parallelism: {spark.sparkContext.defaultParallelism}")

        # TAMBAHAN: Test sederhana 1000x1000 terlebih dahulu
        print("\n🚀 MENJALANKAN TEST SEDERHANA 1000x1000")
        success, simple_results = simple_test_1000x1000(spark)

        if success:
            print("\n✓ Test sederhana 1000x1000 berhasil!")
        else:
            print("\n✗ Test sederhana 1000x1000 gagal, melanjutkan ke test lainnya...")

        # Test dengan ukuran berbeda
        test_sizes = [100, 500, 1000]

        for size in test_sizes:
            print(f"\n{'='*50}")
            print(f"BENCHMARK UNTUK MATRIKS {size}x{size}")
            print(f"{'='*50}")

            try:
                results, final_result = benchmark_methods(spark, size)

                print(f"\nHasil perkalian (5x5 pertama):")
                print(final_result[:5, :5])

                print(f"\nPerbandingan waktu:")
                for method, time_taken in results.items():
                    print(f"  {method}: {time_taken:.4f} detik")

                if 'numpy' in results and 'spark_optimized' in results:
                    speedup = results['numpy'] / results['spark_optimized']
                    print(f"  Speedup Spark vs NumPy: {speedup:.2f}x")

            except Exception as e:
                print(f"Error pada ukuran {size}: {e}")
                continue

            # Pause sejenak untuk monitoring
            time.sleep(2)

    except Exception as e:
        print(f"Error dalam aplikasi utama: {e}")

        # Fallback ke mode lokal jika cluster gagal
        print("\nMencoba fallback ke mode lokal...")
        try:
            if spark:
                spark.stop()

            spark = SparkSession.builder \
                .appName("MatrixMultiplication") \
                .master("local[*]") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "1g") \
                .getOrCreate()

            print("✓ Berhasil membuat Spark session lokal")

            # Jalankan test sederhana dengan mode lokal
            print("\n🔄 MENJALANKAN TEST 1000x1000 DALAM MODE LOKAL")
            success, results = simple_test_1000x1000(spark)

            if success:
                print("\n✓ Test dalam mode lokal berhasil!")
            else:
                print("\n✗ Test dalam mode lokal juga gagal")

        except Exception as e2:
            print(f"✗ Fallback ke mode lokal juga gagal: {e2}")
            sys.exit(1)

    finally:
        if spark:
            print("\nMenghentikan Spark session...")
            spark.stop()
            print("Spark session dihentikan.")

if _name_ == "_main_":
    main()
