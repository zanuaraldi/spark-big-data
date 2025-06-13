import org.apache.spark.sql.SparkSession
import breeze.linalg.{DenseMatrix => BDM}
import scala.util.Random
import scala.collection.mutable.ArrayBuffer

object MatrixMultiply {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: spark-submit MatrixMultiply.jar <matrix_size> <num_partitions>")
      System.exit(1)
    }
    
    val matrixSize = args(0).toInt
    val numPartitions = args(1).toInt
    
    val spark = SparkSession.builder()
      .appName(s"MatrixMultiplication-$matrixSize-$numPartitions")
      .config("spark.default.parallelism", numPartitions)
      .getOrCreate()
    
    val sc = spark.sparkContext
    
    println(s"Menjalankan perkalian matriks ${matrixSize}x${matrixSize} dengan $numPartitions partisi")
    
    val matA = Array.fill(matrixSize, matrixSize)(Random.nextDouble())
    val matB = Array.fill(matrixSize, matrixSize)(Random.nextDouble())
    
    val startTime = System.currentTimeMillis()
    val result = matrixMultiplySpark(sc, matA, matB, numPartitions)
    val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0
    
    println(s"Dimensi matriks: ${matrixSize}x${matrixSize}")
    println(s"Jumlah partisi/workers: $numPartitions")
    println(f"Waktu eksekusi: $elapsedTime%.4f detik")
    
    if (matrixSize <= 10) {
      println("Hasil perkalian matriks:")
      result.foreach(row => println(row.mkString("\t")))
    } else {
      println("Hasil perkalian matriks (5x5 pertama):")
      result.take(5).foreach(row => println(row.take(5).mkString("\t")))
    }
    
    spark.stop()
  }
  
  def matrixMultiplySpark(sc: org.apache.spark.SparkContext, 
                         matA: Array[Array[Double]], 
                         matB: Array[Array[Double]],
                         numPartitions: Int): Array[Array[Double]] = {
    val m = matA.length
    val n = matB(0).length
    val p = matB.length
    
    val matBT = transpose(matB)
    
    val rddA = sc.parallelize(matA.zipWithIndex, numPartitions)  // (row, rowIndex)
    val rddBT = sc.parallelize(matBT.zipWithIndex, numPartitions)  // (col, colIndex)
    
    val result = rddA.cartesian(rddBT)
      .map{ case ((row, i), (col, j)) =>
        ((i, j), row.zip(col).map{ case (a, b) => a * b }.sum)
      }
      .collectAsMap()
    
    Array.tabulate(m, n)((i, j) => result.getOrElse((i, j), 0.0))
  }
  
  def transpose(matrix: Array[Array[Double]]): Array[Array[Double]] = {
    val rows = matrix.length
    val cols = matrix(0).length
    val result = Array.fill(cols, rows)(0.0)
    
    for (i <- 0 until rows; j <- 0 until cols) {
      result(j)(i) = matrix(i)(j)
    }
    
    result
  }
}
