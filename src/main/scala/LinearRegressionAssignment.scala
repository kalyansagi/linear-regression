
import org.apache.spark
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.regression.LinearRegression


object LinearRegressionAssignment {


  def main(args: Array[String]) {
    //spark context
    val sc = new SparkContext(new SparkConf().setAppName("Spark Word Count").setMaster("local"))

    //reading the input
    val training = spark.read.format("libsvm").load("src/main/resources/sample_linear_regression.txt")

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    //Fit the model
    val lrModel = lr.fit(training)

    //print coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


    //summarize the model over training set and print some metrics.
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

  }
}
