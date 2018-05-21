name := "project3_final"

version := "0.1"

scalaVersion := "2.10.6"

scalacOptions ++= Seq("-deprecation")

resolvers += "Spark Packages Repo" at "https://dl.bintray.com/spark-packages/maven"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.2",
  "org.apache.spark" %% "spark-sql" % "1.6.2",
  "org.apache.spark" %% "spark-mllib" % "1.6.2",
  "org.apache.spark" %% "spark-streaming" % "1.6.2",
  "org.apache.spark" %% "spark-streaming-kafka" % "1.6.2",
  "com.datastax.spark" %% "spark-cassandra-connector" % "1.6.0",
  "junit" % "junit" % "4.10" % "test"
)
        