package kafka_spark

import kafka.serializer.StringDecoder
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.kafka.KafkaUtils
import com.datastax.spark.connector.streaming._

/*stream from kafka*/
object ReviewsStreamingSansClassification {
  //val localLogger = Logger.getLogger("ReviewsStream")
  def main(args: Array[String]) {
    // 1.spark configuration
    val sparkConf = new SparkConf().setAppName("MoviesReviewsSentiments").setMaster("local")
    sparkConf.setIfMissing("spark.cassandra.connection.host", "127.0.0.1")

    //spark streaming context et parametres de kafka
    val ssc = new StreamingContext(sparkConf, Seconds(5))
    val kafkaTopicRaw = "reviews" //le nom du topic kafka
    val kafkaBroker = "127.0.0.1:9092"

    //declaration des données cassandra (keyspace, table)
    val cassandraKeyspace = "reviewstopredic"
    val cassandraTable = "reviewspredicted"

    //passer le StreamingContext, Kafka config map  et kafka topics à la fonction createDirectStream.
    //le type de kafka message (value) et kafka key = string, donc on a besoin de StringDecoders
    val topics: Set[String] = kafkaTopicRaw.split(",").map(_.trim).toSet    //en cas de plusieurs topics
    val kafkaParams = Map[String, String]("metadata.broker.list" -> kafkaBroker)    //j'ai utilisé un seul broker kafka

    //affichage des info à propos de kafka
    println(s"kafka broker = $kafkaBroker  et kafka topic =  $topics")

    ///creation d'un input stream qui poul des messages à partir de kafka. les paramètres sont (spark streaming context, params kafka, set topics kafka)
    /// résultat = InputDStream[(String,String)] = un RDD[(Kafka message key, kafka message value)]
    val rawReviewsStream : InputDStream[(String,String)] = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](ssc, kafkaParams, topics)
    rawReviewsStream.print     //pour imprimer
    //rawReviewsStream.foreachRDD(println(_))

    //transformation des DStream : la fonction "processing"  formate le DStream, predit les sentiments (en utilisant notre modèle)
    // et prepare les données à inserer dans cassandra la forme d'un RDD d'objet MyRow (case class)
    val resultReviewsStream : DStream[MyRow] = processing(rawReviewsStream)

    //writing to cassandra via la fonction writeToCassandra
    writeToCassandra(cassandraKeyspace, cassandraTable, resultReviewsStream)
    //resultReviewsStream.print    //pour voir les objets de type MyRow

    //lancement du streaming
    ssc.start()
    ssc.awaitTermination()
    ssc.stop()
  }

  //fonction qui traite le DStream
  def processing(rawReviewsStream: InputDStream[(String, String)]): DStream[MyRow] = {
    val parsedReviewsStream =  rawReviewsStream.map(_._2.split("\t"))
    //faire le calcule de prediction de sentiment
    val commentRDD = parsedReviewsStream.map(x => x(1))
    val commentIdRDD = parsedReviewsStream.map(x => x(0))
    //les traitements
    val resultReviewsStream : DStream[MyRow] = parsedReviewsStream.map(x => MyRow(x(0).toString,0,x(1).toString))
    resultReviewsStream
  }

  //fonction qui sauvegarde les données traitées dans cassandra
  def writeToCassandra(keyspace: String, table: String, value: DStream[MyRow]):Unit ={
    value.saveToCassandra(keyspace, table)
  }

  //fonction qui utilise mon model de regression logistic pour predire les sentiments (0/1)
}
