package kafka_spark

import java.io.File
import kafka.serializer.StringDecoder
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.kafka.KafkaUtils
import com.datastax.spark.connector.streaming._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.classification.LogisticRegressionModel

object Spark {
  // 1.spark configuration pour eviter le bug de la sérialisation que j'ai eu
  val sparkConf = new SparkConf().setAppName("MoviesReviewsSentimentClassification").setMaster("local").setIfMissing("spark.cassandra.connection.host", "127.0.0.1")
  val ssc = new StreamingContext(sparkConf, Seconds(5))
  val sc =ssc.sparkContext
  //d'autres données utiles pour éviter de les créer à chaque fois
  val dim1 = math.pow(2, 18).toInt //vector de taille 262144 (ce choix est basé sur l'analyse de nos données de train, il y a 50.000 mots)
  val hashingTF = new HashingTF(dim1)
  //mon modèle de regression logistic est dans le repertoire ressources/myModelLogisticRegression)
  val resourceModel = getClass.getClassLoader.getResource("myModelLogisticRegression")
  val filePathModel = new File(resourceModel.toURI).getPath
  val model = LogisticRegressionModel.load(Spark.sc, filePathModel)
}

/*stream from kafka*/
object ReviewsStreaming {
   def main(args: Array[String]) {
    //declaration des paramètres de kafka
    val kafkaTopicRaw = "reviews" //le nom du topic kafka
    val kafkaBroker = "127.0.0.1:9092"

    //declaration des données cassandra (keyspace, table)
    val cassandraKeyspace = "reviewstopredic"
    val cassandraTable = "reviewspredicted"

    //passer le StreamingContext, Kafka config map  et kafka topics à la fonction createDirectStream.
    //le type de kafka message (value) et kafka key = string, donc on a besoin de StringDecoders
    val topics: Set[String] = kafkaTopicRaw.split(",").map(_.trim).toSet //en cas de plusieurs topics
    val kafkaParams = Map[String, String]("metadata.broker.list" -> kafkaBroker) //j'ai utilisé un seul broker kafka

    //affichage des info à propos de kafka
    println(s"kafka broker = $kafkaBroker  et kafka topic =  $topics")

    ///creation d'un input stream qui poul des messages à partir de kafka. les paramètres sont (spark streaming context, params kafka, set topics kafka)
    /// résultat = InputDStream[(String,String)] = un RDD[(Kafka message key, kafka message value)]
    val rawReviewsStream: InputDStream[(String, String)] = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](Spark.ssc, kafkaParams, topics)
    rawReviewsStream.print //pour imprimer
    //rawReviewsStream.foreachRDD(println(_))

    //transformation des DStream : la fonction "processing"  formate le DStream, predit les sentiments (en utilisant notre modèle)
    // et prepare les données à inserer dans cassandra (case class MyRow)
    val resultReviewsStream: DStream[MyRow] = processing(rawReviewsStream)

    //writing to cassandra via la fonction writeToCassandra
    writeToCassandra(cassandraKeyspace, cassandraTable, resultReviewsStream)
    //resultReviewsStream.print    //pour voir les objets de type MyRow

    //lancement du streaming
    Spark.ssc.start()
    Spark.ssc.awaitTermination()
    Spark.ssc.stop()
  }

  //fonction qui traite le DStream
  def processing(rawReviewsStream: InputDStream[(String, String)]): DStream[MyRow] = {
    val parsedReviewsStream = rawReviewsStream.map(_._2.split("\t")) //prendre la valeur du message v=(commentId,comment)
    //formatter le resultat comme un raw pour la table cassandra.
    val resultReviewsStream: DStream[MyRow] = parsedReviewsStream.map(x => MyRow(x(0).toString, classification(x(1).toString).toInt, x(1).toString)) //commentId,sentiemnt, comment
    //val resultReviewsStream: DStream[MyRow] = parsedReviewsStream.map(x => MyRow(x(0).toString, 1, x(1).toString)) //commentId,sentiemnt, comment
    resultReviewsStream
  }

  //fonction qui sauvegarde les données traitées dans cassandra
  def writeToCassandra(keyspace: String, table: String, value: DStream[MyRow]): Unit = {
    value.saveToCassandra(keyspace, table)
  }

  //fonction de classification qui utilise mon model de regression logistic pour predire les sentiments (0/1)
  def classification(text: String): Int = {
    //tokeniser (eclater le text et le filtrer) et appeler le modele pour retourner le sentiment (0/1)
    val tokens = (tokenize(text))
    var predictedSentiment = Spark.model.predict(Spark.hashingTF.transform(tokens))
    predictedSentiment.toInt
  }

  //function for créer des mots à partir d'un texte (line) puis de filtrer ces mots
  def tokenize(line: String): Seq[String] = {
    val regex = """[^0-9]*""".r //expression à utiliser pour filter les mots comportant des chiffres
    val stopwords = Set("the", "a", "an", "of", "or", "in", "for", "by", "on", "but", "is", "not", "with", "as", "was", "if", "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to")
    //stopwords = un ensemble de mots très commun en anglais, à ignorer
    //line.split("""\W+""") permet d'eclater la ligne en mots en utilisant les blancs, les signes de ponctuation, les caractères spéciaux...
    //supprimer aussi les mots composés de 1 char => token.size >= 2

    line.split("""\W+""").map(_.toLowerCase).filter(token => regex.pattern.matcher(token).matches)
      .filterNot(token => stopwords.contains(token)).filter(token => token.size >= 2).toSeq
  }
}
