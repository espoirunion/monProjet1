package MLmodelsPrediction

import java.io.File
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD,LogisticRegressionModel}

object LogisticRegressionPrediction {

    //function for créer des mots à partir d'un texte (line) puis de filtrer ces mots
    def tokenize(line: String): Seq[String] = {
      val regex = """[^0-9]*""".r       //expression à utiliser pour filter les mots comportant des chiffres
      val stopwords = Set("the", "a", "an", "of", "or", "in", "for", "by", "on", "but", "is", "not", "with", "as", "was", "if", "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to")
      //stopwords = un ensemble de mots très commun en anglais, à ignorer
      //line.split("""\W+""") permet d'eclater la ligne en mots en utilisant les blancs, les signes de ponctuation, les caractères spéciaux...
      //supprimer aussi les mots composés de 1 char => token.size >= 2

      line.split("""\W+""").map(_.toLowerCase).filter(token => regex.pattern.matcher(token).matches)
        .filterNot(token => stopwords.contains(token)).filter(token => token.size >= 2).toSeq
    }

    def main(args: Array[String]): Unit = {
      // 1.spark config
      val conf: SparkConf = new SparkConf()
           .setMaster("local")
           .setAppName("LogisticRegressionModelTest")
      // 2. SparkContext creation
      val sc: SparkContext = new SparkContext(conf)

      //data filePath : les fichiers des reviews à classifier sont dans ressources/unsup1
      val resourceReviews = getClass.getClassLoader.getResource("unsup1")
      var filePathReviews = new File(resourceReviews.toURI).getPath
      filePathReviews = filePathReviews + "\\*"
      //mon modèle de regression logistic est dans ressources/myModelLogisticRegression)
      //mon modèle de regression logistic est dans ressources/myModelNaiveBayes)
      val resourceModel = getClass.getClassLoader.getResource("myModelLogisticRegression")
      val filePathModel = new File(resourceModel.toURI).getPath

      //chargement (dans un RDD)des textes à classer à partir d'un repertoire (unsup1) de fichiers texte
      val unsupRDD = sc.wholeTextFiles(filePathReviews)
      println("the number of files is : "+ unsupRDD.count)
      val unsupFiles = unsupRDD.map { case (file, text) => file.split("/").takeRight(1).head }
      val unsupTexts = unsupRDD.map { case (file, text) => text }
      println("list of file names :"+ unsupFiles.collect.mkString("\n")) //liste of noms de fichiers

      //tokeniser (eclater en mots et filtrer) chaque texte du RDD unsupTexts
      val unsupTokens = unsupTexts.map(doc => tokenize(doc))
      println("example of text tokenized : " + unsupTokens.first.take(20).mkString(",")) //example of the first text tokenized

      //utilisation de la fonction hashingTF pour convertir la séquence de tokens d'un texte en un sparce vector (features)
      val dim = math.pow(2, 18).toInt //vector de taille 262144 (ce choix est basé sur l'analyse de nos données de train, il y a 50.000 mots)
      val hashingTF = new HashingTF(dim)
      val unsupTf = hashingTF.transform(unsupTokens)  //un rdd[SparseVector] dont chaque vecteur represente les features du texte

      //charger le modele de regression logistic pour l'utiliser dans la classification (prediction du sentiment)
      val model = LogisticRegressionModel.load(sc, filePathModel)

      //classer le review en sentiment positif ou negatif (0: neg, 1: pos)
      val predictLabels = unsupTf.map(p => model.predict(p))
      println(predictLabels.collect.mkString("\n"))

      //mettre les labels (0/1) et les noms des fichiers texte (reviews) dans un même rdd pour imprimer le résultat
      val totalFilePredict = unsupFiles.zip(predictLabels)  //totalFilePredict[(nomFichier, review)] = rdd[(k,v)]
      println(totalFilePredict.collect.mkString("\n"))
    }
  }
