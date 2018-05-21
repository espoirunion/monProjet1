package mlModelsCreation
import java.io.File
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint

object CreationLogisticRegressionModel {
  //fonction qui créé des mots (tokens) à partir d'un texte (line) puis filtre les mots
  def tokenize(line: String): Seq[String] = {
    val regex = """[^0-9]*""".r       //expression à utiliser pour filter les mots comportant des chiffres
    val stopwords = Set("the", "a", "an", "of", "or", "in", "for", "by", "on", "but", "is", "not", "with", "as", "was", "if", "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to")
    //stopwords = un ensemble de mots très communs en anglais, à ignorer
    //line.split("""\W+""") permet d'eclater la ligne en mots en utilisant les blancs, les signes de ponctuation, les caractères spéciaux...
    //la fonction tokenize supprime aussi les mots composés de 1 char => token.size >= 2

    line.split("""\W+""").map(_.toLowerCase).filter(token => regex.pattern.matcher(token).matches)
      .filterNot(token => stopwords.contains(token)).filter(token => token.size >= 2).toSeq  //produit = sequence de mots
  }

  def main(args: Array[String]): Unit = {
    // spark config
    val conf: SparkConf = new SparkConf()
      .setMaster("local")
      .setAppName("Logistic RegressionModelCreation")
    // Spark Context creation
    val sc: SparkContext = new SparkContext(conf)

    //chargement des données train : les fichiers des textes (reviews) du train sont dans le repertoire resources/train
    val resourceTrain = getClass.getClassLoader.getResource("train")
    var filePathTrain = new File(resourceTrain.toURI).getPath
    filePathTrain = filePathTrain + "\\*"
    val rdd = sc.wholeTextFiles(filePathTrain)
    val text = rdd.map { case (file, text) => text }

    //zone test
    println("filepath = " + filePathTrain)
    println ("le nombre de documents chargés dans le rdd = " + text.count)   //25000  fichiers texte entre reviews positifs et negatifs
    val reviewsGroups = rdd.map { case (file, text) => file.split("/").takeRight(2).head } //le nom du repertoire du fichier est pos ou neg => le sentiment

    //tokenization de tous les textes du train pour une bonne analyse
    val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_. toLowerCase))
    println("Nombre de mots total avant filtrage = " + whiteSpaceSplit.distinct.count)         //nombre initial de mots dans le corpus : 252192
    val moreSplitedWords = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))  //spliter sur les char de ponctuation, les char spéciaux..., garder que les mots
    //println(moreSplitedWords.distinct.count)      //nombre de mots réduit à 74630
    val regex = """[^0-9]*""".r   //expression regulière qui permet de filtrer les mots comportant des chiffres (donc peu significatifs)
    val filterNumbers = moreSplitedWords.filter(token => regex.pattern.matcher(token).matches)
    //println(filterNumbers.distinct.count)     //nombre de mots du corpus réduit à 73346
    val wordCounts = filterNumbers.map(t => (t, 1)).reduceByKey (_ + _)    //compter le nombre d'occurences de chaque mot dans le corpus, pour voir les mots rares
    //en anglais, voici un exemple de mots très communs qu'on appelle stop words à éliminer du corpus
    val stopwords = Set( "the","a","an","of","or","in","for","by","on","but", "is", "not", "with", "as", "was", "if","they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to" )
    val wordCountsFilteredStopwords = wordCounts.filter { case (k, v) => !stopwords.contains(k) }
    //chercher les mots rares (1 occurance) qui n'aurant pas du poid dans la construction de notre modele de machine learning
    val rareWords = wordCounts.filter{ case (k, v) => v < 2 }.map { case (k, v) => k }.collect.toSet //composer un set avec ces mots rares
    val wordCountsFilteredRare = wordCountsFilteredStopwords.filter { case (k, v) => !rareWords.contains(k) }
    //il faut aussi filter les mots de longeur 1
    val wordCountsFiltered = wordCountsFilteredRare.filter { case (k, v) => k.size >= 2 }
    println("Nombre de mots total après filtrage = " + wordCountsFiltered.distinct.count)  //nombre de mots du corpus réduit à 50134

    //on définit une fonction générale qui combine toutes les transformations précédentes
    def tokenize(text: String): Seq[String] = {
      text.split("""\W+""").map(_.toLowerCase).filter(word => regex.pattern.matcher(word).matches)
        .filterNot(word => stopwords.contains(word)).filterNot(word => rareWords.contains(word))
        .filter(token => token.size >= 2).toSeq
    }
    val tokens = text.map(doc => tokenize(doc))   //chaque élément (texte) dans ce rdd est transformé en une sequence de tokens (mots) filtré
    println("exemple de texte tokenisé : " + tokens.first.take(20))
    //WrappedArray(remember, watching, movie, several, times, very, young, kid, there, were, parts, many, fact, did, understand, think, seen, once, adult, then)

    /**** nous utilisons tf/idf pour la représentation des features des textes du corpus (train)****/
    /* idf permet de mieux pendérer l'importance d'un mots dans un texte en fonction du nombre de ses occurences dans le corpus */
    /* tf.transform permet de transformer la sequence des mots d'un texte en un vecteur (Sparce vector) = features , à travers une fnction de hashage*/
    val dim = math.pow(2, 18).toInt  //taille du vecteur des features (262144), valeur choisie >> nombre de mots dans le corpus trains (50134)
    val hashingTF = new HashingTF(dim)
    //créer un vecteur (features) pour chaque texte tokenisé (du train)
    val tf = hashingTF.transform(tokens)  // dans ce vecteur, indexe = résultat hashage d'un mot, la valeur = nombre d'occurence du mot dans le texte
    tf.cache                         //persister ce rdd car il sera utilisé par la suite pour éviter de le recalculer
    val idf = new IDF().fit(tf)      //calculer pour chaque mot aussi le nombre de textes le comportant (dans le corpus)
    val tfidf = idf.transform(tf)    //l'occumence de chaque mot * log (N/d) , N = nombre de textes du corpus, d = nombre de textes comportant le mot

    //création et entrainement d'un modèle de regression logistique (classification binaire)
    val reviewsGroupsMap = Map("pos" -> 1, "neg" -> 0)   //deux classes : sentiement positif ou negatif
    val zipped = reviewsGroups.zip(tfidf)      //rdd [(k,v)] : k = 0 ou 1 , v = vecteur features du texte
    val train = zipped.map { case (label, vector) => LabeledPoint(reviewsGroupsMap(label), vector) } //LabeledPoint combine un label à un vecteur features
    train.cache        //persister le rdd des données train car il est grand et sera utilisé par la suite
    val lrLearner = new LogisticRegressionWithSGD()  // création d'un modèle de regression logistic (binaire) qui utilise l'optimiseur LBFGS.
    val model = lrLearner.run(train)    // execution l'algorithm d'apprentissage sur les données (train).
    //sauvegrade du modèle dans un dossier en local pour l'utiliser ultérieurement (dans le repertoire src/main/resources du projet)
    model.save(sc, "C:/Users/Leila/IdeaProjects/project3_final/src/main/resources/MyModelLR1")

    //evaluation du model sur les données de test
    val resourceTest = getClass.getClassLoader.getResource("test")
    var filePathTest = new File(resourceTest.toURI).getPath
    filePathTest = filePathTest + "\\*"
    //chragement des données test dans un rdd
    val testRDD = sc.wholeTextFiles(filePathTest)
    val testLabels = testRDD.map { case (file, text) =>
      val topic = file.split("/").takeRight(2).head
      reviewsGroupsMap(topic)   }                  //avoir le RDD des sentiments des textes du test (0 ou 1)
    val testTf = testRDD.map { case (file, text) =>  hashingTF.transform(tokenize(text)) }  //RDD des features des textes du test
    val testTfIdf = idf.transform(testTf)           //appliquer la penderation idf globale
    val zippedTest = testLabels.zip(testTfIdf)      //RDD[(k,v)] : k =0 ou 1 , v= features du texte
    val test = zippedTest.map { case (topic, vector) =>  LabeledPoint(topic, vector) }   //RDD de LabelPoint
    val predictionAndLabel = test.map(p => (model.predict(p.features),  p.label))   //RDD de couple (label predit, label réel)
    val accuracy = 1.0 * predictionAndLabel.filter (x => x._1 == x._2).count() / test.count()  //le nombre d'égalité des labels /nombre de textes (du test)
    println("accuracy = "+ accuracy)  //précision du modèle = 0.88748
  }
}