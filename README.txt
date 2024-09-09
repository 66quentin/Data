Programmes réalisés par Quentin Guardia, quentin.guardia@protonmail.com

Méthodes de prédiction utilisées:
    • Classifieur bayésien naïf
    • CART
    • randomForest
    • LDA
    • QDA
    • SVM linéaire et non-linéaires (linéaire, polynomial, base radiale et sigmoïde) 
    • régression logistique
    • KNN 
    
Outils permettant de mesurer l'efficacité des méthodes:
    • F-Measure
    • Accuracy 
    • AUC sous la Precision-Recall Curve


synth.R: (données synthétiques)

	Prédit les valeurs de la troisième colonne dans les fichiers Aggregation.txt, flame.txt et spiral.txt

visa.R: (données réelles)

	Prédit la variable cartevpr du jeu de données VisaPremier.txt, qui indique si la personne détient un carte Visa Premier.

fraude.R: (données réelles)

	Prédit la variable class du jeu de données creditcard.csv, qui indique s'il y a fraude lors de la transaction. Le fichier étant trop lourd pour github, on le retrouve sur: https://www.kaggle.com/mlg-ulb/creditcardfraud
