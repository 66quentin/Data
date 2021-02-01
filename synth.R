#fichier à éxecuter dans le dossiet des datasets
library(caret) #confusionMatrix
library(e1071) #naiveBayes et svm
library(class) #knn
library(MASS) #lda et qda
library(rpart) #rpart pour CART
library(randomForest) #randomForest
#en cas d'absence du package, écrire: install.packages("nom_du_paquet")

donnees <- read.delim("spiral.txt") #Valable pour les données synthétiques: flame, spiral, aggregation
attach(donnees)


#Aperçu textuel:
summary(donnees)
apply(donnees, 2, sd) #ecart-type


#Aperçu graphique:
boxplot(donnees[,1],main="Colomne 1")
boxplot(donnees[,2],main="Colomne 2")
boxplot(donnees[,3],main="Colomne 3")
plot(donnees[,1], donnees[,2], col=donnees[,3]) #Intéressant, il faut prédire la 3ème colonne


#Création du tableau bilan
stat<-matrix(list(), nrow=11, ncol=2)
colnames(stat) <- c("accuracy","F-measure")
rownames(stat) <- c("Bayésien naïf", "randomForest", "CART", "KNN", "Régression logistique", "LDA", "QDA", "SVM linéaire", "SVM polynomial", "SVM base radiale", "SVM sigmoide")


#On choisit préalablement le meilleur k pour knn
iterations= 50
indice <- sample(1:nrow(donnees),0.8*nrow(donnees))
apprentissage <- donnees[ indice,]
test <- donnees[-indice,]
erreur <- vector(mode="integer", length=iterations)
for(k1 in 1:iterations){ #On choisit le meilleur k
	prediction <- knn(apprentissage[,-3],test[,-3],cl=apprentissage[,3],k1)
	erreur[k1] <- sum(test[,3] != prediction)/length(donnees[,3])
}
k1=which.min(erreur)


#On fait 100 itérations pour chaque méthode de prédiction pour calculer accuracy et F-measure moyen
n=407
accuracy <- vector(length = n)
fmeasure <- vector(length = n)

print("On attend quelques secondes: les résultats chargent")
for (k in 1:407) {

	#Échantillonage aléatoire
	set.seed(3*k)
	indice <- sample(1:nrow(donnees),0.7*nrow(donnees))
	apprentissage <- donnees[ indice,]
	test <- donnees[-indice,]
	
	if(k<201){
		if(k <=100){ #Classifieur Bayésien naïf
			modele <- naiveBayes(as.factor(apprentissage[,3])~., data=apprentissage[,-3])
		}else if(k>100 ){#randomForest
			modele <- randomForest(apprentissage[,1:2], as.factor(apprentissage[,3]))
		}
		prediction <- predict(modele, test)
		resultats <- confusionMatrix(table(prediction,test[,3]))
	}else if(k>200 & k < 301){#CART
			modele <- rpart(as.factor(donnees[,3]) ~ donnees[,1]+donnees[,2], data = donnees)
			prediction <- predict(modele, donnees, type="class")
			resultats <- confusionMatrix(table(donnees[,3],prediction))
	}else if(k>300 & k < 401){ #knn
		prediction <- knn(apprentissage[,-3],test[,-3],cl=apprentissage[,3],k1) 
		resultats <- confusionMatrix(table(prediction,test[,3]))
	}else if(k==401){ #Régression logistique
		modele <- glm(donnees[,3]~.,data=donnees[,-3])
		prediction <- round(predict(modele, as.data.frame(donnees[,3]))) #Threshold temporaire
		prediction <- factor(as.factor(prediction), levels=levels(as.factor(donnees[,3])))
		resultats <- confusionMatrix(table(prediction,donnees[,3]))
	}else if(k>401 & k < 404){
		if(k==402){#LDA
			modele <- lda(donnees[,3] ~ ., data = donnees[-3])
		}else if(k==403){#QDA
			modele <- qda(donnees[,3] ~ ., data = donnees[-3])
		}
		prediction <- predict(modele, donnees[-3])
		resultats <- confusionMatrix(table(donnees[,3],prediction$class))
	}else if(k>403){
		if(k==404){ #SVM linéaire
			modele <- svm(donnees[,-3],as.factor(as.character(donnees[,3])),kernel="linear")
		}else if(k==405){ #SVM polynomial
			modele <- svm(donnees[,-3],as.factor(as.character(donnees[,3])),kernel="polynomial")
		}else if(k==406){ #SVM base radiale
			modele <- svm(donnees[,-3],as.factor(as.character(donnees[,3])),kernel="radial")
		}else if(k==407){ #SVM sigmoide
			modele <- svm(donnees[,-3],as.factor(as.character(donnees[,3])),kernel="sigmoid")
		}
		prediction <- predict(modele, donnees[,-3])
		resultats <- confusionMatrix(table(prediction,donnees[,3]))
	}
	
	#Calculs: accuracy et F-Measure
	cm <- as.matrix(resultats)
	n = sum(cm) 
	nc = nrow(cm) 
	rowsums = apply(cm, 1, sum) 
	colsums = apply(cm, 2, sum)
	diag = diag(cm)  
	precision = diag / colsums 
	recall = diag / rowsums 
	fmeasure_multiclasses = 2 * precision * recall / (precision + recall) 
	if(k<401){
		accuracy[k] <- resultats$overall['Accuracy'] 
		fmeasure[k] <- sum(fmeasure_multiclasses*colsums)/sum(colsums)
	}else{
		stat[k-396,1] <- resultats$overall['Accuracy']
		stat[k-396,2] <- sum(fmeasure_multiclasses*colsums)/sum(colsums)
	}
}


#Remplissage du reste du tableau bilan
for(i in 0:3){
	inf <- i*100+1
	sup <- i*100+100
	stat[i+1,1]<-mean(accuracy[inf:sup])
	stat[i+1,2]<-mean(fmeasure[inf:sup])
}
stat

