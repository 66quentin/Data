#Par Quentin Guardia, qguardia66@gmail.com
library(caret) #confusionMatrix
library(e1071) #naiveBayes et svm
library(class) #knn
library(MASS) #lda et qda
library(rpart) #rpart pour CART
library(randomForest) #randomForest
#en cas d'absence du package, écrire: install.packages("nom_du_paquet")

donnees <- read.delim("datasets/VisaPremier.txt") 

#On supprime les données non ou peu prédictives
donnees <- donnees[, !(names(donnees) %in% c("matricul","cartevp","sexe", "departem", "nbimpaye", "mteparte", "nbbon", "mtbon", "nbeparte"))]
attach(donnees)


#Numérisation des données
levels(donnees$sitfamil) <- c(1:length(levels(as.factor(donnees$sitfamil))))
levels(donnees$csp) <- c(1:length(levels(as.factor(donnees$csp))))
levels(donnees$codeqlt) <- c(1:length(levels(as.factor(donnees$codeqlt))))

#En cas d'erreur de NA par la suite, relancer le programme en décommentant la ligne dessous
#donnees <- donnees[, !(names(donnees) %in% c("sitfamil","csp","codeqlt"))]

#Aperçu textuel:
summary(donnees)


#Contenu de la colonne cartevpr
length(which(donnees$cartevpr == 0))
length(which(donnees$cartevpr == 1))


#Aperçu graphique en exprimant les trois premières données en fonction de la données à prédire (cartevpr)
pairs(donnees[,1:3], col=as.numeric(cartevpr))


#RandomForest ne supporte que 53 levels. On enlève colonnes avec plus de levels
i=1
while(i < ncol(donnees)){
	if(length(levels(as.factor(donnees[,i])))>52){
		donnees<-donnees[,-c(i)]
	}else{
		i=i+1
	}
}

#Tableau récapitulatif
stat<-matrix(list(), nrow=11, ncol=2)
colnames(stat) <- c("accuracy","F-measure")
rownames(stat) <- c("Bayésien naïf", "randomForest", "CART", "KNN", "Régression logistique", "LDA", "QDA", "SVM linéaire", "SVM polynomial", "SVM base radiale", "SVM sigmoide")

#On fait 10 itérations pour chaque méthode de prédiction pour calculer accuracy et F-measure moyen
n=47
accuracy <- vector(length = n)
fmeasure <- vector(length = n)


#On choisit préalablement le meilleur k pour knn
iterations= 50
set.seed(3)
indice <- sample(1:nrow(donnees),0.7*nrow(donnees))
apprentissage <- donnees[ indice,]
test <- donnees[-indice,]
apprentissage_feature <- subset(apprentissage,select=-cartevpr)
donnees_feature <- subset(donnees,select=-cartevpr)
test_feature <- subset(test,select=-cartevpr)
erreur <- vector(mode="integer", length=iterations)
for(k1 in 1:iterations){ #On choisit le meilleur k
	prediction <- knn(apprentissage_feature,test_feature,cl=apprentissage$cartevpr,k1)
	erreur[k1] <- sum(test$cartevpr != prediction)/length(donnees$cartevpr)
}
k1=which.min(erreur)

#La boucle
for(i in 1:47){
	set.seed(3*i)
	indice <- sample(1:nrow(donnees),0.7*nrow(donnees))
	apprentissage <- donnees[ indice,]
	test <- donnees[-indice,]
	apprentissage_feature <- subset(apprentissage,select=-cartevpr)
	donnees_feature <- subset(donnees,select=-cartevpr)
	test_feature <- subset(test,select=-cartevpr)
	if(i < 21){#Bayes et RF
		if(i<11){
			modele <- naiveBayes(as.factor(apprentissage$cartevpr)~., data=apprentissage_feature)
		}else{
			modele <- randomForest(apprentissage_feature, as.factor(apprentissage$cartevpr))
		}
		prediction <- predict(modele, test)
		resultats <- confusionMatrix(table(prediction,test$cartevpr), positive="1")
	}else if(i > 20 & i < 31){ #CART
		modele <- rpart(factor(donnees$cartevpr) ~ ., data = donnees_feature)
		prediction <- predict(modele, donnees, type="class")
		resultats <- confusionMatrix(table(donnees$cartevpr,prediction), positive="1")
	}else if(i > 30 & i < 41){ #KNN
		prediction <- knn(apprentissage_feature,test_feature,cl=apprentissage$cartevpr,k1)
		resultats <- confusionMatrix(table(prediction,test$cartevpr), positive="1")
	}else if(i==41){ #Regression logitique
		
		#Sélection du meilleur seuil
		fiabilite <- vector(mode="integer", length=9)
		j <- 1
		for (seuil in seq(0.30, 0.70, by=0.05)) {
			modele <- glm(donnees$cartevpr~.,data=donnees[,!colnames(donnees) %in% c("sitfamil","csp","codeqlt")])
			prediction <- ifelse(predict(modele, as.data.frame(donnees$cartevpr)) >= seuil, 1, 0)
			resultats <- confusionMatrix(table(prediction,donnees$cartevpr))		
			fiabilite[j] <- resultats$overall['Accuracy'] + resultats$byClass['F1']
			j <- j+1
		}
		seuil1=which.max(fiabilite)
		seuil2=0.30+0.05*(seuil1-1)
		modele <- glm(donnees$cartevpr~.,data=donnees[,!colnames(donnees) %in% c("sitfamil","csp","codeqlt")])
		prediction <- ifelse(predict(modele, as.data.frame(donnees$cartevpr)) >= seuil2, 1, 0)
		resultats <- confusionMatrix(table(prediction,donnees$cartevpr))
	}else if(i>41 & i < 44){
		if(i==42){#LDA
			modele <- lda(donnees$cartevpr ~ ., data = donnees_feature[,-6])
		}else if(i==43){#QDA
			modele <- qda(donnees$cartevpr ~ ., data = donnees_feature[,-4])
		}
		prediction <- predict(modele, donnees_feature)
		resultats <- confusionMatrix(table(donnees$cartevpr,prediction$class), positive="1")
	}else if(i>43){
		if(i==44){ #SVM linéaire
			modele <- svm(donnees$cartevpr ~ ., data=donnees, kernel="linear")
		}else if(i==45){ #SVM polynomial
			modele <- svm(donnees$cartevpr ~ ., data=donnees, kernel="polynomial")
		}else if(i==46){ #SVM base radiale
			modele <- svm(donnees$cartevpr ~ ., data=donnees, kernel="radial")
		}else if(i==47){ #SVM sigmoide
			modele <- svm(donnees$cartevpr ~ ., data=donnees, kernel="sigmoid")
		}
		#Selection du seuil
		fiabilite <- vector(mode="integer", length=9)
		j <- 1
		for (seuil in seq(0.30, 0.70, by=0.05)) {
			prediction = ifelse(predict(modele,donnees) >=seuil, 1, 0)
			resultats <- confusionMatrix(table(prediction,donnees$cartevpr))		
			fiabilite[j] <- resultats$overall['Accuracy'] + resultats$byClass['F1']
			j <- j+1
		}
		seuil1=which.max(fiabilite)
		seuil2=0.30+0.05*(seuil1-1)
		prediction = ifelse(predict(modele,donnees) >=seuil2, 1, 0)
		resultats <- confusionMatrix(table(prediction,donnees$cartevpr), positive="1")
	}
	
	#Accuracy et F-Measure
	if(i<41){
		accuracy[i] <- resultats$overall['Accuracy'] 
		fmeasure[i] <- resultats$byClass['F1']
	}else{
		stat[i-36,1] <- resultats$overall['Accuracy']
		stat[i-36,2] <- resultats$byClass['F1']
	}
}

#Remplissage du reste du tableau bilan
for(i in 0:3){
	inf <- i*10+1
	sup <- i*10+10
	stat[i+1,1]<-mean(accuracy[inf:sup])
	stat[i+1,2]<-mean(fmeasure[inf:sup])
}
stat
