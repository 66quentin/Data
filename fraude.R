#Par Quentin Guardia, qguardia66@gmail.com
library(e1071) #naiveBayes et svm
library(class) #knn
library(MASS) #lda et qda
library(rpart) #rpart pour CART
library(randomForest) #randomForest
library(ROCR) #PRCurve
library(PRROC) #PRCurve
#en cas d'absence du package, écrire: install.packages("nom_du_paquet")

donnees <- read.csv(file = "creditcard.csv", head = TRUE) #télécharger creditcard.csv d'abord

#On enlève Time
donnees$Time <- NULL


#Contenu de la colonne Class
length(which(donnees$Class == 0))
length(which(donnees$Class == 1))


#Ces deux lignes sont à enlever si la machine le permet
indice <- sample(1:nrow(donnees),0.1*nrow(donnees))
donnees <- donnees[ indice,]

#Aperçu textuel:
summary(donnees)


#Aperçu graphique en exprimant les trois premières données en fonction de la données à prédire (cartevp)
pairs(donnees[,1:3], col=as.numeric(donnees$Class))


#Création du tableau bilan
stat<-matrix(list(), nrow=11, ncol=1)
colnames(stat) <- c("AUC")
rownames(stat) <- c("Bayésien naïf", "randomForest", "CART", "KNN" "Régression logistique", "LDA", "QDA", "SVM linéaire", "SVM polynomial", "SVM base radiale", "SVM sigmoide")


#Échantillonage
indice <- sample(1:nrow(donnees),0.7*nrow(donnees))
apprentissage <- donnees[ indice,]
test <- donnees[-indice,]
apprentissage_feature <- subset(apprentissage,select=-Class)
donnees_feature <- subset(donnees,select=-Class)
test_feature <- subset(test,select=-Class)

#Boucle affectant le résultat de chaque méthode au tableau stat
for(i in 1:11){
	if(i==1){ #Classifieur Bayésien naïf
		modele <- naiveBayes(as.factor(apprentissage$Class)~., data=apprentissage_feature)
		prediction <- predict(modele, test)
	}else if(i==2){#RF
		modele <- randomForest(apprentissage_feature, as.factor(apprentissage$Class))
		prediction <- predict(modele, test)
	}else if(i==3){#CART
		modele <- rpart(factor(donnees$Class) ~ ., data = donnees_feature)
		prediction <- predict(modele, donnees, type="class")
	}else if(i==4){#KNN
		iterations= 10
		erreur <- vector(mode="integer", length=iterations)
		for(k1 in 1:iterations){ #On choisit le meilleur k
			prediction <- knn(apprentissage_feature,test_feature,cl=apprentissage$Class,k1)
			erreur[k1] <- sum(test$Class != prediction)/length(donnees$Class)
		}
		k1=which.min(erreur)
		prediction <- knn(apprentissage_feature,test_feature,cl=apprentissage$Class,k1)
	}else if(i==5){#Régression logistique
		modele = glm(formula = apprentissage$Class ~.,family = binomial,data = apprentissage_feature)
		prediction = predict(modele,type = 'response',newdata = test_feature)
	}else if(i==6){#LDA
		modele <- lda(donnees$Class ~ ., data = donnees_feature)
		prediction <- predict(modele, donnees_feature)
		pr <- pr.curve( prediction$class, donnees$Class, curve = TRUE );
	}else if(i==7){#QDA
		modele <- qda(donnees$Class ~ ., data = donnees_feature)
		prediction <- predict(modele, donnees_feature)
		pr <- pr.curve( prediction$class, donnees$Class, curve = TRUE );
	}else if(i==8){#SVM
		modele <- svm(donnees$Class ~ ., data=donnees, kernel="linear")
	}else if(i==9){
		modele <- svm(donnees$Class ~ ., data=donnees, kernel="polynomial")
	}else if(i==10){
		modele <- svm(donnees$Class ~ ., data=donnees, kernel="radial")
	}else if(i==11){
		modele <- svm(donnees$Class ~ ., data=donnees, kernel="sigmoid")
	}
	if(i > 7){
		prediction <- predict(modele,donnees)
		pr <- pr.curve( prediction, donnees$Class, curve = TRUE );
	}
	if(i < 6){
		pr <- pr.curve( prediction, test$Class, curve = TRUE )
	}
	stat[i,1] <-pr$auc.integral
}
stat
