library(tidyverse)
if (!require(ModelMetrics)){
  install.packages("ModelMetrics")
  library(ModelMetrics)
}
library(caret)
library(dplyr)
####################
#GENERATE THE DATA#
##################
n_samples <- 1000
#Create a matrix to store our generated data
data <- data.frame(matrix(nrow = n_samples, ncol = 2))
data <- data %>% rename( "obese" = X1 ,'weight' = X2 )

#Draw random samples
set.seed(42)
data['weight'] <- rnorm(n_samples, mean = 177/2.2, sd = 29/2.2 )

#Generete binary classification of obesity 

data["rank"] <- rank(data$weight)/1000
data["random_numbers"] <- runif(1000)
data["obese"] <- ifelse(data$rank < data$random_numbers, yes = 0, no = 1 )

#####################
#PARTITION THE DATA#
###################
#Choose random indices form data 
index <- createDataPartition(data$obese, p = 0.8, list = FALSE, times = 1)  
#Choose the data with these indices from our matrix to create training and testing data
train <- data[index,]
test <- data[-index, ]
test["rank"] <- NULL
test["random_numbers"] <- NULL
train["rank"] <- NULL
train["random_numbers"] <- NULL

#convert to factors and valid r variable 
train$obese <- factor(train$obese, levels = c("0", "1"), labels = c("not_obese", "obese"))
test$obese <- factor(test$obese, levels = c("0", "1"), labels = c("not_obese", "obese"))

#############################################
#SPECIFY THE TRAINING METHODS AND N OF FOLDS#
############################################
#n = floor(runif(1)*100)
n = 10
ctrlspecs <- trainControl(method = "cv", number = n, savePredictions = "all" , classProbs = TRUE   )
model1 <- train(obese ~ weight, data = train, method = "glm", family = binomial, trControl = ctrlspecs)
summary(model1)
varImp(model1)

#####################
#PREDICT THE OUTCOME#
####################
predictions <- predict(model1, newdata = test)
confusionMatrix(predictions, test$obese)
