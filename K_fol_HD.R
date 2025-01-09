library(dplyr)
library(caret)
library(tidyverse)
library(pROC)
set.seed(42)

###############
#READ#THE#DATA#
##############
setwd("D:\\Documents\\Regresie\\Heart Disease\\Data")

data <- read.csv("processed.cleveland.data.txt")

colnames(data) <- c(
  "age",
  "sex",# 0 = female, 1 = male
  "cp", # chest pain
  # 1 = typical angina,
  # 2 = atypical angina,
  # 3 = non-anginal pain,
  # 4 = asymptomatic
  "trestbps", # resting blood pressure (in mm Hg)
  "chol", # serum cholestoral in mg/dl
  "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
  "restecg", # resting electrocardiographic results
  # 1 = normal
  # 2 = having ST-T wave abnormality
  # 3 = showing probable or definite left ventricular hypertrophy
  "thalach", # maximum heart rate achieved
  "exang",   # exercise induced angina, 1 = yes, 0 = no
  "oldpeak", # ST depression induced by exercise relative to rest
  "slope", # the slope of the peak exercise ST segment
  # 1 = upsloping
  # 2 = flat
  # 3 = downsloping
  "ca", # number of major vessels (0-3) colored by fluoroscopy
  "thal", # this is short of thalium heart scan
  # 3 = normal (no cold spots)
  # 6 = fixed defect (cold spots during rest and exercise)
  # 7 = reversible defect (when cold spots only appear during exercise)
  "hd" # (the predicted attribute) - diagnosis of heart disease
  # 0 if less than or equal to 50% diameter narrowing
  # 1 if greater than 50% diameter narrowing
)

###########################
#CREATE#TRAIN#TEST#DATASET#
########################### 

indices <- createDataPartition(data$hd, list = FALSE, p = 0.8, times = 1)
#
train <- data[indices, ]
test <- data[-indices, ]

#CONVERT THE VARIABLES TO VALID R NAMES AND FACTORISE

train$hd <- ifelse(train$hd == 0, yes = "healthy", no = "sick" )
test$hd <- ifelse(test$hd == 0, yes = "healthy", no = "sick" )
train$sex <- ifelse(train$sex == 1, yes = "M", no = "F")
test$sex <- ifelse(test$sex == 1, yes = "M", no = "F")


columns <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "hd")
train[columns] <- lapply(train[columns], as.factor)
test[columns] <- lapply(test[columns], as.factor)



#SPECIFY THE TYPE OF RESAMPLING

n = 10
ctrlspecs <- trainControl(method = "cv", number = n, savePredictions = "all", classProbs = TRUE)

#################
#TRAIN#THE#MODEL#
#################

model1 <- train(hd ~ ., method = "glm", family = "binomial", trControl = ctrlspecs, data = train)
summary(model1)
varImp(model1)

########################
#INTRODUCE#THE#NEW#DATA#
########################

predictions <- predict(model1, newdata = test)
confusionMatrix(predictions, test$hd)

##########################
#REPEAT#FOR#DECISION#TREE#
##########################

model2 <- train(hd ~ ., method = "rpart", trControl = ctrlspecs, data = train)
predictions2 <- predict(model2, newdata = test)
confusionMatrix(predictions2, test$hd)

###############
#PLOT#THE#ROCs#
##############

#caret doesnt store fitted values the same way glm() does, so we need to get them
logistic_probs <- predict(model1, newdata = test, type="prob")[,"sick"]
tree_probs <- predict(model2, newdata = test, type="prob")[,"sick"]

#get the roc information
par(pty = "s")
roc_logistic <-roc(test$hd, logistic_probs, plot = FALSE, legacy.axes = FALSE)
roc_tree <- roc(test$hd, tree_probs, plot = FALSE, legacy.axes = FALSE)

plot.roc(roc_logistic, col = "blue", main = "ROC CURVES", ylab = "True Positive Rate", legacy.axes = TRUE, xlab = "False Positive Rate")
plot.roc(roc_tree, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"),
       col = c("blue", "red"), lwd = 2)


cat("AUC (Logistic Regression):", auc(roc_logistic), "\n")
cat("AUC (Decision Tree):", auc(roc_tree), "\n")

