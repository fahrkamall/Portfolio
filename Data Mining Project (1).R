install.packages("Boruta")
install.packages("mlbench")
install.packages("caret")
install.packages("randomForest")
install.packages("ggplot2")
install.packages("FSelector")
install.packages("rsample")
install.packages("ROSE")
install.packages("RWeka")
install.packages("smotefamily")
install.packages("xgboost")
install.packages("adabag")


library(nnet)
library(Boruta)
library(mlbench)
library(randomForest)
library(ggplot2)
library(FSelector)
library(rsample)
library(ROSE)
library(caret)
library(e1071)
library(ROSE)
library(RWeka) 
library(smotefamily)
library(xgboost)
library(adabag)
library(rgl)

        
df <- read.csv("/Users/fahrkamal/Documents/R Projects/project_dataset.csv")
head(df)
#------TRAIN TEST SPLIT --------------------------------
set.seed(31)
df$X<- NULL
df$o_bullied <- factor(df$o_bullied)

split <- initial_split(df, prop = 0.66 , strata =  o_bullied)

train <- training(split)
test <- testing(split)
barplot(table(df$o_bullied))

table(train$o_bullied)
table(test$o_bullied)


#----------------------------------Initial Model and Model Results--------------------------------------------------------
#Decision Tree

J48.tree <- J48(o_bullied~., data = train)
pred_tree  <- predict(J48.tree, newdata = test, type = "class")

performance_measure_tree <- confusionMatrix(data = pred_tree , reference = test$o_bullied,positive = "1",mode = "everything")

performance_measure_tree

#Random Forest
rf <- randomForest(o_bullied~.,data= train) 
rf$mtry

pred_forest <- predict(rf, newdata = test, type = "class")
performance_measure_forest <- confusionMatrix(data =pred_forest, reference = test$o_bullied, positive = "1",mode = "everything")
performance_measure_forest

#Naive Bayes

model_nb <- naiveBayes(o_bullied~., data = train)
pred_nb <- predict(model_nb, newdata = test, type = "class")
performance_measure_nb <- confusionMatrix(data =pred_nb, reference = test$o_bullied, positive = "1",mode = "everything")
performance_measure_nb

#Logistic Regression
logitModel <- glm(o_bullied ~ ., data = train, family = "binomial") # it doesn't converge

#XGBoost
xgboost <- suppressWarnings(
  train(o_bullied~ .,
        data = train,  
        method = "xgbTree",
        objective = "binary:logistic")
)


pred_xgboost <- predict(xgboost, newdata = test, type = "raw")
performance_measure_xgboost <- confusionMatrix(data = pred_xgboost, reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_xgboost

#SVM

svm_classifier  <- svm(formula = o_bullied~ ., 
                 data = train, 
                 type = 'C-classification', 
                 kernel = 'linear') 



svm_preds <- predict(svm_classifier, newdata = test, type = "class")
performance_measure_svm <- confusionMatrix(data = svm_preds, reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_svm


#ADABOOST
model_adaboost <- boosting(o_bullied~., data=train)

adaboost_preds <- predict(model_adaboost,newdata = test, type = "class")
as.factor(adaboost_preds$class) 
performance_measure_adaboost <- confusionMatrix(data = as.factor(adaboost_preds$class) , reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_adaboost



#-----SAMPLING-----
#OverSampling
over <- ovun.sample(o_bullied~.,data = train , method = "over", N =4947)

over = over$data
head(over)
barplot(table(over$o_bullied))
table(over$o_bullied)

###Undersampling

# Count of the minority class
minority_count <- sum(train$o_bullied == 1)

# Undersampling Data 
under <- ovun.sample(o_bullied ~ ., data = train, method = "under", N = 2 * minority_count)
under_data <- under$data


# Boruta for feature selection on undersampled data
boruta_under <- Boruta(o_bullied ~., data = under_data, doTrace = 2, maxRuns = 100)
print(boruta_under)

plot(boruta_under, las = 2)

boruta_under_fixed <- TentativeRoughFix(boruta_under) # Optional
getNonRejectedFormula(boruta_under) # Optional
boruta_attributes_undersampling <- getConfirmedFormula(boruta_under)
boruta_attributes_undersampling

#Decision Tree

train_control <- trainControl(method = "repeatedcv", number = 10)

model_tree_under <- train(boruta_attributes_undersampling,
                          data = under_data,  
                          method = "J48",
                          trControl = train_control)

pred_tree_under <- predict(model_tree_under, newdata = test)
performance_measure_tree_under <- confusionMatrix(data = pred_tree_under, reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_tree_under
print(model_tree_under)


#Random Forest 
hyperparameters <- expand.grid(.mtry = c(50, 100, 150))
control <- trainControl(method = 'cv', number = 5)

rf_gridsearch_under <- train(boruta_attributes_undersampling, 
                             data = under_data,
                             method = 'rf',
                             metric = 'F1',
                             trControl = control,
                             verbose = FALSE,
                             tuneGrid = hyperparameters)

print(rf_gridsearch_under)
pred_rf_under <- predict(rf_gridsearch_under, newdata = test)
performance_measure_forest_under <- confusionMatrix(data = pred_rf_under, reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_forest_under

rf_gridsearch_under$bestTune

#Logistic Regression

logitModel_under <- glm(o_bullied ~ ., data = under_data, family = "binomial")
pred_logit_under <- predict(logitModel_under, newdata = test, type = "response")
pred_logit_under_class <- ifelse(pred_logit_under > 0.5, 1, 0)
performance_measure_logit_under <- confusionMatrix(data = factor(pred_logit_under_class, levels = levels(test$o_bullied)), reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_logit_under


#SVM
svm_classifier_under <- svm(formula = o_bullied ~ ., 
                            data = under_data, 
                            type = 'C-classification', 
                            kernel = 'linear')
svm_preds_under <- predict(svm_classifier_under, newdata = test, type = "class")
performance_measure_svm_under <- confusionMatrix(data = svm_preds_under, reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_svm_under

#ADABoost
model_adaboost_under <- boosting(o_bullied ~ ., data = under_data)
adaboost_preds_under <- predict(model_adaboost_under, newdata = test, type = "class")
performance_measure_adaboost_under <- confusionMatrix(data = as.factor(adaboost_preds_under$class), reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_adaboost_under

# Neural Network
set.seed(33) # for reproducibility
nn_model_under <- nnet(boruta_attributes_undersampling, 
                       data = under_data, 
                       size = 10, # number of units in the hidden layer
                       decay = 0.1, # weight decay
                       maxit = 200) # maximum iterations
nn_preds_under <- factor(nn_preds_under, levels = levels(test$o_bullied))

performance_measure_nn_under <- confusionMatrix(data = nn_preds_under, reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_nn_under

# PCA with Neural Network
nzv <- nearZeroVar(under_data)
under_data_nzv <- under_data[, -nzv]

pre_proc <- preProcess(under_data_nzv, method = "pca", thresh = 0.95)

under_data_pca <- predict(pre_proc, under_data_nzv)

nn_model_pca_under <- nnet(o_bullied ~ ., 
                           data = under_data_pca, 
                           size = 10, 
                           decay = 0.1, 
                           maxit = 200)

test_pca <- predict(pre_proc, test[, -nzv]) # Ensure you remove the zero-variance variables from the test set as well

nn_pca_preds_under <- factor(nn_pca_preds_under, levels = levels(test$o_bullied))

performance_measure_nn_pca_under <- confusionMatrix(data = nn_pca_preds_under, 
                                                    reference = test$o_bullied, 
                                                    positive = "1", 
                                                    mode = "everything")

performance_measure_nn_pca_under

# Stochastic XGBoost

# If the values are -1 and 0, we can simply add 1 to change them to 0 and 1
if(any(under_data$o_bullied == -1)) {
  under_data$o_bullied <- under_data$o_bullied + 1
}

# Now check the range to ensure it's only 0 and 1
range(under_data$o_bullied)



# Create the DMatrix again, now with the labels ensured to be in the correct range
xgb_data <- xgb.DMatrix(as.matrix(under_data[, -which(names(under_data) == "o_bullied")]), label = under_data$o_bullied)

xgb_data <- xgb.DMatrix(as.matrix(under_data[, -which(names(under_data) == "o_bullied")]), label = under_data$o_bullied)
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  subsample = 0.5, # 50% of the data samples to grow trees - prevents overfitting.
  eta = 0.1,
  max_depth = 6
)
xgb_model_under <- xgb.train(params, 
                             xgb_data, 
                             nrounds = 100, 
                             watchlist = list(val = xgb_data), 
                             print_every_n = 10)
xgb_preds_under <- predict(xgb_model_under, as.matrix(test[, -which(names(test) == "o_bullied")]))
xgb_preds_under_class <- ifelse(xgb_preds_under > 0.5, 1, 0)
performance_measure_xgb_under <- confusionMatrix(data = factor(xgb_preds_under_class, levels = levels(test$o_bullied)), reference = test$o_bullied, positive = "1", mode = "everything")
performance_measure_xgb_under

###### Table ######
extract_performance_metrics <- function(cm) {
  data.frame(
    TPR = cm$byClass["Sensitivity"],
    TNR = cm$byClass["Specificity"],
    Precision = cm$byClass["Pos Pred Value"],
    Recall = cm$byClass["Sensitivity"],
    F_Measure = cm$byClass["F1"],
    ROC = cm$overall["ROC"],
    MCC = cm$overall["Matthews Correlation Coefficient"],
    Kappa = cm$overall["Kappa"]
  )
}

performance_summary <- data.frame(
  Model = character(),
  TPR = numeric(),
  TNR = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  F_Measure = numeric(),
  ROC = numeric(),
  MCC = numeric(),
  Kappa = numeric(),
  stringsAsFactors = FALSE
)

# Add each model's performance metrics to the summary table
performance_summary[nrow(performance_summary) + 1, ] <- c("Decision Tree", unlist(extract_performance_metrics(performance_measure_tree_under)))
performance_summary[nrow(performance_summary) + 1, ] <- c("Random Forest", unlist(extract_performance_metrics(performance_measure_forest_under)))
performance_summary[nrow(performance_summary) + 1, ] <- c("Logistic Regression", unlist(extract_performance_metrics(performance_measure_logit_under)))
performance_summary[nrow(performance_summary) + 1, ] <- c("SVM", unlist(extract_performance_metrics(performance_measure_svm_under)))
performance_summary[nrow(performance_summary) + 1, ] <- c("AdaBoost", unlist(extract_performance_metrics(performance_measure_adaboost_under)))

# Display the summary table
print(performance_summary)

#MCC Function
mcc <- function(cm) {
  tp <- as.numeric(cm$table[2,2])
  tn <- as.numeric(cm$table[1,1])
  fp <- as.numeric(cm$table[1,2])
  fn <- as.numeric(cm$table[2,1])
  numerator <- (tp * tn) - (fp * fn)
  denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (denominator == 0) return(NA) # Prevent division by zero
  numerator / denominator
}


# Calculate MCC for each model
mcc_tree_under <- mcc(performance_measure_tree_under)
mcc_forest_under <- mcc(performance_measure_forest_under)
mcc_logit_under <- mcc(performance_measure_logit_under)
mcc_svm_under <- mcc(performance_measure_svm_under)
mcc_adaboost_under <- mcc(performance_measure_adaboost_under)

# Print the MCC values
print(paste("MCC for Decision Tree:", mcc_tree_under))
print(paste("MCC for Random Forest:", mcc_forest_under))
print(paste("MCC for Logistic Regression:", mcc_logit_under))
print(paste("MCC for SVM:", mcc_svm_under))
print(paste("MCC for AdaBoost:", mcc_adaboost_under))


print(performance_measure_tree_under)



#SMOTE
ncol(train)
column_types <- sapply(df, class)
print(column_types)

X = train[,-170]
y = train[,170]
smoted <- SMOTE(X,y)
table(smoted$data$class)






### ---------------OVERSAMPLING AND FEATURE SELECTION------------------------
#boruta
boruta <- Boruta(o_bullied ~., data = over, doTrace=2,maxRuns = 100)
print(boruta)

plot(boruta,las = 2)

bor_fixed <- TentativeRoughFix(boruta) #optional
getNonRejectedFormula(boruta) #optional
boruta_attributes_oversampling <- getConfirmedFormula(boruta)
boruta_attributes_oversampling

#decision tree

train_control <- trainControl(
  method = "repeatedcv",
  number = 10,

  
)

model_tree <- train(boruta_attributes_oversampling,
               data = over,  
               method = "J48",
               trControl = train_control,
              )

pred_tree  <- predict(model_tree, newdata = test)

performance_measure_tree <- confusionMatrix(data = pred_tree , reference = test$o_bullied,positive = "1",mode = "everything")

performance_measure_tree
print(model_tree)

#random forest
hyperparameters <- expand.grid(
  .mtry = c(50,100,150) )



control <- trainControl(method='cv', 
                        number=5,
                        )
                        

hyperparameters


rf_gridsearch <- train(boruta_attributes_oversampling, 
                       data = over,
                       method = 'rf',
                       metric = 'F1',
                       trControl = control,
                      verbose = F,
                      tuneGrid = hyperparameters,
      
                      )

print(rf_gridsearch)
pred_rf <-  predict(rf_gridsearch ,newdata = test)
performance_measure_forest <- confusionMatrix(data = pred_rf, reference = test$o_bullied,positive = "1",mode = "everything")
performance_measure_forest

rf_gridsearch$bestTune







#------FEATURE SELECTION----------
#Boruta
boruta <- Boruta(o_bullied ~., data = train, doTrace=2,maxRuns = 200)
print(boruta)

plot(boruta,las = 2)

bor_fixed <- TentativeRoughFix(boruta)
getNonRejectedFormula(boruta)
boruta_attributes <- getConfirmedFormula(boruta)
boruta_attributes


#rfe
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
results <- rfe(train[,1:(ncol(train)-1)], train[,ncol(train)], sizes=c(ncol(train)), rfeControl=control)
results

predictors(results)
rfe_importance <- varImp(results)

a<- c(1,2,3,)
a
