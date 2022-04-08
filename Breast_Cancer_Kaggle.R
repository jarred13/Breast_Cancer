#Breast Cancer paper for Kaggle using the Wisconin data set

#installing required packages if not previouly installed
if(!require(matrixStats)) install.packages("matrixStats")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(dslabs)) install.packages("dslabs")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(knitr)) install.packages("knitr")

#setting digits to 3 places
options(digits = 3)

#downloading the libraries
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
library(dplyr)
library(tidyr)
library(ggthemes)
library(knitr)

#downloading the data from the dslabs library
data(brca)

#the data is in two list
#looking at the dimenions of both list
dim(brca$x)
dim(brca$y)

#taking a look at the data
head(brca$x)
head(brca$y)

#changing brca$x to just x
x <- brca$x

#changing brca$y to just y
y <- brca$y

#taking a look at the variables in x
colnames(x)

#structure of x
str(x)

#summary of x
summary(x)

#taking a look to see if there are any NAs or blank cells
colSums(is.na(x))
sum(x == "")

#After looking at the summary of x we can see that the features do not
#have the same ranges. In fact some are quite larger than others.
#So to avoid any features influence the algoirthms in an adverse way,
#we are now going to scale the matrix
x_centered <- sweep(x, 2, colMeans(x))
x_scaled <- sweep(x_centered, 2, colSds(x), FUN = "/")

#checking the first column's standard divation, should be close to 1 since we
#scaled the matrix
sd(x_scaled[,1])

#checking the first column's median valuse, should be close to 0 after scaling
median(x_scaled[,1])

#is our outcomes balanced?
#our outcomes are not balance around 2/3 are benign (not cancerous) 
mean(y == "M")
mean(y == "B")

#Heatmap
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)

#Hierarchical clustering
h <- hclust(d_features)
groups <- cutree(h, k = 5)
groups
split(names(groups), groups)
plot(h)

#PCA: proportion of variance
pc <- prcomp(x_scaled)

#Plot the first two principal components with color representing tumor type 
#(benign/malignant)
data.frame(pc$x[,1:2], tumor=brca$y) %>% 
  ggplot(aes(PC1,PC2, fill = tumor, color = tumor))+
  geom_point() +
  labs(title = "first two principal components with color representing tumor type") +
  theme_economist()

#plot showing the density of first principal component
data.frame(pc$x[,1:2], tumor=brca$y) %>% 
  ggplot(aes(PC1,fill = tumor))+
  geom_density() +
  labs(title = "first principal component density with color representing tumor type") +
  theme_economist()

#boxplot of first ten principal components
data.frame(type = brca$y, pc$x[,1:10]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot() +
  theme_economist()
  
#Algorithms

#Training and test sets
# set.seed(1) if using R 3.5 or earlier
set.seed(30, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- y[-test_index]

#What proportion of the training set is benign?
mean(train_y == "B")

#What proportion of the test set is benign?
mean(test_y == "B")

#Will be using k-fold cross validation on all the algorithms
#creating the k-fold parameters, k is 10
set.seed(30, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9)

#logistic regression

#training the model using train set
set.seed(9, sample.kind = "Rounding")
train_glm <- train(train_x, as.factor(train_y),
                   method = "glm",
                   family = "binomial",
                   trControl = control)

#creating the predictions
glm_preds <- predict(train_glm, test_x)

#confusion matrix
cm_glm <- confusionMatrix(glm_preds,test_y, positive = "M")
cm_glm

#random forest
#training the model using train set
set.seed(9, sample.kind = "Rounding")
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = data.frame(mtry = seq(2,40,2)),
                  importance = TRUE,
                  trControl = control)

#best tune
train_rf$bestTune

#plot of training results
plot(train_rf)

#predictions
rf_preds <- predict(train_rf, test_x)

#variable importance
varImp(train_rf)

#confusion matrix 
cm_rf <- confusionMatrix(rf_preds, test_y, positive = "M")
cm_rf

#K Nearest Neighbors
set.seed(7, sample.kind = "Rounding")

#tuning parameter
tuning <- data.frame(k = seq(1, 20, 1))

#training the model
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning,
                   trControl = control)

#best tune
train_knn$bestTune

#plot of training model results
plot(train_knn)

#predictions
knn_preds <- predict(train_knn, test_x)

#confusion matrix
cm_knn <- confusionMatrix(knn_preds, test_y, positive = "M")
cm_knn

#Linear discriminant analysis

set.seed(7, sample.kind = "Rounding")

#training the model using the training set
train_lda <- train(train_x, train_y, 
                   method = "lda",
                   trControl = control)

#predictions
lda_preds <- predict(train_lda, test_x)

#
cm_LDA <- confusionMatrix(lda_preds, test_y, positive = "M")
cm_LDA

#Neural Network
set.seed(7, sample.kind = "Rounding")

#setting the tuning parameter alpha
tuning <- data.frame(size = seq(100), decay = seq(.01,1,.1))

#training the model on the train set
train_nn <- train(train_x, train_y,
                  method = "nnet",
                  tuneGrid = tuning,
                  trControl = control)

#creating a graph for the tuning results
ggplot(train_nn, highlight = TRUE) +
  ggtitle("Neural Network")

#finding best tune
train_nn$bestTune

#creating predictions
nn_preds <- predict(train_nn, test_x)

#getting accuracy results
cm_nn <- confusionMatrix(nn_preds, test_y, positive = "M")

#viewing accuracy results
cm_nn

#ensemble

preds <- data.frame(log_r = glm_preds, 
                    rf = rf_preds,
                    knn = knn_preds,
                    lda = lda_preds,
                    nn = nn_preds)
preds

ensemble <- apply(preds,1,function(x) names(which.max(table(x))))

ensemble <- as.factor(ensemble)

cm_ensemble <- confusionMatrix(ensemble, test_y, positive = "M")
cm_ensemble

#results

cm_list <- list(log_r = cm_glm,
                rf = cm_rf,
                knn = cm_knn,
                lda = cm_LDA,
                nn = cm_nn,
                ensemble = cm_ensemble)

cm_results <- sapply(cm_list, function(x) x$byClass)
cm_results

results_table <- kable(cm_results)
results_table

#Best model
which.max(cm_results[1,])
which.max(cm_results[2,])
which.max(cm_results[7,])
which.max(cm_results[11,])
