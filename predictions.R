###################################################################################
### Flight delay classification ###################################################
###################################################################################


# clear workspace
rm(list=ls())

# computer wd
#setwd("C:/KaggleData")

# Kaggle kernel path (not wd!):
# setwd("../input")

# install and load packages
# Packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  bigmemory, xgboost, mlr, gbm, rgenoud, #mlrMBO, #devtools, stringr,
  nnet, randomForest, caret, e1071, data.table, glmnet, #rstanarm, tibble, car,
  varhandle, foreign, ggfortify, broom, grid, gridExtra, timeDate, fastAdaboost, #GGally,
  # those beneath always load last
  dplyr,
  readr,
  ggplot2,
  tidyverse)
  
#install.packages("autoxgboost")

sample <- read.csv(file="../input/vu-aml2018/sample_submission.csv", header=TRUE, sep=",")

# load data files created by other kernel
# other kernel cleans data, engineers features and performs
# randomforest imputation on the missinv values
load(file = "../input/rfcleanimp/ctestimp.RDA")
load(file = "../input/rfcleanimp/ctrainimp.RDA")

# recode month variable as factor (forgot in data cleaning)
ctrain.imp$month <- as.factor(ctrain.imp$month)
ctest.imp$month <- as.factor(ctest.imp$month)

# create weekday factor variable and holiday indicator (forgot in data cleaning)

# create date variable using timeDate package
# then generate weekday factor variable
ctrain.imp$timechar <- paste("2013", ctrain.imp$month, ctrain.imp$day, sep="-")
ctrain.imp$date <- timeDate(ctrain.imp$timechar, format = "%Y-%m-%d")
ctrain.imp$dayofweek <- dayOfWeek(ctrain.imp$date)
ctrain.imp$dayofweek <- as.factor(ctrain.imp$dayofweek)

ctest.imp$timechar <- paste("2013", ctest.imp$month, ctest.imp$day, sep="-")
ctest.imp$date <- timeDate(ctest.imp$timechar, format = "%Y-%m-%d")
ctest.imp$dayofweek <- dayOfWeek(ctest.imp$date)
ctest.imp$dayofweek <- as.factor(ctest.imp$dayofweek)

# generate holiday variable
hols <- as.character(holidayNYSE(year=2013))

ctrain.imp$datechar <- as.character(ctrain.imp$date)
ctrain.imp$holiday <- 0
ctrain.imp$holiday[ctrain.imp$datechar %in% hols] <- 1
summary(ctrain.imp$holiday)

ctest.imp$datechar <- as.character(ctest.imp$date)
ctest.imp$holiday <- 0
ctest.imp$holiday[ctest.imp$datechar %in% hols] <- 1
summary(ctest.imp$holiday)

# delete helper variables
ctrain.imp$timechar <- NULL
ctrain.imp$date <- NULL
ctrain.imp$datechar <- NULL

ctest.imp$timechar <- NULL
ctest.imp$date <- NULL
ctest.imp$datechar <- NULL

# add quadratic term for sched_dep to adjust for night effects
ctrain.imp$dep2 <- ctrain.imp$sched_dep^2
ctest.imp$dep2 <- ctest.imp$sched_dep^2


#define model formula
delay.form <- as.formula("is_delayed ~ month + dayofweek + holiday + carrier +
                         origin  + distance + alt_orig + 
                         lat_dest + lon_dest + alt_dest + tzone_dest + 
                         sched_dep + dep2 + sched_arr + sched_speed + precip + 
                         pressure + visib +
                         temp + dewp + humid + wind_dir + wind_speed + 
                         wind_gust")

###################################################################################
###################################################################################
###################################################################################
###################################################################################



### Simple Logit-Classifyer ###

logit.model <- glm(delay.form, data = ctrain.imp, family = "binomial")
summary(logit.model)
print(logit.model)

# Comparison: Ordinary Least Squares Regression
#ctrain.imp$is_delayed <- as.numeric(ctrain.imp$is_delayed)
#ols.model <- lm(delay.form, data = ctrain.imp)
#summary(ols.model)
#ctrain.imp$is_delayed <- as.factor(ctrain.imp$is_delayed)

# predictions on test data
logitpred <- stats::predict(logit.model, newdata = ctest.imp, type="response")
logitpred.train <- stats::predict(logit.model, newdata = ctrain.imp, type="response")

# create vector for submission
sample.logit <- sample
sample.logit$is_delayed <- logitpred
summary(sample.logit$is_delayed)

#score 0.6xx
write.csv(sample.logit, file = "samplelogit.csv", row.names=FALSE)

detach("package:glmnet", unload=TRUE)

###################################################################################

# AdaBoost

#ada.boost <- adaboost(delay.form, data=ctrain.imp, nIter = 100)

# predict delay probability in test set
#adapred <- predict(ada.boost, newdata=ctest.imp)
#adapred <- adapred$prob
#adapred <- adapred[, 2]

#sample.ada <- sample
#sample.ada$is_delayed <- adapred
#write.csv(sample.ada, file = "sampleada.csv", row.names=FALSE)


###################################################################################


# Random Forest
rf.train <- ctrain.imp
rf.test <- ctest.imp

rf.train$unif.id <- runif(nrow(rf.train), min=0, max=1)
rf.train.small <- subset(rf.train, unif.id<0.1 )
rf.formula <- delay.form
rf.train$unif.id <- NULL
# tune mtry
#rf.y <- rf.train$is_delayed
#rf.x <- rf.train[ , -which(names(rf.train) %in%c("id", "is_delayed", "unif.id"))]

# perform hyperparameter selection for mtry
#best.try <- tuneRF(x = rf.x, y = rf.y, stepFactor=1.5, improve=0.09, ntree=50, plot = FALSE, mtryStart = 10)

# use caret package and train on smaller dataset (25k obs)

# define crossvalidation parameters and grid to search over

#rf.control <- trainControl(method="repeatedcv", number=5, 
#                        repeats=1, search="grid",
#                       classProbs=TRUE)

#rf.grid <- expand.grid(.mtry=c(1,2,3,11,16,20))

# set seed for reproducibility
#set.seed(123)
# rename invalid factor levels ("0", "1" are invalid labels)
#levels(rf.train$is_delayed) <- c("nodelay", "delay")
#levels(rf.train.small$is_delayed) <- c("nodelay", "delay")

# train cross validation grid search
#rf.train.small <- subset(rf.train, unif.id<0.01 )
#print("randomForest grid search")
#rf.small.gridsearch <- train(rf.formula, data=rf.train.small, 
#                             method="rf", metric="Accuracy", 
#                             tuneGrid=rf.grid, 
#                             trControl=rf.control,
#                            verbose=TRUE)
#print(rf.small.gridsearch)
#plot(rf_gridsearch)


# select optimal mtry hyperparameter
#opti <- rf.small.gridsearch$bestTune$mtry


# RF model
rf.model <- randomForest(rf.formula, data=rf.train, do.trace=TRUE, mtry = 8, ntree = 700 )


# predictions on test set
rfpred <- predict(rf.model, rf.test, type="prob")
rfpred.train <- predict(rf.model, rf.train, type="prob")
summary(rfpred)

#save vector for submission
sample.rf<- sample
sample.rf$is_delayed<-rfpred[,2]
write.csv(sample.rf, file = "samplerf.csv", row.names=FALSE)




###################################################################################




# Gradient Boosting Model Classifier

# prepare datasets
#gbm.train <- ctrain.imp
#gbm.train$is_delayed <- as.numeric(gbm.train$is_delayed) - 1
#gbm.test <- ctest.imp
#gbm.train$unif <- runif(nrow(gbm.train), min=0, max=1)
#gbm.train.small <- subset(gbm.train, unif<0.01 )
#gbm.formula <- delay.form


# hyperparameter optimization using caret package
#gbm.train.small$is_delayed <- as.factor(gbm.train.small$is_delayed)
#gbm.train$is_delayed <- as.factor(gbm.train$is_delayed)
#levels(gbm.train$is_delayed) <- c("nodelay", "delay")
#levels(gbm.train.small$is_delayed) <- c("nodelay", "delay")

#gbm.train.small$is_delayed <- as.numeric(gbm.train.small$is_delayed)
#gbm.train.small$is_delayed <- as.factor(gbm.train.small$is_delayed)
#levels(gbm.train.small$is_delayed)
#levels(gbm.train.small$is_delayed) <- c("no", "yes")


#trainControl <- trainControl(method="cv", number=4, classProbs = TRUE)


#caretGrid.small <- expand.grid(.n.trees = c(90,120,150),
#                         .interaction.depth = c(2,4,5),
#                        .shrinkage = c(0.05, 0.075, 0.1, 0.22),
#                         .n.minobsinnode = c(10))


#caretGrid <- expand.grid(.n.trees = c(150),
#                         .interaction.depth = c(2,3,4,5),
#                         .shrinkage = c(0.05, 0.075, 0.1, 0.15, 0,22),
#                         .n.minobsinnode = c(5,10,30)
#                          )


#set.seed(123)

# to avoid problems with caret, unload all packages
#lapply(paste('package:',names(sessionInfo()$otherPkgs),sep=""),detach,character.only=TRUE,unload=TRUE)

#library(caret)
# Training and validating gradient boosting model
#gbm.caret <- caret::train(gbm.formula
#                   , data=gbm.train.small
#                   , distribution="bernoulli"
#                   , method="gbm"
#                   , verbose=TRUE
#                  , tuneGrid=caretGrid.small
#                   ,  maxit=100) 

#print(gbm.caret)

# optimal hyperparameter
#shr <- gbm.caret$bestTune$shrinkage
#ntr <- gbm.caret$bestTune$n.trees
#depth <- gbm.caret$bestTune$interaction.depth
#minobs <-gbm.caret$bestTune$n.minobsinnde


# train GBM on whole data with tuned hyperparameters

# model
#gbm.model.tuned <- gbm(gbm.formula, distribution = "bernoulli",
#                 data = gbm.train.small, n.trees = ntr,
#                 interaction.depth = depth, shrinkage = shr,
#                 n.minobsinnode = minobs,
#                 keep.data = TRUE, verbose = TRUE)


# manually selected hyperparameters model:
#gbm.model <- gbm(gbm.formula, distribution = "bernoulli",
#                 data = gbm.train, n.trees = 90,
#                interaction.depth = 2, shrinkage = 0.01,
#                 keep.data = TRUE, verbose = TRUE)


# predict on test set
#predgbm <- predict(gbm.model, gbm.test, n.trees = 90,
#                   type = "response")
#predgbm.train <- predict(gbm.model, gbm.train, n.trees = 90,
#                   type = "response")

#summary(predgbm)

#save vector for submission
#sample.gbm <- sample
#sample.gbm$is_delayed<-predgbm
#summary(sample.gbm$is_delayed)
#write.csv(sample.gbm, file = "samplegbm.csv", row.names=FALSE)



##############################################################################################################
# Extreme gradient boosting machine classifier
##############################################################################################################


#library(xgboost)


# Variable Normalization and One-Hot-Encoding of Factor Variables


# prepare data sets
xboost.dat <- ctrain.imp
set.seed(123)
unif = runif(nrow(xboost.dat), min=0, max=1)

sxboost.dat<- subset(xboost.dat, unif < 0.1)
sxboost.dat$id <- NULL

xboost.test <- ctest.imp
xboost.test$id <- NULL

xboost.dat$is_delayed <- as.factor(xboost.dat$is_delayed)
xboost.dat$id <- NULL


# Normalize Features
sxboost.dat <- normalizeFeatures(sxboost.dat, target = "is_delayed")
xboost.dat <- normalizeFeatures(xboost.dat, target = "is_delayed")
xboost.test <- normalizeFeatures(xboost.test)

# One-Hot encoding of factor variables

sxboost.dat <- createDummyFeatures(sxboost.dat, 
                                  target = "is_delayed",
                                  cols = c("carrier", "origin", "dest",
                                           "tzone_dest", "month", "dayofweek"))
                                           
xboost.test <- createDummyFeatures(xboost.test, cols = c("carrier", "origin", "dest",
                                                       "tzone_dest", "month", "dayofweek"))

xboost.dat <- createDummyFeatures(xboost.dat,
                                  target = "is_delayed",
                                  cols = c("carrier", "origin", "dest",
                                            "tzone_dest", "month", "dayofweek"))


# convert is_delayed to numeric
xboost.dat$is_delayed <- as.numeric(xboost.dat$is_delayed)-1
sxboost.dat$is_delayed <- as.numeric(sxboost.dat$is_delayed)-1

# code as data tables
xboost.dat <- setDT(xboost.dat)
sxboost.dat <- setDT(sxboost.dat)
xboost.test <- setDT(xboost.test)

# convert to xgboost preferred matrix type

# first, separate target and variables
xb.delay <- xboost.dat$is_delayed
xboost.dat <- xboost.dat[,-c("is_delayed")]
xboost.mat <- as.matrix(xboost.dat)
sxb.delay <- sxboost.dat$is_delayed
sxboost.dat <- sxboost.dat[,-c("is_delayed")]
sxboost.mat <- as.matrix(sxboost.dat)
# also recode test data as matrix 
xboost.test.mat <- as.matrix(xboost.test)

sxboost.mat <- xgb.DMatrix(data = sxboost.mat,label = sxb.delay)
xboost.mat <- xgb.DMatrix(data = xboost.mat,label = xb.delay)
xboost.test.mat <- xgb.DMatrix(data = xboost.test.mat)


# run model in easy syntax version
#xboost.model <- xgboost(data=sxboost.mat,
#                    max.depth = 4,
#                    eta = 0.05, nthread = 2, nrounds = 100,
#                    objective = "binary:logistic",
#                    subsample = 0.7, #add random
#                    colsample_bytree =0.7, #dropout
#                    shrinkage = 0.3,
#                    verbose=TRUE)


#params <- list(booster = "gbtree", objective = "binary:logistic",
#               eta=0.3, gamma=0, max_depth=6,
#               min_child_weight=1, subsample=1, colsample_bytree=1)

#15-250 nrounds optimal
#xgbcv <- xgb.cv( params = params, data = xboost.mat, nrounds = 200,
#                 nfold = 5, showsd = T, stratified = T, print_ever_n = 1,
#                 early_stopping_rounds = 10, maximize = F)

# tune additional hyperparameters using mlr package

# need to adjust data for mlr package

xboost.dat <- cbind(xb.delay, xboost.dat)
names(xboost.dat)[names(xboost.dat) == "xb.delay"] <- "is_delayed"
xboost.dat$is_delayed <- as.factor(xboost.dat$is_delayed)
levels(xboost.dat$is_delayed) <- c("nodelay", "delay")
xboost.dat <- as.data.frame(xboost.dat)

sxboost.dat <- cbind(sxb.delay, sxboost.dat)
names(sxboost.dat)[names(sxboost.dat) == "sxb.delay"] <- "is_delayed"
sxboost.dat$is_delayed <- as.factor(sxboost.dat$is_delayed)
levels(sxboost.dat$is_delayed) <- c("nodelay", "delay")
sxboost.dat <- as.data.frame(sxboost.dat)

xboost.test <- as.data.frame(xboost.test)

# define tasks

smalltask <- makeClassifTask (data = sxboost.dat, target = "is_delayed",
                              positive = "delay")
bigtask <- makeClassifTask (data = xboost.dat, target = "is_delayed",
                            positive = "delay")
# define learner

boost.learner <- makeLearner("classif.xgboost", predict.type = "prob",
                             par.vals = list(objective = "binary:logistic",
                                             eval_metric = "error"))


# create hyperparameter space

boost.parms <- makeParamSet(
  # The number of trees in the model 
  makeIntegerParam("nrounds", lower = 90, upper = 250),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 4L, upper = 10L),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = 0.05, upper = .25),
  # L2 regularization - prevents overfitting
  makeNumericParam("lambda", lower = -2, upper = -0.8, trafo = function(x) 10^x),
  makeNumericParam("min_child_weight",lower = 2L,upper = 6L),
  makeNumericParam("subsample",lower = 0.5,upper = 0.8),
  makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))


# set resampling strategy
rdesc <- makeResampleDesc("CV", stratify = T, iters=5L)

# define how many iterations the random search should so
control <- makeTuneControlRandom(maxit = 20)


# perform hyperparameter tuning

mytune <- tuneParams(learner = boost.learner,
                     task = smalltask,
                     resampling = rdesc,
                     measures = acc,
                     par.set = boost.parms,
                     control = control,
                     show.info = T)


# Update found hyperparameters to model
learner.tuned <- setHyperPars(learner = boost.learner,
                              par.vals = mytune$x)


# Re-train model using tuned hyperparameters (and full training set)
#big.learner <- train(learner_tuned, bigtask)


big.learner <- mlr::train(learner.tuned, bigtask)

# Predict test values

# to avoid problems with different column names
colnames(xboost.test) <- big.learner$features
predxgb <- predict(big.learner, newdata = xboost.test)
pred.prob.xgb <- getPredictionProbabilities(predxgb)

predxgb.train <- predict(big.learner, bigtask)
pred.prob.xgb.train <- getPredictionProbabilities(predxgb.train)

#save vector for submission
sample.xgb <- sample
sample.xgb$is_delayed <- pred.prob.xgb
write.csv(sample.xgb, file = "samplexgb.csv", row.names=FALSE)




###################################

# create ensemble (averaged) predicted probability over various models:

sample.ens <- sample
sample.ens$logit <- sample.logit$is_delayed 
sample.ens$rf <- sample.rf$is_delayed
#sample.ens$gbm <- sample.gbm$is_delayed
sample.ens$xgb <- sample.xgb$is_delayed


write.csv(sample.ens, file = "ensembledata.csv", row.names=FALSE)

# create minmax ensemble for randomForest and XGBOOST

rfxgb.ens <- sample.ens
rfxgb.ens$max <- pmax(rfxgb.ens$rf, rfxgb.ens$xgb)
rfxgb.ens$min <- pmin(rfxgb.ens$rf, rfxgb.ens$xgb)

rfxgb.ens$is_delayed <- if_else(rfxgb.ens$xgb>0.5,
                                      rfxgb.ens$max,
                                      rfxgb.ens$min)
                                      
rfxgb.ens <- rfxgb.ens[, c(1:2)]

write.csv(rfxgb.ens, file = "rfxgbminmax.csv", row.names=FALSE)

# create average ensemble of randomForest and XGBOOST

avg.ens <- rfxgb.ens
avg.ens$is_delayed <- (sample.ens$rf + sample.ens$xgb)/2

write.csv(avg.ens, file = "rfxgbavg.csv", row.names=FALSE)


# stack different models via logistic regression

# need dataset of training labels and predicted probabilities from each classifier
# then train logistic regression to optimally combine these models

stack.dat <- ctrain.imp[, c(1:2)]
stack.dat$logit <- logitpred.train
stack.dat$rf <- rfpred.train[,2]
#stack.dat$gbm <- predgbm.train
stack.dat$xgb <- pred.prob.xgb.train

#  train metamodel

meta.logit <- glm(is_delayed ~ logit + rf + xgb, data = stack.dat, family = "binomial")
summary(meta.logit)

stacked.pred <- stats::predict(meta.logit, newdata = sample.ens, type="response")

sample.stacked <- sample
sample.stacked$is_delayed <- stacked.pred

write.csv(sample.stacked, file = "stackedmodel.csv", row.names=FALSE)










# help line