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
  bigmemory,
  plm,
  rgenoud,
  mlr,
  gbm,
  xgboost,
  #mlrMBO,
  devtools,
  stringr,
  randomForestSRC,
  mice,
  imputeMissings,
  nnet,
  randomForest,
  caret,
  e1071,
  glmnet,
  kernlab,
  rstanarm,
  tibble,
  car,
  varhandle,
  foreign,
  ggfortify,
  broom,
  grid,
  gridExtra,
  # those beneath always load last
  dplyr,
  readr,
  ggplot2,
  tidyverse)
  
#install.packages("autoxgboost")

# load data
airports <- read.csv(file="../input/airports.csv", header=TRUE, sep=",")
test <- read.csv(file="../input/test.csv", header=TRUE, sep=",")
train <- read.csv(file="../input/train.csv", header=TRUE, sep=",")
weather <- read.csv(file="../input/weather.csv", header=TRUE, sep=",")
sample <- read.csv(file="../input/sample_submission.csv", header=TRUE, sep=",")

# merge data sets

# first, merge append test to training data
train$train_ind = 1
test$test_ind = 1
ttdata <- bind_rows(train,test)

# next merge airport characteristics of origin airport

air <- airports
air$origin <- NA
air$origin[air$faa == "EWR"] <- "EWR"
air$origin[air$faa == "JFK"] <- "JFK"
air$origin[air$faa == "LGA"] <- "LGA"


air$origin <- as.factor(air$origin)

levels(air$origin)

temp <- left_join(ttdata, air)

# drop unnecessary merged variables from airport set
temp$faa <- NULL
temp$name <- NULL
temp$tz <- NULL
temp$dst <- NULL
# those variables are the same for all obs. except variable altitude
temp$tzone <- NULL
temp$lat <- NULL
temp$lon <- NULL


# rename origin airport variables
names(temp)[names(temp) == "alt"] <- "alt_orig"


# Next merge airport characteristics of destination airport to train data

air$dest <- air$faa
levels(air$dest)
air$dest <- as.character(air$faa)
temp$dest <- as.character(temp$dest)

# merge data sets 
names(temp)[names(temp) == "origin"] <- "rename"

temp <- left_join(temp, air)

temp$origin <- NULL
names(temp)[names(temp) == "rename"] <- "origin"

temp$dest <- as.factor(temp$dest)
levels(temp$dest)

# drop unnecessarily merged variables from airport set
temp$faa <- NULL
temp$name <- NULL
temp$tz <- NULL
temp$dst <- NULL

# rename origin airport variables
names(temp)[names(temp) == "lat"] <- "lat_dest"
names(temp)[names(temp) == "lon"] <- "lon_dest"
names(temp)[names(temp) == "alt"] <- "alt_dest"
names(temp)[names(temp) == "tzone"] <- "tzone_dest"

# record tzone as factor
temp$tzone_dest = as.factor(temp$tzone_dest)

# Now we have all the relevant airport specific information in our
# temporary data set

# Next, we match the airport specific weather information to the 
# individual flights in the temporary data frame

# First, create "hour" variable to have exact matching with weather data
# To do this, split sched_dep_time and sched_arr_time up
temp$sched_dep_time <- as.character(temp$sched_dep_time)
temp$sched_arr_time <- as.character(temp$sched_arr_time)
temp <- separate(temp, sched_dep_time, into = c('sched_dep_h', 'sched_dep_m'), sep = -2, convert = TRUE)
temp <- separate(temp, sched_arr_time, into = c('sched_arr_h', 'sched_arr_m'), sep = -2, convert = TRUE)
temp$sched_dep <- temp$sched_dep_h+temp$sched_dep_m/60
temp$sched_arr <- temp$sched_arr_h+temp$sched_arr_m/60



# rename hour variable in weather.csv to match sched_dep_h
names(weather)[names(weather) == "hour"] <- "sched_dep_h"

temp <- left_join(temp, weather)

# next, we split the merged train and test data again
# and remove unnecessary variables

ctrain <- subset(temp, train_ind==1)

ctrain$is_delayed <- as.factor(ctrain$is_delayed)


# define list with variables to be dropped

drop.cols = c("train_ind", "test_ind", "year",
              "sched_dep_h", "sched_dep_m", "sched_arr_h",
              "sched_arr_m", "time_hour")
ctrain <- select(ctrain, -one_of(drop.cols))

# create test set
ctest <- subset(temp, test_ind==1)

# remove unnecessary variables also in test set
ctest <- select(ctest, -one_of(drop.cols))

#remove missing is_delayed response in test set
ctest$is_delayed <- NULL


###################################################################################
### Missing Values ################################################################
###################################################################################



# count missing values
sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))
# windgust, wind_dir and pressure have high number of NA
sapply(weather, function(x) sum(is.na(x)))
sapply(airports, function(x) sum(is.na(x)))
sapply(ctest, function(x) sum(is.na(x)))
sapply(ctrain, function(x) sum(is.na(x)))

# remove old objects
rm(temp, ttdata, air, airports)


# Impute missing values 

# Random Forest/MICE Imputation takes too long to compute, hence 
# median/mode imputation as weak alternative is implemented

# RandomForest Imputation
set.seed(123)
#ctest.imp <- impute(data = ctest, nimpute = 1)
#ctrain.imp <- impute(data = ctrain, nimpute = 1)
ctrain.test <- ctrain
ctrain.test$unif <- runif(nrow(ctrain.test), min = 0, max = 1)
ctrain.test <- subset(ctrain.test, unif<0.1)

ctrain.imp<- impute.rfsrc(data = ctrain,
       ntree = 500, nodesize = 3, nsplit = 10,
       nimpute = 3, fast = TRUE, blocks = 2,
       max.iter = 10, eps = 0.1,
       ytry = NULL, always.use = NULL, verbose = TRUE)
       
ctest.imp<- impute.rfsrc(data = ctest,
       ntree = 500, nodesize = 3, nsplit = 10,
       nimpute = 3, fast = TRUE, blocks = 2,
       max.iter = 10, eps = 0.1,
       ytry = NULL, always.use = NULL, verbose = TRUE)


# MICE imputation
#ctest.imp <- mice(ctest)
#sapply(ctest.imp, function(x) sum(is.na(x)))
#ctrain.imp <- mice(ctrain)
#sapply(ctrain.imp, function(x) sum(is.na(x)))

# median/mode imputation
#ctest.imp <- imputeMissings::impute(ctest, method = "median/mode", flag = FALSE)
sapply(ctest.imp, function(x) sum(is.na(x)))
#ctrain.imp <- imputeMissings::impute(ctrain, method = "median/mode", flag = FALSE)
sapply(ctrain.imp, function(x) sum(is.na(x)))

# compare means of variables
sapply(ctest, function(x) mean(x))
sapply(ctest.imp, function(x) mean(x))
sapply(ctrain, function(x) mean(x))
sapply(ctrain.imp, function(x) mean(x))


################ other stuff for data cleaning

# create additional covariate: scheduled flight length
# if arrival<departure: flight time is (24-dep)+arr
# if departure<arrival: flight time is arr-dep
ctrain.imp$sched_dur = 0
# arr<dep
ctrain.imp$time1 <- 24 - ctrain.imp$sched_dep + ctrain.imp$sched_arr
ctrain.imp$sched_dur[ctrain.imp$sched_arr<=ctrain.imp$sched_dep] <- ctrain.imp$time1[ctrain.imp$sched_arr<=ctrain.imp$sched_dep]
# depp<arr
ctrain.imp$time2 <- ctrain.imp$sched_arr - ctrain.imp$sched_dep
ctrain.imp$sched_dur[ctrain.imp$sched_arr>ctrain.imp$sched_dep] <- ctrain.imp$time2[ctrain.imp$sched_arr>ctrain.imp$sched_dep]


ctest.imp$sched_dur = 0
# arr<dep
ctest.imp$time1 <- 24 - ctest.imp$sched_dep + ctest.imp$sched_arr
ctest.imp$sched_dur[ctest.imp$sched_arr<=ctest.imp$sched_dep] <- ctest.imp$time1[ctest.imp$sched_arr<=ctest.imp$sched_dep]
# depp<arr
ctest.imp$time2 <- ctest.imp$sched_arr - ctest.imp$sched_dep
ctest.imp$sched_dur[ctest.imp$sched_arr>ctest.imp$sched_dep] <- ctest.imp$time2[ctest.imp$sched_arr>ctest.imp$sched_dep]

# remove helper variables
ctrain.imp$time1 <- NULL
ctrain.imp$time2 <- NULL
ctest.imp$time1 <- NULL
ctest.imp$time2 <- NULL


# creade additional covariate:
# ratio of distance/sched_dur (=speed) to average dist/dur for non_delayed flights 

# ctrain
ctrain.imp$sched_speed <- 0
ctrain.imp$sched_speed <- ctrain.imp$distance / ctrain.imp$sched_dur
summary(ctrain.imp$sched_speed)
ctrain.imp$rel_speed <- ctrain.imp$sched_speed / median(ctrain.imp$sched_speed[ctrain.imp$is_delayed == 0])
summary(ctrain.imp$rel_speed)
# ctest
ctest.imp$sched_speed <- 0
ctest.imp$sched_speed <- ctest.imp$distance / ctest.imp$sched_dur
summary(ctest.imp$sched_speed)
ctest.imp$rel_speed <- ctest.imp$sched_speed / median(ctrain.imp$sched_speed[ctrain.imp$is_delayed == 0])
summary(ctest.imp$rel_speed)




# next we have to fix the destination "dest" varibale
# factor variable dest has more than 53 categories => causes problems
# with several packages/algorithms

# to reduce factors, we group the most "uninteresting" destinations
# in one factor level "other" (median share of delayed flights)

# therefore we first group the destinations by percentage delayed
dest.num <- as.numeric(ctrain.imp$is_delayed) - 1 
dest.delay <- aggregate(dest.num,list(ctrain.imp$dest), mean)
dest.delay <- dest.delay[order(-dest.delay$x),]
names(dest.delay)[names(dest.delay) == "x"] <- "delayshare"
summary(dest.delay$delayshare)

# group all destinations with near median share of delayed flights
# into one category "other", such that the total number of levels
# is reduced and algorithms converge better

# group the modal 70% of destinations in other category
lquart <- quantile(dest.delay$delayshare, prob = 0.15)
uquart <- quantile(dest.delay$delayshare, prob = 0.85)

dest.delay$othergroup <- 1
dest.delay$othergroup[dest.delay$delayshare<=lquart] <- 0
dest.delay$othergroup[dest.delay$delayshare>=uquart] <- 0
summary(dest.delay$othergroup)

# Group.1 = old dest, Group.2 = new dest with "other" category
dest.delay$Group.2 <- as.character(dest.delay$Group.1)
dest.delay$Group.2[dest.delay$othergroup == 1] <- "other"

dest.delay$Group.2 <- as.factor(dest.delay$Group.2)
other.dest <- subset(dest.delay, othergroup == 1)
# to get new set of levels, do: to characther and again to factor
other.dest$Group.1 <- as.character(other.dest$Group.1)
other.dest$Group.1 <- as.factor(other.dest$Group.1)
# make same list for "interesting" destinations
# that are at the tail of the distribution of delayed flights
important.dest <- subset(dest.delay, othergroup == 0)
important.dest$Group.1 <- as.character(important.dest$Group.1)
important.dest$Group.1 <- as.factor(important.dest$Group.1)

# now we have divided all destinations into important and "other"
other.dest <- levels(other.dest$Group.1)
important.dest <- levels(important.dest$Group.1)

#check: must sum up to length of levels(dist)
length(levels(ctrain.imp$dest))
length(other.dest)
length(important.dest)

# Now that we identified the important destinations, we have to
# change the labels of all the others in ctrain and ctest to "other"

ctrain.imp$dest <- as.character(ctrain.imp$dest)
ctrain.imp$dest[!(ctrain.imp$dest %in% important.dest)] <- "other"
ctrain.imp$dest <- as.factor(ctrain.imp$dest)
length(levels(ctrain.imp$dest))


# same procedure for ctest
ctest.imp$dest <- as.character(ctest.imp$dest)
ctest.imp$dest[!(ctest.imp$dest %in% important.dest)] <- "other"
ctest.imp$dest <- as.factor(ctest.imp$dest)
levels(ctest.imp$dest) <- c(levels(ctest.imp$dest), "LEX", "LGA")

length(levels(ctest.imp$dest))
length(levels(ctrain.imp$dest))
# now dest has much fewer levels and can be used for training

# save cleaned dataset
save(ctest.imp, file="ctestimp.RDA")
save(ctest.imp, file="ctestimp.csv")
save(ctrain.imp, file="ctrainimp.RDA")
save(ctrain.imp, file="ctrainimp.csv")

# remove temporary files to keep workspace clean
rm(weather, train, test, ctrain, ctest, dest.num,
   important.dest, lquart, uquart, other.dest, drop.cols, dest.delay)



#define model formula
delay.form <- as.formula("is_delayed ~ month + day + carrier +
                         origin + dest + distance + alt_orig + 
                         lat_dest + lon_dest + alt_dest + tzone_dest + 
                         sched_dep + sched_arr + sched_speed + precip+ 
                         pressure + visib")
