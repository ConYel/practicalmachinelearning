---
title: "Practical Machine Learning Course Final Project"
author: "Constantinos Yeles"
date: "July 5, 2018"
output: 
  html_document: 
    keep_md: yes
editor_options: 
  chunk_output_type: console
---

# Practical Machine Learning Course - Final Project

## First download the data sets in the working directory manually
For the purpose of the analysis we need consistent data and so we modify empty stings and division error strings with zero as NA values

```r
training = read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!",""))
testing  = read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!",""))
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

```r
View(training)
sum(is.na(training$min_roll_belt))
```

```
## [1] 19216
```

## Cleaning the Data From NAs and variables with minimun to none predictive value 
As we can observe, there are 19622 observations and 160 variables for the training set and the test set has 20 observations and 160 variables.
Many of these observations have NA values and we need to clean the data sets from them.

```r
training = training[, colSums(is.na(training)) == 0]
testing = testing[, colSums(is.na(testing)) == 0]
```

Looking at the Data we can observe that the first seven columns have little valuable or none information.

```r
names(training[,1:7])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```
So we drop them from the Data

```r
trData = training[, -c(1:7)]
teData = testing[, -c(1:7)]
```

## Loading the R packages for Machine learning

```r
library(caret); library(rattle); library(rpart); library(rpart.plot)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

## Splitting the Data for Cross-Validation

```r
set.seed(4365)
inTrain = createDataPartition(trData$classe, p = 0.75, list = FALSE)
trainData = trData[inTrain, ]
validData = trData[-inTrain, ]
```

## Searching for the best Machine Learning Method for prediction regarding these data
### 1st using Random Forests

```r
rForest = randomForest(classe~., data=trainData, ntree = 300)
print(rForest, digits = 3)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = trainData, ntree = 300) 
##                Type of random forest: classification
##                      Number of trees: 300
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.48%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4180    4    0    0    1 0.001194743
## B   16 2828    4    0    0 0.007022472
## C    0   11 2553    3    0 0.005453837
## D    0    0   22 2387    3 0.010364842
## E    0    0    1    5 2700 0.002217295
```

```r
valid_rf = predict(rForest, validData)
(confMatr_rf = confusionMatrix(validData$classe, valid_rf))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    0  946    3    0    0
##          C    0    6  848    1    0
##          D    0    0    5  797    2
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9961         
##                  95% CI : (0.994, 0.9977)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9951         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9927   0.9907   0.9975   0.9978
## Specificity            0.9997   0.9992   0.9983   0.9983   0.9998
## Pos Pred Value         0.9993   0.9968   0.9918   0.9913   0.9989
## Neg Pred Value         1.0000   0.9982   0.9980   0.9995   0.9995
## Prevalence             0.2843   0.1943   0.1746   0.1629   0.1839
## Detection Rate         0.2843   0.1929   0.1729   0.1625   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9999   0.9959   0.9945   0.9979   0.9988
```

```r
(accuracy_rf = confMatr_rf$overall[1])
```

```
##  Accuracy 
## 0.9961256
```

```r
(predict(rForest, teData))
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
So Random Forest method has a very high Accuracy : 0.99531.

### 2nd using SVM

```r
library(e1071)
svm1 = svm(classe ~ ., data = trainData)
valid_svm = predict(svm1, validData)
(confMatr_svm = confusionMatrix(validData$classe, valid_svm))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1389    1    4    0    1
##          B   81  847   17    0    4
##          C    3   26  818    8    0
##          D    1    1   76  722    4
##          E    0    4   22   19  856
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9445          
##                  95% CI : (0.9378, 0.9508)
##     No Information Rate : 0.3006          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9297          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9423   0.9636   0.8730   0.9640   0.9896
## Specificity            0.9983   0.9747   0.9907   0.9803   0.9889
## Pos Pred Value         0.9957   0.8925   0.9567   0.8980   0.9501
## Neg Pred Value         0.9758   0.9919   0.9706   0.9934   0.9978
## Prevalence             0.3006   0.1792   0.1911   0.1527   0.1764
## Detection Rate         0.2832   0.1727   0.1668   0.1472   0.1746
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9703   0.9691   0.9318   0.9721   0.9892
```

```r
(accuracy_svm = confMatr_svm$overall[1])
```

```
##  Accuracy 
## 0.9445351
```

```r
(predict(svm1, teData))
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
The svm method has a good Accuracy 0.9445351 but not that high as Random Forest.

### 3rd using classification tree

```r
classtr = train(classe ~ ., method = "rpart", data= trainData)
print(classtr, digits = 3)
```

```
## CART 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa 
##   0.0366  0.507     0.3605
##   0.0595  0.420     0.2174
##   0.1152  0.335     0.0782
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.0366.
```

```r
fancyRpartPlot(classtr$finalModel)
```

![](index_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

```r
valid_tr = predict(classtr, validData)
(confMatr_tr = confusionMatrix(validData$classe, valid_tr))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1270   23   95    0    7
##          B  380  334  235    0    0
##          C  394   25  436    0    0
##          D  349  152  303    0    0
##          E  126  124  240    0  411
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4998          
##                  95% CI : (0.4857, 0.5139)
##     No Information Rate : 0.5137          
##     P-Value [Acc > NIR] : 0.9748          
##                                           
##                   Kappa : 0.3468          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5042  0.50760  0.33308       NA  0.98325
## Specificity            0.9476  0.85516  0.88345   0.8361  0.89077
## Pos Pred Value         0.9104  0.35195  0.50994       NA  0.45616
## Neg Pred Value         0.6441  0.91808  0.78439       NA  0.99825
## Prevalence             0.5137  0.13418  0.26692   0.0000  0.08524
## Detection Rate         0.2590  0.06811  0.08891   0.0000  0.08381
## Detection Prevalence   0.2845  0.19352  0.17435   0.1639  0.18373
## Balanced Accuracy      0.7259  0.68138  0.60826       NA  0.93701
```

```r
(accuracy_tr = confMatr_tr$overall[1])
```

```
##  Accuracy 
## 0.4997961
```

```r
(predict(classtr, teData))
```

```
##  [1] C A C A A C C A A A C C C A C A A A A C
## Levels: A B C D E
```
Using classification tree the accuracy is 0.4997961, the lowest so far accuracy

From the three  ML methods we used for this dataset, random forest method has the best accuracy and so it's the best method to use. Accuracy 0.996 and so the out-of-sample error rate is 0.004


