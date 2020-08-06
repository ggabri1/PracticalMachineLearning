ClassProject
================
GG
8/5/2020

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
(see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.
If you use the document you create for this class for any purpose please
cite them as they have been very generous in allowing their data to be
used for this kind of assignment.

## Executive Summary

The goal of this project is to predict the manner in which six
participants did the exercise. This is the “classe” variable in the
training set.

Six young health participants were asked to perform one set of 10
repetitions of the Unilateral Dumbbell Biceps Curl in five different
fashions: exactly according to the specification (Class A), throwing the
elbows to the front (Class B), lifting the dumbbell only halfway (Class
C), lowering the dumbbell only halfway (Class D) and throwing the hips
to the front (Class E). These are the categories for the “classe”
variable.

Class A corresponds to the specified execution of the exercise, while
the other 4 classes correspond to common mistakes. Participants were
supervised by an experienced weight lifter to make sure the execution
complied to the manner they were supposed to simulate. The exercises
were performed by six male participants aged between 20-28 years, with
little weight lifting experience. We made sure that all participants
could easily simulate the mistakes in a safe and controlled manner by
using a relatively light dumbbell (1.25kg).

## Main Analysis

### Importing and Cleaning the Data

``` r
##Importing the libraries for the analysis
library(utils)
library(caret)
library(randomForest)
library(ggplot2)
library(rpart)
```

``` r
fileURLtrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fileURLtest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

## Marking all missing data as NA in the data
training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!","")) 
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!","")) 

## Checking the dimentions of both datasets
dim(training)
```

    ## [1] 19622   160

``` r
dim(testing)
```

    ## [1]  20 160

``` r
## Next we remove the variables that have no recorded value (all missing)
training <- training[,colSums(is.na(training)) == 0]
testing <- testing[,colSums(is.na(testing)) == 0]

## We can also remove the variabes that are not necessary of our analysis
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

Before cleaning  
Training dataset has 19622 observations with 160 variables  
Testing dataset has 20 observations with 160 variables

``` r
## Checking the dimentions of both datasets after cleaning them
dim(training)
```

    ## [1] 19622    53

``` r
dim(testing)
```

    ## [1] 20 53

After cleaning  
Training dataset has 19622 observations with 53 variables  
Testing dataset has 20 observations with 53 variables

``` r
## Partitioning the training dataset with 70% for training and 30% testing 
inTrain <- createDataPartition(y=training$classe, p=0.70, list=FALSE)
training_train <- training[inTrain, ] 
training_test <- training[-inTrain, ]
dim(training_train)
```

    ## [1] 13737    53

``` r
dim(training_test)
```

    ## [1] 5885   53

I partition the training dataset with 70% for training and 30% for
testing.

Thus, we get 13737 observations for training, 5885 observations for
testing subsets.

### Checking the frequency of **classe** variable’s categories

As we mentioned earlier the variable consists of 5 categories and while
category A has the highest frequency the other categories appear at a
similar rate.

    ## 
    ##    A    B    C    D    E 
    ## 3906 2658 2396 2252 2525

## Predictions and Cross Validation

Random Forest and Decision Tree models are used for training purposes.

``` r
mod1 <- train(classe ~., method ="rpart", data = training_train)
        ## Decision Tree model
pred1 <- predict(mod1, training_test)
mod2 <- randomForest(classe ~., type = "class", data=training_train)
        ## Random forest model
pred2 <- predict(mod2, training_test)
```

Now we can check the accuracies of all the models.

``` r
## Model 1 - Decision tree 
confusionMatrix(pred1, training_test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1522  471  501  440  180
    ##          B   22  383   27  170  135
    ##          C  128  285  498  354  286
    ##          D    0    0    0    0    0
    ##          E    2    0    0    0  481
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4901          
    ##                  95% CI : (0.4772, 0.5029)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3327          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9092  0.33626  0.48538   0.0000  0.44455
    ## Specificity            0.6219  0.92541  0.78329   1.0000  0.99958
    ## Pos Pred Value         0.4888  0.51967  0.32108      NaN  0.99586
    ## Neg Pred Value         0.9451  0.85315  0.87817   0.8362  0.88874
    ## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
    ## Detection Rate         0.2586  0.06508  0.08462   0.0000  0.08173
    ## Detection Prevalence   0.5291  0.12523  0.26355   0.0000  0.08207
    ## Balanced Accuracy      0.7656  0.63084  0.63433   0.5000  0.72207

``` r
## Model 2 - Random Forest 
confusionMatrix(pred2, training_test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    5    0    0    0
    ##          B    2 1133    1    0    0
    ##          C    0    1 1025   11    0
    ##          D    0    0    0  953    6
    ##          E    0    0    0    0 1076
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9956          
    ##                  95% CI : (0.9935, 0.9971)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9944          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9947   0.9990   0.9886   0.9945
    ## Specificity            0.9988   0.9994   0.9975   0.9988   1.0000
    ## Pos Pred Value         0.9970   0.9974   0.9884   0.9937   1.0000
    ## Neg Pred Value         0.9995   0.9987   0.9998   0.9978   0.9988
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1925   0.1742   0.1619   0.1828
    ## Detection Prevalence   0.2850   0.1930   0.1762   0.1630   0.1828
    ## Balanced Accuracy      0.9988   0.9971   0.9983   0.9937   0.9972

As we can see the model using Random Forest method gives us the best
predictor with model accuracy of 0.9913. The Decision tree model is in
distant second place with the accuracy of 0.5593. Combining two
predictors in this case did not give a better predictor. It’s accuracy
is only 0.4757. The expected out of sample error is 0.6% (1-0.9942). As
a result, we will use the random forest model for our final prediction
on validation data set.

## Predicting the outcomes in the **testing** dataset

Now we can use your prediction model to predict 20 different test cases.

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
