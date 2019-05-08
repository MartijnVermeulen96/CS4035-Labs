# for importing csv, or arff file formats
library(foreign)
# multiple imputation, finding likely values for missing data
library(mice)
# several machine learning algorithms
library(e1071)
# decision and regression trees
library(rpart)

# create random data
randomdata = data.frame(matrix(rnorm(100000),nrow=1000))
randomdata$X100 = randomdata$X100 > 0.0
table(randomdata$X100)
# learn decision tree model from random data
model = rpart(X100 ~ ., randomdata)
model
# we obtain a model, and it should be somewhat accurate on the training data
table(predict(model,randomdata) > 0.5,randomdata$X100)

# read the training data and test sets
data = read.csv("~/Downloads/adult.data.txt", header=FALSE, na.strings = " ?")
test =  read.csv("~/Downloads/adult.test.txt", header=FALSE, na.strings = " ?")

# split train and test data into male and female rows
maledata = data[data$V10 == " Male",]
femaledata = data[data$V10 == " Female",]
maletest = test[test$V10 == " Male",]
femaletest = test[test$V10 == " Female",]
# same for positive and negative classes
positivedata = data[data$V15 == " >50K",]
negativedata = data[data$V15 == " <=50K",]

# show discrimination, the probability of females having a positive class is 0.1094606, for males it is 0.3057366
table(maledata$V15) / nrow(maledata)
table(femaledata$V15) / nrow(femaledata)
# learn a naive bayes model
model = naiveBayes(V15 ~ ., data = data)
# make predictions and put into table, seperated by gender
# the resulting discrimination is slightly worse
table(predict(model, maletest)) / nrow(maletest)
table(predict(model, femaletest)) / nrow(femaletest)
# the naive Bayes model uses a discriminatory rule to compute its predictions
model$tables$V10

# we can predict gender form the other attributes (without the class label V15)
# this shows redlining can occur
model = naiveBayes(V10 ~ ., data[,1:14])
table(test$V10, predict(model, test[,1:14]))

# we remove gender from the data and make predictions
# there is still a lot of dicrimination 0.1653775 - 0.06345693
model = naiveBayes(V15 ~ ., data[,-10])
table(predict(model, maletest[,-10])) / nrow(maletest[,-10])
table(predict(model, femaletest[,-10])) / nrow(femaletest[,-10])
# Wife/Husband are very correlated to Male/Female
table(data$V8,data$V10)
# Also remove this attribute, discrimination remains, although slightly less, 0.1400552 - 0.0641948
model = naiveBayes(V15 ~ ., data[,c(-8,-10)])
table(predict(model, maletest[,c(-8,-10)])) / nrow(maletest[,c(-8,-10)])
table(predict(model, femaletest[,c(-8,-10)])) / nrow(femaletest[,c(-8,-10)])

# resample the data to remove discrimination in the training data
# this shows the numer of rows that should have a positive class for males and females
# in order to remove discrimination while keeping the overall positive class probability
(table(data$V15) / nrow(data)) * nrow(femaledata)
(table(data$V15) / nrow(data)) * nrow(maledata)
# split the data into gender-class sets
femalenegativedata = femaledata[femaledata$V15 == " <=50K",]
femalepositivedata = femaledata[femaledata$V15 == " >50K",]
# sample the required number of rows from positive and negative class females, bind them into one table
femalesample = rbind(femalepositivedata[sample(nrow(femalepositivedata),2594, replace=TRUE),], femalenegativedata[sample(nrow(femalenegativedata),8177, replace=TRUE),])
malenegativedata = maledata[maledata$V15 == " <=50K",]
malepositivedata = maledata[maledata$V15 == " >50K",]
# sample the required number of rows from positive and negative class males, bind them into one table
malesample = rbind(malepositivedata[sample(nrow(malepositivedata),5247, replace=TRUE),], malenegativedata[sample(nrow(malenegativedata),16543, replace=TRUE),])
# create a data set without discrimination
nondiscriminatorydata = rbind(malesample, femalesample)
# learn a model and show resulting discrimation values
model = naiveBayes(V15 ~ ., data = nondiscriminatorydata)
table(predict(model, maletest)) / nrow(maletest)
table(predict(model, femaletest)) / nrow(femaletest)
# although still some discrimination remains, the model does not use a discriminatory rule based on gender
model$tables$V10
# in order to completely remove the discrimination in the predictions, we would need to oversample the positive females and undersample the positive males, or use samples weights, oversampling the rows close to the decision boundary

# using different decision thresholds
# learn diferent models for different gender
malemodel = naiveBayes(V15 ~ ., data = maledata)
femalemodel = naiveBayes(V15 ~ ., data = femaledata)
# and one overall
model = naiveBayes(V15 ~ ., data = data)
# predicitons on the data set for the different models
table(predict(femalemodel,femaledata))
table(predict(femalemodel,femaledata)) / nrow(femaledata)
table(predict(malemodel,maledata))
table(predict(malemodel,maledata)) / nrow(maledata)
# positive class probability (aim for this in both classifiers):
table(data$V15) / nrow(data)
# or aim for the models:
table(c(predict(malemodel,maledata),predict(femalemodel,femaledata))) / (nrow(maledata) + nrow(femaledata))
# even better would be to find the setting that maximizes accuracy...
# we optimize thresholds on the data set instead of the test set
# they can be found by trial and error, trying all, or some binary search
table(predict(femalemodel,femaledata, type="raw")[,2] > 0.01) / nrow(femaledata)
table(predict(malemodel,maledata, type="raw")[,2] > 0.265) / nrow(maledata)
# no discrimination remains, in fact it is a little positive on the test set
table(predict(femalemodel,femaletest, type="raw")[,2] > 0.01) / nrow(femaletest)
table(predict(malemodel,maletest, type="raw")[,2] > 0.265) / nrow(maletest)
# we can do the same for a single model
table(predict(model,femaledata, type="raw")[,2] > 0.0021) / nrow(femaledata)
table(predict(model,maledata, type="raw")[,2] > 0.4) / nrow(maledata)
table(predict(model,femaletest, type="raw")[,2] > 0.0021) / nrow(femaletest)
table(predict(model,maletest, type="raw")[,2] > 0.4) / nrow(maletest)

# some discrimination might be OK, for instance, education level is correlated with gender:
table(maledata$V4) / nrow(maledata)
table(femaledata$V4) / nrow(femaledata)
# but given the same education level, there is still gender discrimination...
table(femaledata$V4, femaledata$V15) / c(table(femaledata$V4),table(femaledata$V4))
table(maledata$V4, maledata$V15) / c(table(maledata$V4),table(maledata$V4))
# there is also discrimination based on ethnicity
table(femaledata$V9, femaledata$V15) / c(table(femaledata$V9),table(femaledata$V9))
table(maledata$V9, maledata$V15) / c(table(maledata$V9),table(maledata$V9))
# how to incorporate OK discrimination, and multiple forms of discrimination, is difficult and topic of research

# false positive rates
# the two models actually apply too much positive discrimination under this measure!
# females have a false positive rate of 0.1813559, males of 0.4714373
table(predict(femalemodel,femaletest[femaletest$V15 == " >50K.",], type="raw")[,2] > 0.01) / nrow(femaletest[femaletest$V15 == " >50K.",])
table(predict(malemodel,maletest[maletest$V15 == " >50K.",], type="raw")[,2] > 0.265) / nrow(maletest[maletest$V15 == " >50K.",])
# for the original model female false positive probability is greater
table(predict(model,maletest[maletest$V15 == " >50K.",], type="raw")[,2] > 0.5) / nrow(maletest[maletest$V15 == " >50K.",])
table(predict(model,femaletest[femaletest$V15 == " >50K.",], type="raw")[,2] > 0.5) / nrow(femaletest[femaletest$V15 == " >50K.",])
# with corrected thresholds, it seems to discriminate males
table(predict(model,maletest[maletest$V15 == " >50K.",], type="raw")[,2] > 0.4) / nrow(maletest[maletest$V15 == " >50K.",])
table(predict(model,femaletest[femaletest$V15 == " >50K.",], type="raw")[,2] > 0.0021) / nrow(femaletest[femaletest$V15 == " >50K.",])
# to avoid this, the thresholds should have been optimized based on the false positive rate instead of positive class probability

# accuracies in crosstables:
table(predict(model,maletest, type="raw")[,2] > 0.5, maletest$V15)
# 8444 males are correctly classified by the original model
table(predict(model,femaletest, type="raw")[,2] > 0.5, femaletest$V15)
# 4893 females are correctly classified by the original model
table(predict(malemodel,maletest, type="raw")[,2] > 0.265, maletest$V15)
# 8575 males are correctly classified by the modified male model
table(predict(femalemodel,femaletest, type="raw")[,2] > 0.01, femaletest$V15)
# 4524 females are correctly classified by the modified female model
table(predict(model,maletest, type="raw")[,2] > 0.4, maletest$V15)
# 8490 males are correctly classified by the modified overall model
table(predict(model,femaletest, type="raw")[,2] >  0.0021, femaletest$V15)
# 4675 females are correctly classified by the modified overall model

# the modified models are slightly less accurate on females and slighty more accurate on males
# feel free to draw your own conclusions...

# test number of missing values
table(is.na(data))
# test missing data type, it is not missing at random
# the number of missing values for employer is correlated with gender!
table(is.na(data$V2), data$V10)
# call the imputation for a subset of the data (first 100 rows)
imp = mice(data[1:100,])
# com is the first imputation, by default 5 are generated
com = complete(imp, 1)
# test the missing values in com (should be none)
table(is.na(com))

# resample the dataset to obtain balanced classes
table(data$V15)
a = positivedata[sample(nrow(positivedata),1000),]
b = negativedata[sample(nrow(negativedata),1000),]
sam = rbind(a,b)
table(sam$V15)