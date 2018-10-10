library(data.table)
library(tidyverse)
library(stringr)
library(car)
library(anchors)
library(GGally)
library(h2o)
require(MASS)
require(dplyr)
library(magrittr)
library(caret)

#directory path
path <- "/home/sanaz/Desktop/DataMining/projectlab"
#set working directory
setwd(path)
#load train and test file
train <- read.csv("train.csv")
test <- read.csv("test.csv")
train1<- train[0:100000,] 
test1<- test[0:100000,]

str(train1)
summary(train1)
head(train1)
#No duplicated row
duplicated(train1)
#to convert these -1's to NA
train1[train1== -1] <- NA
test1[test1==-1]<- NA
#Number of missing values
sum(is.na(train1)) #141891
sum(is.na(test1)) #142288

#Target Feature Analysis
#https://www.kaggle.com/captcalculator/a-very-extensive-porto-exploratory-analysis
#First let’s look at the target variable.
#How many positives (files claimed) are there?
library(ggplot2)
ggplot(data=train1, aes(x = as.factor(target))) +
  geom_bar(fill = '#84a5a3')+
  labs(title ='Distribution of Target')

c <- table(train1$target)
print(c)

#Missing Data 

data.frame(feature = names(train),
           miss_val = map_dbl(train, function(x) { sum(x == -1) / length(x) }))+
ggplot(aes(x= reorder(feature, -miss_val), y= miss_val))+
  geom_bar(stat = 'identity', colors='white', fill = '#5a64cd')+
  labs(x = '', y = '% missing', title = 'Missing Values by Feature') + 
  scale_y_continuous(labels = scales::percent)
library(Amelia)
library(mlbench)
missmap(train, col=c("black", "grey"), legend=FALSE)
###################################################################
library(corrplot)

cont_vars <- names(train1)[!grepl("_cat|_bin", names(train1))]

corrplot(cor(train1[, cont_vars][3:length(cont_vars)]), 
         type = 'lower', 
         col = colorRampPalette(c('#feeb8c', '#5a64cd'))(50),
         tl.col = 'grey40',
         mar = c(0,0,1,0),
         title = 'Correlation Matrix of Continuous Features')
#################################################################
cont_vars <- names(train)[!grepl("_cat|_bin", names(train))]
traind<- train[,cont_vars][3:length(cont_vars)]

#For ind features
ind_vars <- c('target', names(traind)[grepl('_ind_[0-9]{2}$', names(traind))])
train_ind<- train[,ind_vars]
for(i in 1:5){
  boxplot(train_ind[,4], main=names(train_ind)[4])
}
#For reg features
reg_vars <- c('target', names(traind)[grepl('_ind_[0-9]{2}$', names(traind))])
train_reg<- train[,reg_vars]

##############################################################
# Correlations:Get features names that are not binary or categorical(ind,car,reg,calc)
#https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html
library(corrplot)
cont_vars <- names(train1)[!grepl("_cat|_bin", names(train1))]
correlation <- cor(train1[,cont_vars][3:length(cont_vars)])
corrplot(correlation, method = "circle", title = 'Correlation Matrix of Continuous Features')
#Positive correlations are displayed in blue and negative correlations in red color
#Let’s break the correlations down by feature group:
#for ind:
ind_vars<- c('target', names(train1)[grepl('_ind_[0-9]{2}$', names(train1))])
correlation <- cor(train[,ind_vars])
corrplot.mixed(correlation, lower.col = "black", number.cex = .7)
#for reg:
reg_vars<- c('target', names(train)[grepl('_reg_[0-9]{2}$', names(train))])
correlation <- cor(train[,reg_vars])
corrplot.mixed(correlation, lower.col = "black", number.cex = .7)
#For car:
car_vars<- c('target', names(train)[grepl('_car_[0-9]{2}$', names(train))])
correlation <- cor(train[,car_vars])
corrplot.mixed(correlation, lower.col = "black", number.cex = .7)

#Categorical features
cat_vars <- names(train)[grepl("_cat|_bin", names(train))]
correlation <- cor(train[,cat_vars])
corrplot(correlation,order = "hclust", addrect = 3, col = heat.colors(100))
###########################################################DONE###############################
##############################################################################################
#####################################PCA################################################
#Add a column
test1$target <- 1
#combine the data set
comb <- rbind(train1, test1)



sapply(train, function(x) sum(is.na(x))) # missing values in each column
sapply(test, function(x) sum(is.na(x))) # missing values in each column
#impute missing values with median
comb$ps_ind_05_cat[is.na(comb$ps_ind_05_cat)]<- median(comb$ps_ind_05_cat, na.rm = TRUE)
comb$ps_ind_02_cat[is.na(comb$ps_ind_02_cat)]<- median(comb$ps_ind_02_cat, na.rm = TRUE)
comb$ps_ind_04_cat[is.na(comb$ps_ind_04_cat)]<- median(comb$ps_ind_04_cat, na.rm = TRUE)
comb$ps_car_01_cat[is.na(comb$ps_car_01_cat)]<- median(comb$ps_car_01_cat, na.rm = TRUE)
comb$ps_car_02_cat[is.na(comb$ps_car_02_cat)]<- median(comb$ps_car_02_cat, na.rm = TRUE)
comb$ps_car_03_cat[is.na(comb$ps_car_03_cat)]<- median(comb$ps_car_03_cat, na.rm = TRUE)
comb$ps_car_05_cat[is.na(comb$ps_car_05_cat)]<- median(comb$ps_car_05_cat, na.rm = TRUE)
comb$ps_car_07_cat[is.na(comb$ps_car_07_cat)]<- median(comb$ps_car_07_cat, na.rm = TRUE)
comb$ps_car_09_cat[is.na(comb$ps_car_09_cat)]<- median(comb$ps_car_09_cat, na.rm = TRUE)
comb$ps_reg_03[is.na(comb$ps_reg_03)]<- median(comb$ps_reg_03, na.rm = TRUE)
comb$ps_car_11 [is.na(comb$ps_car_11)]<- median(comb$ps_car_11, na.rm = TRUE)

comb$ps_car_12 [is.na(comb$ps_car_12)]<- median(comb$ps_car_12, na.rm = TRUE)
comb$ps_car_14 [is.na(comb$ps_car_14)]<- median(comb$ps_car_14, na.rm = TRUE)

Train <- comb[1:nrow(train1),]
Test <-  comb[1:nrow(-train1),]
sum(is.na(Train)) # 0
sum(is.na(Train))# 0


my_data <- subset(comb, select = -c(target,id))
str(my_data)
sum(is.na(my_data))#0  Now we don't have missing values



pca.train <- my_data[1:nrow(train1),]
pca.test <- my_data[-(1:nrow(train1)),]

#principal component analysis
prin_comp <- prcomp(pca.train, scale. = T)
names(prin_comp)  #"sdev"     "rotation" "center"   "scale"    "x"

#outputs the mean of variables
prin_comp$center
prin_comp$rotation

prin_comp$rotation[1:10,1:4]

dim(prin_comp$x)  #[1] 100000     57

#Let’s plot the resultant principal components:
biplot(prin_comp, scale = 0)
#compute standard deviation of each principal component
std_dev <- prin_comp$sdev
#compute variance
pr_var <- std_dev^2
#check variance of first 10 components
pr_var[1:10]
#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

#scree plot: A scree plot is used to access components
#or factors which explains the most of variability in the data.
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance",
     type = "b")
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance",
     type = "b")
#########Predictive Modeling with PCA Components##########
#add a training set with principal components
train.data <- data.frame(target = train1$target, prin_comp$x)
#we are interested in first 20 PCAs
train.data <- train.data[,1:21]
test.data <- predict(prin_comp, newdata = pca.test)
test.data <- as.data.frame(test.data)
test.data <- test.data[,1:20]
test.data$predicted<-NULL
gc()
#            used   (Mb) gc trigger   (Mb)  max used   (Mb)
#Ncells  48196522 2574.0   62761417 3351.9  49091534 2621.8
#Vcells 499957106 3814.4  825531125 6298.4 811662218 6192.5

model_log <- glm(as.factor(train1$target) ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + 
               PC7 + PC8 + PC9 + PC10 + PC11 + PC14 + PC15 + PC18 + PC19 + 
               PC20 ,family=binomial(link='logit'),data=train.data)

results <- predict(model_log,newdata=subset(test.data),type='response')
result <- data.frame(id = test1$id,target = results)
res <- data.frame(pred = results, actual=train.data$target)




#####################run a decision tree###################DONE!
library(rpart)
n<- nrow(train1)
Data <- comb[0:n,]
tr <- sort(sample(0:n,floor(n/2)))
Train_data <- train1[tr,]
Test_data <- train1[-tr,]
rp <- rpart(target ~ ., data=train1[2:59,], subset= tr, method = "class")
pred.rpt <- predict(rp, newdata = Test_data, type = "class")
pred.rpt
table(train1[-tr,]$target, pred.rpt)
# pred.rpt
      #0     1
#0 48175     0
#1  1826     0
#How many examples are well predicted?48175



