library(dplyr)
#Change wd to match where your files are stored
## The data that Stefanie cleaned.
train <- read_csv("Downloads/Training.csv", 
                  col_types = cols(BranchID = col_character()))

train$BusinessDate<-as.Date(train$BusinessDate,format = "%m/%d/%Y")
train$BranchID<-as.character(train$BranchID)
train$TotalCashUsed<-as.numeric(train$TotalCashUsed)

Holidays <- read_csv("Downloads/crossroads-classic-datathon-2021/TCU_HolidayList.csv")
Holidays$HolidayDate<-as.Date(Holidays$HolidayDate,format = "%m/%d/%Y")

trainAll<-train%>%
  dplyr::select(-CashOrdersReceived,-CashBalance_StartOfDay)%>%
  mutate(DayofYear=format(BusinessDate, '%m-%d'),
         Year = format(BusinessDate, '%Y'),
         WeekDay = format(BusinessDate, '%A'),
         Month = format(BusinessDate, "%m"),
         Day = format(BusinessDate, "%d"),
         Holiday = ifelse(BusinessDate%in%Holidays$HolidayDate ,1,0),
         Missing = ifelse(is.na(TotalCashUsed),1,0),
         Covid = ifelse(BusinessDate>=as.Date("2020-03-20"),1,0)
  )

write.csv(trainAll, file = "trainAll.csv",row.names = FALSE)

trainAll$TotalCashUsed<-ifelse(is.na(trainAll$TotalCashUsed),0,trainAll$TotalCashUsed)

########################Optional Removing Missing
d<-trainAll%>%
  filter(Missing==0)

########################Optional Removing closed days
d<-d%>% #If you did not remove missing change to trainAll
  filter(WeekDay !="Saturday")%>%
  filter(BranchID!="126" | WeekDay != "Friday")%>%
  filter(BranchID!="319" | WeekDay != "Friday")%>%
  filter(BranchID!="458" | WeekDay != "Monday")%>%
  filter(BranchID!="458" | WeekDay != "Friday")

d<-d%>%
  dplyr::select(y=TotalCashUsed,BranchID,Year,WeekDay,Month,Day,Holiday,Covid)


library(caret)
#Create dummies
dummies <- dummyVars(y ~ ., data = d)            # create dummies for Xs
ex <- data.frame(predict(dummies, newdata = d))  # actually creates the dummies
d <- cbind(d$y, ex)                              # combine target var with Xs
names(d)[1] <- "y"                               # name target var 'y'
rm(dummies, ex) 

B99<-d$BranchID99
# Check linear combos ##Only removed one of each factor
y <- d$y
d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])
names(d)[1] <- "ones"
comboInfo <- findLinearCombos(d)
(remove<-names(d)[comboInfo$remove])
d <- d[, -comboInfo$remove]
d <- d[, c(2:ncol(d))]
d <- cbind(y, d)
rm(y, comboInfo)

library(caret)

#Partition Data
set.seed(1234)
inTrain <- createDataPartition(y = d$Covid,
                               p = .70,
                               list = F)
tr <- d[inTrain,]
te <- d[-inTrain,]

#################################Change train outliers
library(outliers)
#Standard Deviation
# scores<-scores(tr$y,type = "z")
# #sum(abs(scores)>3)
# newmax<-3*sd(tr$y)+mean(tr$y)
# newmin<- -3*sd(tr$y)+mean(tr$y)
# newtry<-ifelse(scores>3,newmax,tr$y)
# newtry<-ifelse(scores< -3,newmin,newtry)
# summary(tr$y)
# summary(newtry)
# # By branch
# branches<-unique(trainAll$BranchID)
# par(mfrow=c(1,1))
# newtry<-tr$y
# BranchID99<-as.logical(B99[inTrain])#[-trSat])
# for(i in branches){
#   nam<-paste0("BranchID",i)
#   print(nam)
#   if(i!= "99"){logicVect<-tr[,nam]==1}
#   else{logicVect<-BranchID99}
#   scores<-scores(tr$y[logicVect],type = "z")
#   newmax<-3*sd(tr$y[logicVect])+mean(tr$y[logicVect])
#   newmin<- -3*sd(tr$y[logicVect])+mean(tr$y[logicVect])
#   newtry<-ifelse(tr$y < newmin & logicVect,newmin,newtry)
#   newtry<-ifelse(tr$y > newmax & logicVect,newmax,newtry)
# }

# IQR
# box<-boxplot(tr$y)
# #length(box$out)
# min<-min(tr$y[!tr$y%in%box$out])
# max<-max(tr$y[!tr$y%in%box$out])
# newtry<-ifelse(tr$y< min,min,tr$y)
# newtry<-ifelse(tr$y>max,max,newtry)
# summary(tr$y)
# summary(newtry)
# boxplot(newtry)
#By branch
branches<-unique(tr$BranchID)
par(mfrow=c(1,1))
newtry<-tr$y
BranchID99<-as.logical(B99[inTrain])#[-trSat])
for(i in branches){
  nam<-paste0("BranchID",i)
  print(nam)
  if(i!= "99"){logicVect<-tr[,nam]==1}
  else{logicVect<-BranchID99}
  out<-boxplot(tr$y[logicVect])$out
  min<-min(tr$y[!tr$y%in%out & logicVect])
  max<-max(tr$y[!tr$y%in%out & logicVect])
  newtry<-ifelse(tr$y < min & logicVect,min,newtry)
  newtry<-ifelse(tr$y > max & logicVect,max,newtry)
}
par(mfrow=c(1,7)) # Make the Plots area really wide in RStudio
for(i in branches){
  nam<-paste0("BranchID",i)
  print(nam)
  if(i!= "469"){logicVect<-tr[,nam]==1}
  else{logicVect<-BranchID469}
  boxplot(newtry[logicVect])
}

##########################lightgbm
ytr<-tr$y
yte<-te$y
xtr<-as.matrix(tr%>%dplyr::select(-y))
xte<-as.matrix(te%>%dplyr::select(-y))
library(lightgbm)
lgbm <- lightgbm(data = xtr,label = newtry,
                 metric = "RMSE",
                 max_depth = 16,
                 num_iterations = 425,
                 learning_rate = .05,
                 num_leaves = 55,
                 objective = "regression",
                 eval_freq = 50)

l_tr <- predict(lgbm, data=xtr)
l_te <- predict(lgbm, data=xte)
l_teC<- ifelse(HolSat,0,l_te)
library(DescTools)
SMAPE(l_tr,newtry)*100
SMAPE(l_tr,ytr)*100
SMAPE(l_te,yte)*100
#mySMAPE(l_teC,yte)

p<-c(100,200,300)
saves<-as.data.frame(matrix(NA,nrow = length(p),ncol = 3))
for(i in 1:length(p)){
  s<-p[i]
  print(s)
  lgbm <- lightgbm(data = xtr,label = newtry,
                   metric = "RMSE",
                   max_depth = 16,
                   num_iterations = 425,
                   learning_rate = 0.05,
                   max_bin = s,
                   num_leaves = 55,
                   objective = "regression",
                   eval_freq = 50)

  l_tr <- predict(lgbm, data=xtr)
  l_te <- predict(lgbm, data=xte)
  #l_teC<- ifelse(HolSat,0,l_te)
  saves[i,]<-c(s,SMAPE(l_tr,ytr)*100,SMAPE(l_te,yte)*100
               #,mySMAPE(l_teC,yte)
               )
}
saves

#############Load model and get info from it.
lgbm<-lgb.load("lightgbm.model")

l_tr <- predict(lgbm, data=xtr)
l_te <- predict(lgbm, data=xte)
library(DescTools)
SMAPE(l_tr,newtry)*100
SMAPE(l_tr,ytr)*100
SMAPE(l_te,yte)*100


importance<-lgb.importance(lgbm)
tree_dt<-lgb.model.dt.tree(lgbm)

# d2<-trainAll%>%
#   filter(Missing==0)
# 
# d2<-d2%>% #If you did not remove missing change to trainAll
#   filter(WeekDay !="Saturday")%>%
#   filter(BranchID!="126" | WeekDay != "Friday")%>%
#   filter(BranchID!="319" | WeekDay != "Friday")%>%
#   filter(BranchID!="458" | WeekDay != "Monday")%>%
#   filter(BranchID!="458" | WeekDay != "Friday")

#d2<-d2%>%
#  dplyr::select(y=TotalCashUsed,BranchID,Year,WeekDay,Month,Day,Holiday,Covid)

d2 <- trainAll %>%
  rename(y = TotalCashUsed)

pred_df<- as.data.frame(cbind(yte, l_te))

pred_df <- pred_df %>%
  rename(y = yte, Predictions = l_te)

pred_df2 <- pred_df %>%
  group_by(y) %>% 
  filter(n()<=1) %>%
  select(y, Predictions)

pred_df3 <- pred_df2 %>%
  count(y)

df_combined_predictions <- pred_df2 %>%
  left_join(d2, by = "y")

df_combined_predictions2 <- df_combined_predictions %>%
  group_by(y) %>% 
  filter(n()<=1) 


df_combined_predictions <- df_combined_predictions %>% distinct()

write.csv(df_combined_predictions2,file = "PredictionDF.csv",row.names = FALSE)

write.csv(df_combined_predictions3,file = "PredictionDF_WithMissing.csv",row.names = FALSE)

########Setting up test set
testStart<-read.table("SubmissionInput.csv", header = T, sep = ",")
testStart$BusinessDate<-as.Date(testStart$BusinessDate,format = "%m/%d/%Y")
testStart$BranchID<-as.character(testStart$BranchID)
Date_Branch<-testStart$Date_Branch
test<-testStart%>%
  mutate(#Year = format(BusinessDate, '%Y'), #dummyVars doesn't like that they are all 2020
       WeekDay = format(BusinessDate, '%A'),
       Month = format(BusinessDate, "%m"),
       Day = format(BusinessDate, "%d"),
       Holiday = ifelse(BusinessDate%in%Holidays$HolidayDate ,1,0)
       ,Covid = ifelse(BusinessDate>=as.Date("2020-03-20"),1,0)
        )
names(test)
test<-test[,4:ncol(test)]
#I'll take care of the year later
test<-as.data.frame(cbind(Date_Branch,test))
names(test)[1]<-"y"
str(test)
write.csv(test,file = "SubmissionTest.csv",row.names = F)
test<-read.csv("SubmissionTest.csv",
               colClasses = c(rep("character",5),rep("numeric",2)
                              ))
str(test)
clusters<-read.csv("Clusters.csv",
                   colClasses = c("character","factor"))
test<-test%>%
  right_join(clusters,by = c("BranchID"))%>%
  filter(Cluster==1)%>%
  dplyr::select(-Cluster)
Date_Branch<-test$y
#Create dummies
dummies <- dummyVars(y ~ ., data = test)            # create dummies for Xs
ex <- data.frame(predict(dummies, newdata = test))  # actually creates the dummies
test <- cbind(test$y, ex)                              # combine target var with Xs
names(test)[1] <- "y"                               # name target var 'y'
rm(dummies, ex) 

test<-cbind(test,Year2010 = rep(0,nrow(test)),
            Year2011 = rep(0,nrow(test)),
            Year2012 = rep(0,nrow(test)),
            Year2013 = rep(0,nrow(test)),
            Year2014 = rep(0,nrow(test)),
            Year2015 = rep(0,nrow(test)),
            Year2016 = rep(0,nrow(test)),
            Year2017 = rep(0,nrow(test)),
            Year2018 = rep(0,nrow(test)),
            Year2019 = rep(0,nrow(test)),
            Month01 = rep(0,nrow(test))
            ,Month02 = rep(0,nrow(test))
            ,Month03 = rep(0,nrow(test))
            ,Month04 = rep(0,nrow(test))
            ,Month05 = rep(0,nrow(test))
            ,Month06 = rep(0,nrow(test))
            ,Month07 = rep(0,nrow(test))
            )


#remove from linear combos
Saturdays<-test$WeekDaySaturday

names<-dimnames(xtr)[[2]]
testm<-test[,names]#xgb$feature_names]
testm<-as.matrix(testm)
#predictions Change reg to the name of the model you want to use
pred<-predict(lgbm, data=testm)

output<-as.data.frame(cbind(Date_Branch,TotalCashUsed=pred))

# Change Saturdays to 0

output[as.logical(Saturdays),"TotalCashUsed"]#Just to see what the model did
output[,"TotalCashUsed"]<-ifelse(Saturdays==1,"0",output[,"TotalCashUsed"])

#Change Holidays to 0
 # Holiday<-test$Holiday
 # output[as.logical(Holiday),"TotalCashUsed"]#Just to see what the model did
 # output[,"TotalCashUsed"]<-ifelse(Holiday==1,0,output[,"TotalCashUsed"])

# Change other special Branch closures 
Friday126<-test$BranchID126*test$WeekDayFriday
output$TotalCashUsed[as.logical(Friday126)]#Just to see what the model did
output[,"TotalCashUsed"]<-ifelse(Friday126==1,0,output[,"TotalCashUsed"])

Friday319<-test$BranchID319*test$WeekDayFriday
output[as.logical(Friday319),"TotalCashUsed"]#Just to see what the model did
output[,"TotalCashUsed"]<-ifelse(Friday319==1,0,output[,"TotalCashUsed"])

Monday458<-test$BranchID458*test$WeekDayMonday
output[as.logical(Monday458),"TotalCashUsed"]#Just to see what the model did
output[,"TotalCashUsed"]<-ifelse(Monday458==1,0,output[,"TotalCashUsed"])

Friday458<-test$BranchID458*test$WeekDayFriday
output[as.logical(Friday458),"TotalCashUsed"]#Just to see what the model did
output[,"TotalCashUsed"]<-ifelse(Friday458==1,0,output[,"TotalCashUsed"])


# Might want to change the file name
write.csv(output,file = "SubmissionLGBM.csv",row.names = FALSE)
#In order for the submission to work, open up the file and 
# add two space after TotalCashUsed.
# so the title of the second column should be "TotalCashUsed "
