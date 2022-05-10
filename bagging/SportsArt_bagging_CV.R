## Main function for the performance comparison of Bagging on the dataset Sports

rm(list=ls())
library(stringr)
library(rpart)
library(adabag)
library(lattice)
library(ggplot2)
library(caret)
library(parallel)
library(iterators)
library(foreach)
library(doParallel)

# set the root path of the code
script.dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
a <- unlist(strsplit(script.dir, split="/"))
pathtmp <- str_c(paste(a[-(length(a))], collapse="/"), "/") 

dataName = "SportsArt_Normalize_TrRatio80"
load(str_c(pathtmp, "data\\", dataName, ".Rdata"))
source(str_c(pathtmp, "bagging/ensemble_bagging.R"))
source(str_c(pathtmp, "bagging/CVensemble_bagging.R"))
source(str_c(pathtmp, "bagging/bagging.modification.R"))
source(str_c(pathtmp, "bagging/predict.bagging.modification.R"))
source(str_c(pathtmp, "bagging/select.R"))


iterMax = 30000
nfolds = 5
max_depth = 1
parallel = FALSE # if the RAM is large enough, it can be set as TRUE

file1 = str_c(pathtmp,"bagging\\results\\Bag_", dataName, "_CVSel_results_iterMax",
              as.character(iterMax), "_maxdepth", as.character(max_depth), ".Rdata")

t_num <- 20 # number of trails
tt <- 1 
control <- rpart.control(cp = -1 , maxdepth = max_depth , minsplit = 0) # choose stumps as weak learners
bag_CVresults <- list()
bag_results_IterOpt <- list()

test_err_IterOpt = rep(0,1,t_num)
train_err_IterOpt = rep(0,1,t_num)
IterOpt = rep(0,1,t_num)

test_err_IterOptAll = matrix(data=0,iterMax,t_num)
train_err_IterOptAll = matrix(data=0,iterMax,t_num)

while(tt<=t_num) # loops
{
  train_x <- dataGer[[tt]]$train_x
  train_y <- dataGer[[tt]]$train_y
  
  test_x <- dataGer[[tt]]$test_x
  test_y <- dataGer[[tt]]$test_y
  
  bag_CVresults[[tt]] <- CVensemble_bagging(train_x, train_y, iter=iterMax, control=control, 
                                            nfolds = nfolds,  parallel = parallel)

  # select the optimal number of iterations
  Iter = bag_CVresults[[tt]]$ParaCVSelIterAll$IterOpt
  IterOpt[tt] = Iter[1]
  
  
  bag_results_IterOpt[[tt]] <- ensemble_bagging(train_x, train_y, test_x, test_y, iter=iterMax, parallel=parallel, control)
  
  
  test_err_IterOptAll[,tt] = bag_results_IterOpt[[tt]]$test_err
  train_err_IterOptAll[,tt] = bag_results_IterOpt[[tt]]$train_err
  
  test_err_IterOpt[tt] = bag_results_IterOpt[[tt]]$test_err[IterOpt[tt]]
  train_err_IterOpt[tt] = bag_results_IterOpt[[tt]]$train_err[IterOpt[tt]]
  
  print(c(tt,test_err_IterOpt[tt]))
  tt <- tt+1
}
results = list(test_err_IterOpt = test_err_IterOpt, train_err_IterOpt = train_err_IterOpt,  IterOpt = IterOpt,
               test_err_IterOptAll = test_err_IterOptAll, train_err_IterOptAll = train_err_IterOptAll,
               MeanStd = list(teMeanStd_IterOpt = c(mean(test_err_IterOpt), sd(test_err_IterOpt)), 
                              trMeanStd_IterOpt = c(mean(train_err_IterOpt), sd(train_err_IterOpt)),
                              IterOptMeanStd = c(mean(IterOpt),sd(IterOpt)),
                              teMeanStd_IterOptAll = cbind(rowMeans(test_err_IterOptAll), apply(test_err_IterOptAll,1,sd)), 
                              trMeanStd_IterOptAll = cbind(rowMeans(train_err_IterOptAll), apply(train_err_IterOptAll,1,sd)) 
               ))

Bag_result_list = list(CVresults = bag_CVresults, results = results)

save(Bag_result_list, file=file1)
