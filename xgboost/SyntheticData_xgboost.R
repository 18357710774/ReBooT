## Main function for the parameter analysis of XGBoost on the low-cross data with 12 dimensions of noise
## Note that one should install "xgboost" first

rm(list=ls())
library(stringr)
library(caret)
library(foreach)
library(parallel)
library(doParallel)
library(xgboost)
library(methods)

# set the root path of the code
script.dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
a <- unlist(strsplit(script.dir, split="/"))
pathtmp <- str_c(paste(a[-(length(a))], collapse="/"), "/") 

dataName = "data4"
nrounds = 30000
parallel = TRUE
eta_seq <- c(0.01, 0.05, 0.1, 0.2, 0.4, 0.8)
lambda_seq <- c(0.001, 0.01, 0.1, 1, 10)

nthread = 1
max_depth = 1
objective = 'binary:logistic'
booster = 'gbtree'
eval_metric = 'error'
print_every_n = 1000
verbose = 1

source(str_c(pathtmp, "GeometricSeries.R"))
source(str_c(pathtmp, "xgboost/boosting_xgb.R"))

load(str_c(pathtmp, "data/", dataName, ".Rdata"))
savefile = str_c(pathtmp,"xgboost/results/XGB_", dataName, "_ParaAnalysis_iterMax",
                 as.character(nrounds), "_maxdepth", as.character(max_depth), ".Rdata")

t_num <- 20 # number of trails
tt <- 1 
xgboost_result <- as.list(seq(t_num))
if (parallel == TRUE){
    no_cores <- detectCores()
    cl <- makeCluster(5)
    registerDoParallel(cl)
    xgboost_result =foreach(num = seq(t_num), .packages =c('xgboost','stringr')) %dopar% {
    source(str_c(pathtmp, "xgboost/boosting_xgb.R"))
    train_x <- dataGer[[num]]$train_x
    train_y <- dataGer[[num]]$train_y
    test_x <- dataGer[[num]]$test_x
    test_y <- dataGer[[num]]$test_y
    
    colnames(train_x) <- paste('x', 1:ncol(train_x), sep = '')
    colnames(test_x) <- paste('x', 1:ncol(test_x), sep = '')
    
    train_y <- c(0,1)[as.numeric(as.factor(train_y))]
    test_y <- c(0,1)[as.numeric(as.factor(test_y))]
    randIndTr = sample(1:length(train_y), length(train_y))
    randIndTe = sample(1:length(test_y), length(test_y))
    train_x <- train_x[randIndTr,]
    train_y <- train_y[randIndTr]
    test_x <- test_x[randIndTe,]
    test_y <- test_y[randIndTe]
    
    obj <-  boosting_xgb(train_x, train_y, test_x, test_y, nrounds=nrounds,
                                          eta_seq=eta_seq, lambda_seq=lambda_seq,
                                          max_depth = max_depth, objective = objective, booster=booster,
                                          eval_metric = eval_metric,  nthread = nthread, 
                                          print_every_n = print_every_n, verbose=verbose)
  }
  stopCluster(cl) 
}else
  {
  while(tt<=t_num) # loops
  {
    train_x <- dataGer[[tt]]$train_x
    train_y <- dataGer[[tt]]$train_y
    test_x <- dataGer[[tt]]$test_x
    test_y <- dataGer[[tt]]$test_y
    
    colnames(train_x) <- paste('x', 1:ncol(train_x), sep = '')
    colnames(test_x) <- paste('x', 1:ncol(test_x), sep = '')
    
    train_y <- c(0,1)[as.numeric(as.factor(train_y))]
    test_y <- c(0,1)[as.numeric(as.factor(test_y))]
    randIndTr = sample(1:length(train_y), length(train_y))
    randIndTe = sample(1:length(test_y), length(test_y))
    train_x <- train_x[randIndTr,]
    train_y <- train_y[randIndTr]
    test_x <- test_x[randIndTe,]
    test_y <- test_y[randIndTe]
    
    xgboost_result[[tt]] <-  boosting_xgb(train_x, train_y, test_x, test_y, nrounds=nrounds,
                                            eta_seq=eta_seq, lambda_seq=lambda_seq,
                                            max_depth = max_depth, objective = objective, booster=booster,
                                            eval_metric = eval_metric,  nthread = nthread, 
                                            print_every_n = print_every_n, verbose=verbose)
    print(tt)
    tt <- tt+1
  }
}
save(xgboost_result,file=savefile)
