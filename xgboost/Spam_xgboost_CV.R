## Main function for the performance comparison of XGBoost on the dataset Spam
## Note that one should install "xgboost" first

rm(list=ls())
library(stringr)
library(xgboost)
library(methods)

# set the root path of the code
script.dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
a <- unlist(strsplit(script.dir, split="/"))
pathtmp <- str_c(paste(a[-(length(a))], collapse="/"), "/") 

dataName = "Spam_Normalize_TrRatio80"
load(str_c(pathtmp, "data\\", dataName, ".Rdata"))
source(str_c(pathtmp, "GeometricSeries.R"))
source(str_c(pathtmp, "xgboost\\CVboosting_xgb.R"))
source(str_c(pathtmp, "xgboost\\boosting_xgb.R"))

# parameter setting
eta_seq <- GeometricSeries(0.01, 1, 21)
lambda_seq <- c(0.001, 0.01, 0.1, 1, 10)
nrounds = 30000

nfold = 5
nthread = 1
max_depth = 1
objective = 'binary:logistic'
booster = 'gbtree'
eval_metric = 'error'
print_every_n = 1000
verbose = 1


file1 = str_c(pathtmp,"xgboost\\results\\XGB_", dataName, "_CVSel_results_iterMax",
              as.character(nrounds), "_maxdepth", as.character(max_depth), ".Rdata")

t_num <- 20 # number of trails
tt <- 1 
xgb_CVresults <- list()
xgb_results_IterOpt <- list()

test_err_IterOpt = rep(0,1,t_num)
train_err_IterOpt = rep(0,1,t_num)
alpha_IterOpt = list()
l1norm_IterOpt = rep(0,1,t_num)
eta_IterOpt = rep(0,1,t_num)
lambda_IterOpt = rep(0,1,t_num)
IterOpt = rep(0,1,t_num)

test_err_IterOptAll = matrix(data=0,nrounds,t_num)
train_err_IterOptAll = matrix(data=0,nrounds,t_num)
alpha_IterOptAll = matrix(data=0,nrounds,t_num)
l1norm_IterOptAll = matrix(data=0,nrounds,t_num)

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
  
  xgb_CVresults[[tt]] <- CVboosting_xgb(train_x, train_y, nrounds = nrounds, eta_seq=eta_seq, lambda_seq=lambda_seq,
                                        max_depth = max_depth, objective = objective, booster = booster, 
                                        nfold = nfold, nthread = nthread, print_every_n = print_every_n, 
                                        verbose=verbose)
  

  # select the optimal values of lambda, eta and the number of iterations
  lambda = xgb_CVresults[[tt]]$ParaCVSelIterAll$lambda
  eta = xgb_CVresults[[tt]]$ParaCVSelIterAll$eta
  Iter = xgb_CVresults[[tt]]$ParaCVSelIterAll$IterOpt
  lambda_IterOpt[tt] = lambda[1]
  eta_IterOpt[tt] = eta[1]
  IterOpt[tt] = Iter[1]


  xgb_results_IterOpt[[tt]] <- boosting_xgb(train_x, train_y, test_x, test_y, nrounds=nrounds,
                                            eta_seq=eta_IterOpt[tt], lambda_seq=lambda_IterOpt[tt],
                                            max_depth = max_depth, objective = objective, booster=booster,
                                            eval_metric = eval_metric,  nthread = nthread, 
                                            print_every_n = print_every_n, verbose=verbose)
                                                    
  
  test_err_IterOptAll[,tt] = xgb_results_IterOpt[[tt]]$test_err
  train_err_IterOptAll[,tt] = xgb_results_IterOpt[[tt]]$train_err
  alpha_IterOptAll[,tt] = xgb_results_IterOpt[[tt]]$alpha
  l1norm_IterOptAll[,tt] = xgb_results_IterOpt[[tt]]$l1norm
  
  test_err_IterOpt[tt] = xgb_results_IterOpt[[tt]]$test_err[IterOpt[tt]]
  train_err_IterOpt[tt] = xgb_results_IterOpt[[tt]]$train_err[IterOpt[tt]]
  alpha_IterOpt[[tt]] = xgb_results_IterOpt[[tt]]$alpha[1:IterOpt[tt]]
  l1norm_IterOpt[tt] = xgb_results_IterOpt[[tt]]$l1norm[IterOpt[tt]]
 
  print(c(tt,test_err_IterOpt[tt]))
  tt <- tt+1
}
results = list(test_err_IterOpt = test_err_IterOpt, train_err_IterOpt = train_err_IterOpt, alpha_IterOpt = alpha_IterOpt,
               l1norm_IterOpt = l1norm_IterOpt, lambda_IterOpt= lambda_IterOpt, eta_IterOpt= eta_IterOpt, IterOpt = IterOpt,
               test_err_IterOptAll = test_err_IterOptAll, train_err_IterOptAll = train_err_IterOptAll,
               alpha_IterOptAll = alpha_IterOptAll, l1norm_IterOptAll = l1norm_IterOptAll,
               MeanStd = list(teMeanStd_IterOpt = c(mean(test_err_IterOpt), sd(test_err_IterOpt)),
                              trMeanStd_IterOpt = c(mean(train_err_IterOpt), sd(train_err_IterOpt)),
                              l1normMeanStd_IterOpt = c(mean(l1norm_IterOpt),sd(l1norm_IterOpt)),
                              lambdaMeanStd_IterOpt = c(mean(lambda_IterOpt),sd(lambda_IterOpt)),
                              etaMeanStd_IterOpt = c(mean(eta_IterOpt),sd(eta_IterOpt)),
                              IterOptMeanStd = c(mean(IterOpt),sd(IterOpt)),
                              teMeanStd_IterOptAll = cbind(rowMeans(test_err_IterOptAll), apply(test_err_IterOptAll,1,sd)),
                              trMeanStd_IterOptAll = cbind(rowMeans(train_err_IterOptAll), apply(train_err_IterOptAll,1,sd)),
                              l1normMeanStd_IterOptAll = cbind(rowMeans(l1norm_IterOptAll), apply(l1norm_IterOptAll,1,sd))
               ))

XGB_result_list = list(CVresults = xgb_CVresults, results = results)

save(XGB_result_list,file=file1)
