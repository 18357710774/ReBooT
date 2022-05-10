## Main function for the performance comparison of ReBooT on the high-cross data with changing number of noise dimensions

rm(list=ls())
library(stringr)
library(rpart)
library(ada)
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

source(str_c(pathtmp, "l1normComputation.R"))
source(str_c(pathtmp, "ada.modification.R"))
source(str_c(pathtmp, "ada.machine.modification.R"))
source(str_c(pathtmp, "GeometricSeries.R"))
source(str_c(pathtmp, "cvada.modification.R"))
source(str_c(pathtmp, "cvada.machine.modification.R"))
source(str_c(pathtmp, "rescale_truncated/CVboosting_rescale_truncated.R")) # truncated boosting
source(str_c(pathtmp, "rescale_truncated/boosting_rescale_truncated.R"))

# parameter setting
iterMax = 10000
c_seq = c(5) # fix C1=5
lc_seq = GeometricSeries(0.1, 50, 21)
nfolds = 5
ctype = 1  # rescale = 1-C/(m+C), the range of C is (0, +inf)
max_depth = 1
loss = 'l'
parallel = TRUE

noise_feature_Cross = c(6, 8, 10, 12, 14, 16, 18, 22, 26, 30)

for (ii in 1:length(noise_feature_Cross)){
  noise_feature = noise_feature_Cross[ii]
  
  dataName = str_c("SynDataYaoTrN200NoiDim", as.character(noise_feature))
  load(str_c(pathtmp, 'data/', dataName, ".Rdata"))
  
  file1 = str_c(pathtmp,"rescale_truncated/results/RT_", dataName, "_CVSel_results_iterMax",
                as.character(iterMax), "C", as.character(c_seq), "_maxdepth", as.character(max_depth), ".Rdata")
  
  t_num <- 20 # number of trails
  tt <- 1 
  control <- rpart.control(cp = -1 , maxdepth = max_depth , minsplit = 0) # choose stumps as weak learners
  rescale_truncated_CVresults <- list()
  rescale_truncated_results_IterOpt <- list()
  
  test_err_IterOpt = rep(0,1,t_num)
  train_err_IterOpt = rep(0,1,t_num)
  alpha_IterOpt = list()
  l1norm_IterOpt = rep(0,1,t_num)
  c_IterOpt = rep(0,1,t_num)
  lc_IterOpt = rep(0,1,t_num)
  IterOpt = rep(0,1,t_num)

  test_err_IterOptAll = matrix(data=0,iterMax,t_num)
  train_err_IterOptAll = matrix(data=0,iterMax,t_num)
  alpha_IterOptAll = matrix(data=0,iterMax,t_num)
  l1norm_IterOptAll = matrix(data=0,iterMax,t_num)
  
  while(tt<=t_num) # loops
  {
    train_x <- dataGer[[tt]]$train_x
    train_y <- dataGer[[tt]]$train_y

    test_x <- dataGer[[tt]]$test_x
    test_y <- dataGer[[tt]]$test_y
  
    rescale_truncated_CVresults[[tt]] <- CVboosting_rescale_truncated(train_x, train_y, iter_tmp = iterMax, c_seq=c_seq, loss=loss, ctype=ctype, 
                                            lc_seq=lc_seq, ltype=1,  control=control, nfolds = nfolds, parallel = parallel, pathtmp = pathtmp)
    
    # select the optimal values of C2 and the number of iterations
    c = rescale_truncated_CVresults[[tt]]$ParaCVSelIterAll$c
    lc = rescale_truncated_CVresults[[tt]]$ParaCVSelIterAll$lc
    Iter = rescale_truncated_CVresults[[tt]]$ParaCVSelIterAll$IterOpt
    c_IterOpt[tt] = c[1]
    lc_IterOpt[tt] = lc[1]
    IterOpt[tt] = Iter[1]
    
    
    rescale_truncated_results_IterOpt[[tt]] <- boosting_rescale_truncated(train_x, train_y, test_x, test_y, ltype=1,
                                                   iter_tmp=iterMax,c_seq=c_IterOpt[tt], lc_seq = lc_IterOpt[tt], 
                                                   loss=loss, ctype=ctype, control = control)
    
    test_err_IterOptAll[,tt] = rescale_truncated_results_IterOpt[[tt]]$test_err
    train_err_IterOptAll[,tt] = rescale_truncated_results_IterOpt[[tt]]$train_err
    alpha_IterOptAll[,tt] = rescale_truncated_results_IterOpt[[tt]]$alpha
    l1norm_IterOptAll[,tt] = rescale_truncated_results_IterOpt[[tt]]$l1norm
    
    
    test_err_IterOpt[tt] = rescale_truncated_results_IterOpt[[tt]]$test_err[IterOpt[tt]]
    train_err_IterOpt[tt] = rescale_truncated_results_IterOpt[[tt]]$train_err[IterOpt[tt]]
    alpha_IterOpt[[tt]] = rescale_truncated_results_IterOpt[[tt]]$alpha[1:IterOpt[tt]]
    l1norm_IterOpt[tt] = rescale_truncated_results_IterOpt[[tt]]$l1norm[IterOpt[tt]]
    
    print(c(tt,test_err_IterOpt[tt]))
    tt <- tt+1
  }
  
  results = list(test_err_IterOpt = test_err_IterOpt, train_err_IterOpt = train_err_IterOpt, alpha_IterOpt = alpha_IterOpt, 
                 l1norm_IterOpt = l1norm_IterOpt, c_IterOpt= c_IterOpt, lc_IterOpt= lc_IterOpt, IterOpt = IterOpt, 
                 test_err_IterOptAll = test_err_IterOptAll, train_err_IterOptAll = train_err_IterOptAll,
                 alpha_IterOptAll = alpha_IterOptAll, l1norm_IterOptAll = l1norm_IterOptAll,
                 MeanStd = list(teMeanStd_IterOpt = c(mean(test_err_IterOpt), sd(test_err_IterOpt)), 
                                trMeanStd_IterOpt = c(mean(train_err_IterOpt), sd(train_err_IterOpt)),
                                l1normMeanStd_IterOpt = c(mean(l1norm_IterOpt),sd(l1norm_IterOpt)),
                                cMeanStd_IterOpt = c(mean(c_IterOpt),sd(c_IterOpt)),
                                lcMeanStd_IterOpt = c(mean(lc_IterOpt),sd(lc_IterOpt)),
                                IterOptMeanStd = c(mean(IterOpt),sd(IterOpt)),
                                teMeanStd_IterOptAll = cbind(rowMeans(test_err_IterOptAll), apply(test_err_IterOptAll,1,sd)), 
                                trMeanStd_IterOptAll = cbind(rowMeans(train_err_IterOptAll), apply(train_err_IterOptAll,1,sd)), 
                                l1normMeanStd_IterOptAll = cbind(rowMeans(l1norm_IterOptAll), apply(l1norm_IterOptAll,1,sd))
                 ))
  
  
  RT_result_list = list(CVresults = rescale_truncated_CVresults, results = results)
  
  save(RT_result_list,file=file1)
}
