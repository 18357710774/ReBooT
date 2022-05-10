## Main function for the performance comparison of CatBoost on the dataset GCredit
## Note that one should install "catboost" first

rm(list=ls())
library(stringr)
library(caret)
library(foreach)
library(parallel)
library(doParallel)
library(catboost)
library(methods)

# set the root path of the code
script.dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
a <- unlist(strsplit(script.dir, split="/"))
pathtmp <- str_c(paste(a[-(length(a))], collapse="/"), "/") 

dataName = "German_Normalize_TrRatio80"
load(str_c(pathtmp, "data\\", dataName, ".Rdata"))
source(str_c(pathtmp, "GeometricSeries.R"))
source(str_c(pathtmp, "catboost\\CVboosting_cat_traindir.R"))
source(str_c(pathtmp, "catboost\\boosting_cat_traindir.R"))

# parameter setting
learning_rate_seq <- GeometricSeries(0.01, 1, 21)
l2_leaf_reg_seq <- c(0.001, 0.01, 0.1, 1, 10)
iterations = 30000

nfolds = 5
thread_count = 1
max_depth = 1
loss_function = 'Logloss'
custom_loss = c('Accuracy') # it is recorded in the file named learn_error 
# custom_loss = c('Accuracy', 'Precision', 'Recall')
boosting_type = 'Ordered' 
logging_level	= 'Verbose'  # other choices can be 'Silent', 'Info'
verbose = 1000
parallel=TRUE

train_dir_base = str_c(pathtmp,"catboost\\results\\CAT_", dataName, "_CVSeldir_iterMax",
                  as.character(iterations), "_maxdepth", as.character(max_depth))


file1 = str_c(pathtmp,"catboost\\results\\CAT_", dataName, "_CVSeldir_results_iterMax",
              as.character(iterations), "_maxdepth", as.character(max_depth), ".Rdata")

t_num <- 20 # number of trails
tt <- 1 
cat_CVresults <- list()
cat_results_IterOpt <- list()

test_err_IterOpt = rep(0,1,t_num)
train_err_IterOpt = rep(0,1,t_num)
alpha_IterOpt = list()
l1norm_IterOpt = rep(0,1,t_num)
learning_rate_IterOpt = rep(0,1,t_num)
l2_leaf_reg_IterOpt = rep(0,1,t_num)
IterOpt = rep(0,1,t_num)

test_err_IterOptAll = matrix(data=0,iterations,t_num)
train_err_IterOptAll = matrix(data=0,iterations,t_num)
alpha_IterOptAll = matrix(data=0,iterations,t_num)
l1norm_IterOptAll = matrix(data=0,iterations,t_num)
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
  
  cat_CVresults[[tt]] <- CVboosting_cat_traindir(train_x, train_y, iterations = iterations, learning_rate_seq=learning_rate_seq,
                                        l2_leaf_reg_seq=l2_leaf_reg_seq, boosting_type=boosting_type, max_depth=max_depth,
                                        loss_function=loss_function, custom_loss=custom_loss, nfolds=nfolds,
                                        thread_count=thread_count, logging_level=logging_level,
                                        verbose=verbose, parallel=parallel, train_dir=train_dir_base)
  

  # select the optimal values of l2_leaf_reg, learning_rate and the number of iterations
  l2_leaf_reg = cat_CVresults[[tt]]$ParaCVSelIterAll$l2_leaf_reg
  learning_rate = cat_CVresults[[tt]]$ParaCVSelIterAll$learning_rate
  Iter = cat_CVresults[[tt]]$ParaCVSelIterAll$IterOpt
  l2_leaf_reg_IterOpt[tt] = l2_leaf_reg[1]
  learning_rate_IterOpt[tt] = learning_rate[1]
  IterOpt[tt] = Iter[1]

  
  cat_results_IterOpt[[tt]] <- boosting_cat_traindir(train_x, train_y, test_x, test_y, iterations=iterations,
                                            learning_rate_seq=learning_rate_IterOpt[tt], 
                                            l2_leaf_reg_seq=l2_leaf_reg_IterOpt[tt],  
                                            boosting_type = boosting_type, max_depth = max_depth, 
                                            loss_function = loss_function, custom_loss = custom_loss, 
                                            thread_count = thread_count, logging_level = logging_level, 
                                            verbose = verbose, train_dir=train_dir_base)


  test_err_IterOptAll[,tt] = cat_results_IterOpt[[tt]]$test_err
  train_err_IterOptAll[,tt] = cat_results_IterOpt[[tt]]$train_err
  alpha_IterOptAll[,tt] = cat_results_IterOpt[[tt]]$alpha
  l1norm_IterOptAll[,tt] = cat_results_IterOpt[[tt]]$l1norm
  
  test_err_IterOpt[tt] = cat_results_IterOpt[[tt]]$test_err[IterOpt[tt]]
  train_err_IterOpt[tt] = cat_results_IterOpt[[tt]]$train_err[IterOpt[tt]]
  alpha_IterOpt[[tt]] = cat_results_IterOpt[[tt]]$alpha[1:IterOpt[tt]]
  l1norm_IterOpt[tt] = cat_results_IterOpt[[tt]]$l1norm[IterOpt[tt]]
 
  print(c(tt,test_err_IterOpt[tt]))
  tt <- tt+1
}
results = list(test_err_IterOpt = test_err_IterOpt, train_err_IterOpt = train_err_IterOpt, alpha_IterOpt = alpha_IterOpt,
               l1norm_IterOpt = l1norm_IterOpt, l2_leaf_reg_IterOpt= l2_leaf_reg_IterOpt, learning_rate_IterOpt= learning_rate_IterOpt, IterOpt = IterOpt,
               test_err_IterOptAll = test_err_IterOptAll, train_err_IterOptAll = train_err_IterOptAll,
               alpha_IterOptAll = alpha_IterOptAll, l1norm_IterOptAll = l1norm_IterOptAll,
               MeanStd = list(teMeanStd_IterOpt = c(mean(test_err_IterOpt), sd(test_err_IterOpt)),
                              trMeanStd_IterOpt = c(mean(train_err_IterOpt), sd(train_err_IterOpt)),
                              l1normMeanStd_IterOpt = c(mean(l1norm_IterOpt),sd(l1norm_IterOpt)),
                              l2_leaf_regMeanStd_IterOpt = c(mean(l2_leaf_reg_IterOpt),sd(l2_leaf_reg_IterOpt)),
                              learning_rateMeanStd_IterOpt = c(mean(learning_rate_IterOpt),sd(learning_rate_IterOpt)),
                              IterOptMeanStd = c(mean(IterOpt),sd(IterOpt)),
                              teMeanStd_IterOptAll = cbind(rowMeans(test_err_IterOptAll), apply(test_err_IterOptAll,1,sd)),
                              trMeanStd_IterOptAll = cbind(rowMeans(train_err_IterOptAll), apply(train_err_IterOptAll,1,sd)),
                              l1normMeanStd_IterOptAll = cbind(rowMeans(l1norm_IterOptAll), apply(l1norm_IterOptAll,1,sd))
               ))

cat_result_list = list(CVresults = cat_CVresults, results = results)

save(cat_result_list,file=file1)