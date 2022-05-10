## Main function for the parameter analysis of CatBoost on the low-cross data with 12 dimensions of noise

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

dataName = "data4"
iterations = 30000
parallel = TRUE

learning_rate_seq <- c(0.01, 0.05, 0.1, 0.2, 0.4, 0.8)
l2_leaf_reg_seq <- c(0.001, 0.01, 0.1, 1, 10)

thread_count = 1
max_depth = 1
loss_function = 'Logloss'
custom_loss = c('Accuracy') # it is recorded in the file named learn_error 
# custom_loss = c('Accuracy', 'Precision', 'Recall')
boosting_type = 'Ordered' 
logging_level	= 'Verbose'  # other choices can be 'Silent', 'Info'
verbose = 1000

source(str_c(pathtmp, "GeometricSeries.R"))
source(str_c(pathtmp, "catboost/boosting_cat_traindir.R"))

load(str_c(pathtmp, "data/", dataName, ".Rdata"))
train_dir_base = str_c(pathtmp,"catboost/results/CAT_", dataName, "_ParaAnalysisdir_iterMax",
                       as.character(iterations), "_maxdepth", as.character(max_depth))

savefile = str_c(pathtmp,"catboost/results/CAT_", dataName, "_ParaAnalysis_iterMax",
                 as.character(iterations), "_maxdepth", as.character(max_depth), ".Rdata")

t_num <- 20 # number of trails
tt <- 1 
catboost_result <- as.list(seq(t_num))
if (parallel == TRUE){
    no_cores <- detectCores()
    cl <- makeCluster(7)
    registerDoParallel(cl)
    catboost_result = foreach(num = seq(t_num), .packages =c('catboost','stringr')) %dopar% {
    source(str_c(pathtmp, "catboost/boosting_cat_traindir.R"))
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
    
    train_dir <- str_c(train_dir_base, 'Ex', as.character(num))
    
    obj <-  boosting_cat_traindir(train_x, train_y, test_x, test_y, iterations=iterations,
                                  learning_rate_seq=learning_rate_seq, 
                                  l2_leaf_reg_seq=l2_leaf_reg_seq,  
                                  boosting_type = boosting_type, max_depth = max_depth, 
                                  loss_function = loss_function, custom_loss = custom_loss, 
                                  thread_count = thread_count, logging_level = logging_level, 
                                  verbose = verbose, train_dir=train_dir)
    
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

    catboost_result[[tt]] <-  boosting_cat_traindir(train_x, train_y, test_x, test_y, iterations=iterations,
                                  learning_rate_seq=learning_rate_seq, 
                                  l2_leaf_reg_seq=l2_leaf_reg_seq,  
                                  boosting_type = boosting_type, max_depth = max_depth, 
                                  loss_function = loss_function, custom_loss = custom_loss, 
                                  thread_count = thread_count, logging_level = logging_level, 
                                  verbose = verbose, train_dir=train_dir_base)
    print(tt)
    tt <- tt+1
  }
}
save(catboost_result,file=savefile)
