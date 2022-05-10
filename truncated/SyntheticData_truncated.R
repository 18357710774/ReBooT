## Main function for the parameter analysis of RTboosting on the low-cross data with 12 dimensions of noise

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

dataName = "data4"
iterMax = 30000
max_depth = 1
loss = 'l'
parallel = TRUE

c_seq = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80,
           90, 100, 500, 1000, 5000, 10000)

source(str_c(pathtmp, "l1normComputation.R"))
source(str_c(pathtmp, "ada.modification.R"))
source(str_c(pathtmp, "ada.machine.modification.R"))
source(str_c(pathtmp, "truncated\\boosting_truncated.R"))

load(str_c(pathtmp, "data\\", dataName, ".Rdata"))
savefile = str_c(pathtmp,"truncated\\results\\T_", dataName, "_ParaAnalysis_iterMax",
                 as.character(iterMax), "_maxdepth", as.character(max_depth), ".Rdata")


t_num <- 20 # number of trails
tt <- 1 
control <- rpart.control(cp = -1 , maxdepth=max_depth , minsplit = 0) # choose stumps as weak learners
truncated_result <- as.list(seq(t_num))

if (parallel == TRUE){
    no_cores <- detectCores()
    cl <- makeCluster(no_cores-2)
    registerDoParallel(cl)
    truncated_result =foreach(num = seq(t_num), .packages =c('rpart','stringr')) %dopar% {
    source(str_c(pathtmp, "ada.modification.R"))
    source(str_c(pathtmp, "ada.machine.modification.R"))
    source(str_c(pathtmp, "truncated\\boosting_truncated.R"))
    source(str_c(pathtmp, "l1normComputation.R"))
    train_x <- dataGer[[num]]$train_x
    train_y <- dataGer[[num]]$train_y
    test_x <- dataGer[[num]]$test_x
    test_y <- dataGer[[num]]$test_y
    
    obj = boosting_truncated(train_x, train_y, test_x, test_y, iter_tmp=iterMax, 
                             control = control, c_seq=c_seq, loss=loss)
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
    
    truncated_result[[tt]] <- boosting_truncated(train_x, train_y, test_x, test_y, iter_tmp=iterMax, 
                                                 control = control, c_seq=c_seq, loss=loss)
    print(tt)
    tt <- tt+1
  }
}
save(truncated_result,file=savefile)
