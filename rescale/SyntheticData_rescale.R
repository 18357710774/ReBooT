## Main function for the parameter analysis of RBoosting on the low-cross data with 12 dimensions of noise

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
ctype = 1 # rescale = 1-C/(m+C), the range of C is (0, +inf)
c_seq_rescale = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,
                  0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000)

source(str_c(pathtmp, "l1normComputation.R"))
source(str_c(pathtmp, "ada.modification.R"))
source(str_c(pathtmp, "ada.machine.modification.R"))
source(str_c(pathtmp, "rescale\\boosting_rescale.R"))

load(str_c(pathtmp, "data/", dataName, ".Rdata"))
file = str_c(pathtmp,"rescale\\results\\R_", dataName, "_ParaAnalysis_iterMax",
             as.character(iterMax), "_maxdepth", as.character(max_depth), ".Rdata")

t_num <- 20 # number of trails
tt <- 1 
control <- rpart.control(cp = -1 , maxdepth = max_depth , minsplit = 0) # choose stumps as weak learners
rescale_result <- as.list(seq(t_num))

if (parallel == TRUE){
    no_cores <- detectCores()
    cl <- makeCluster(no_cores-2)
    registerDoParallel(cl)
    rescale_result =foreach(num = seq(t_num), .packages =c('rpart','stringr')) %dopar% {
    source(str_c(pathtmp, "ada.modification.R"))
    source(str_c(pathtmp, "ada.machine.modification.R"))
    source(str_c(pathtmp, "rescale\\boosting_rescale.R"))
    source(str_c(pathtmp, "l1normComputation.R"))
    train_x <- dataGer[[num]]$train_x
    train_y <- dataGer[[num]]$train_y
    test_x <- dataGer[[num]]$test_x
    test_y <- dataGer[[num]]$test_y
    obj = boosting_rescale(train_x, train_y, test_x, test_y, iter_tmp=iterMax, 
                           c_seq=c_seq_rescale, loss=loss, ctype=ctype, control=control)
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
    rescale_result[[tt]] <- boosting_rescale(train_x, train_y, test_x, test_y, iter_tmp=iterMax, 
                                             c_seq=c_seq_rescale, loss=loss, ctype=ctype, control=control)
    print(tt)
    tt <- tt+1
  }
}
save(rescale_result,file=file)
