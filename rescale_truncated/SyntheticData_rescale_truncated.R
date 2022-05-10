## Main function for the parameter analysis of ReBooT on the low-cross data with 12 dimensions of noise

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
ctype = 1  # rescale = 1-C/(m+C), the range of C is (0, +inf)
max_depth = 1
loss = 'l'
parallel = TRUE

c_seq = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100)
lc_seq = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 10000) # max_depth=1


source(str_c(pathtmp, "l1normComputation.R"))
source(str_c(pathtmp, "ada.modification.R"))
source(str_c(pathtmp, "ada.machine.modification.R"))
source(str_c(pathtmp, "rescale_truncated\\boosting_rescale_truncated.R"))

load(str_c(pathtmp, "data/", dataName, ".Rdata"))


t_num <- 20 # number of trails
control <- rpart.control(cp = -1 , maxdepth = max_depth , minsplit = 0) # choose stumps as weak learners


for(kkk in 1:length(lc_seq)){
  lc_tmp = lc_seq[kkk]
  savefile = str_c(pathtmp,"rescale_truncated\\results\\RT_", dataName, "_ParaAnalysis_iterMax", 
                   as.character(iterMax), "_maxdepth", as.character(max_depth),
                   "ctype", as.character(ctype) ,"_lc", as.character(lc_tmp), ".Rdata")
  
  tt <- 1 
  rescale_truncated_result <- as.list(seq(t_num))
  
  if (parallel == TRUE){
    no_cores <- detectCores()
    cl <- makeCluster(7)
    registerDoParallel(cl)
    rescale_truncated_result =foreach(num = seq(t_num), .packages =c('rpart','stringr')) %dopar% {
      source(str_c(pathtmp, "ada.modification.R"))
      source(str_c(pathtmp, "ada.machine.modification.R"))
      source(str_c(pathtmp, "rescale_truncated\\boosting_rescale_truncated.R"))
      source(str_c(pathtmp, "l1normComputation.R"))
      train_x <- dataGer[[num]]$train_x
      train_y <- dataGer[[num]]$train_y
      test_x <- dataGer[[num]]$test_x
      test_y <- dataGer[[num]]$test_y
      obj = boosting_rescale_truncated(train_x, train_y, test_x, test_y, ltype=1,
                                       iter_tmp = iterMax, c_seq = c_seq, lc_seq = lc_tmp, 
                                       loss=loss, ctype=ctype, control = control)
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
      
      rescale_truncated_result[[tt]] <- boosting_rescale_truncated(train_x, train_y, test_x, test_y, ltype=1,
                                                                   iter_tmp = iterMax, c_seq = c_seq, lc_seq = lc_tmp, 
                                                                   loss=loss, ctype=ctype, control = control)
      
      print(tt)
      tt <- tt+1
    }
  }
  save(rescale_truncated_result,file=savefile)
}
