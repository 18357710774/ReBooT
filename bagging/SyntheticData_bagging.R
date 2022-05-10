## Main function for the parameter analysis of Bagging on the low-cross data with 12 dimensions of noise

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

dataName = "data4"
iterMax = 30000
parallel = FALSE  # if the RAM is large enough, it can be set as TRUE

max_depth = 1

source(str_c(pathtmp, "bagging/ensemble_bagging.R"))
source(str_c(pathtmp, "bagging/bagging.modification.R"))
source(str_c(pathtmp, "bagging/predict.bagging.modification.R"))
source(str_c(pathtmp, "bagging/select.R"))

load(str_c(pathtmp, "data/", dataName, ".Rdata"))
savefile = str_c(pathtmp,"bagging/results/Bag_", dataName, "_ParaAnalysis_iterMax", 
                 as.character(iterMax), "_maxdepth", as.character(max_depth), ".Rdata")


t_num <- 20 # number of trails
tt <- 1 
control <- rpart.control(cp = -1 , maxdepth = max_depth , minsplit = 0) # choose stumps as weak learners
bag_result <- as.list(seq(t_num))

while(tt<=t_num) # loops
{
  train_x <- dataGer[[tt]]$train_x
  train_y <- dataGer[[tt]]$train_y
  test_x <- dataGer[[tt]]$test_x
  test_y <- dataGer[[tt]]$test_y
  
  bag_result[[tt]] <- ensemble_bagging(train_x, train_y, test_x, test_y, iter=iterMax, parallel=parallel, control)
  
  print(str_c("Ex #  ", as.character(tt)))
  tt <- tt+1
}
save(bag_result,file=savefile)