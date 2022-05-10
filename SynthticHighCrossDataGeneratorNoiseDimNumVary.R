## Main function for generating low-cross and high cross data with changing numbers of noise dimension

rm(list=ls())
library(stringr)

t_num = 20
N = 200
noise_feature_Cross = c(6, 8, 10, 12, 14, 16, 18, 22, 26, 30)

# set the root path of the code
script.dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
pathtmp <- str_c(script.dir, "/") 

# set the saving path of synthetic data
savepath = str_c(pathtmp, "data/")

if (dir.exists(savepath) == FALSE){
  dir.create(savepath)
}


source(str_c(pathtmp, "sinusoid_lowcross_data.R")) 
source(str_c(pathtmp, "sinusoid_highcross_data.R"))

for (ii in 1:length(noise_feature_Cross)){
  
  noise_feature = noise_feature_Cross[ii]
  savefile = str_c(savepath,"SynDataYaoTrN200NoiDim", as.character(noise_feature), ".Rdata")
  
  tt = 1
  dataGer = list()
  while(tt<=t_num) # loops
  {
    # generate training dataset 
    p_size_train <- N/2
    n_size_train <- N/2
    data_train <- sinusoid_highcross_data(p_size_train,n_size_train, noise_feature)
    train_x <- data_train$data_x
    train_y <- data_train$data_y

    # generate test dataset for the performance comparison
    p_size_test <- 2000
    n_size_test <- 2000
    data_test <- sinusoid_highcross_data(p_size_test,n_size_test,noise_feature)
    test_x <- data_test$data_x
    test_y <- data_test$data_y
    
    dataGer[[tt]] = list(train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
    
    tt <- tt+1
  }
  
  save(dataGer,file=savefile)
}
  
