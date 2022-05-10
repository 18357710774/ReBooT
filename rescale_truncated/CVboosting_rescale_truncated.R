CVboosting_rescale_truncated <- function(train_x, train_y, ltype = 1, iter_tmp=500, c_seq=c(5),
                                        lc_seq = c(0.1,0.5,1,5,10,30,50), loss="l", ctype=1, 
                                        control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0), 
                                        nfolds = 5, parallel = FALSE, pathtmp, ...)
  ## Main function for cross-validation of ReBooT algorithm
  ## train_x, train_y: training data set
  ## iter_tmp: the number of iterations, the default value is 500
  ## c_seq: the parameter in alpha_k = c/(k+c) 
  ## lc_seq: the parameter in h_k = lc*logk
  ## control: tree setting
  ## nfolds:  the training dataset is randomly partitioned into nfold equal size subsamples, the default value is 5
{ 
  num_c <- length(c_seq)
  num_lc <- length(lc_seq)
  
  Para <- matrix(data=0,2,num_c*num_lc)
  uu <- 0
  for (jj in 1:num_lc){
    for (ii in 1:num_c){
      uu <- uu+1
      Para[,uu] = c(lc_seq[jj],c_seq[ii])
    }
  }
  
  train_x = data.frame(train_x)
  folds = createFolds(y=train_y, k = nfolds, list = TRUE, returnTrain = FALSE) 
  

  Errlist = as.list(seq(nfolds))
  if (parallel == TRUE){
    no_cores <- detectCores()
    cl <- makeCluster(nfolds)
    registerDoParallel(cl)
    
    Errlist = foreach(num = seq(nfolds), .packages =c('rpart','stringr')) %dopar% {
      source(str_c(pathtmp, "cvada.modification.R"))
      source(str_c(pathtmp, "cvada.machine.modification.R"))
      fold_test_x = train_x[folds[[num]],]
      fold_train_x = train_x[-folds[[num]],]
      fold_test_y = train_y[folds[[num]]]
      fold_train_y = train_y[-folds[[num]]]
      test_err <- matrix(data=0,iter_tmp,num_c*num_lc)
      train_err <- matrix(data=0,iter_tmp,num_c*num_lc)
      
      for (m in 1:(num_lc*num_c)){
        lc_tmp <- Para[1,m]
        c_tmp <- Para[2,m]
        
        g_rescale_truncated<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, 
                                                 loss=loss, C_t=c_tmp, lc=lc_tmp, ltype=ltype,
                                                 type2="rescale_truncated", type="real", iter = iter_tmp, 
                                                 ctype=ctype, nu=1, control=control)
        
        test_err[,m] = g_rescale_truncated$model$err[,3]
        train_err[,m] = g_rescale_truncated$model$err[,1]

      }
      obj = list(test_err = test_err, train_err = train_err)
    }
    stopCluster(cl)
  }else
  {
    for (num in 1:nfolds){
      fold_test_x = train_x[folds[[num]],]
      fold_train_x = train_x[-folds[[num]],]
      fold_test_y = train_y[folds[[num]]]
      fold_train_y = train_y[-folds[[num]]]
      test_err <- matrix(data=0,iter_tmp,num_c*num_lc)
      train_err <- matrix(data=0,iter_tmp,num_c*num_lc)
      
      for (m in 1:(num_lc*num_c)){
        lc_tmp <- Para[1,m]
        c_tmp <- Para[2,m]
        
        g_rescale_truncated<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, 
                                                 loss=loss, C_t=c_tmp, lc=lc_tmp, ltype=ltype,
                                                 type2="rescale_truncated", type="real", iter = iter_tmp, 
                                                 ctype=ctype, nu=1, control=control)
        
        test_err[,m] = g_rescale_truncated$model$err[,3]
        train_err[,m] = g_rescale_truncated$model$err[,1]
        
      }
      Errlist[[num]] = list(test_err = test_err, train_err = train_err)
    }
  }
  
  testErrCVM = matrix(data=0,iter_tmp,num_c*num_lc)
  trainErrCVM = matrix(data=0,iter_tmp,num_c*num_lc)
  for (num in 1:nfolds){
    testErrCVM = testErrCVM + Errlist[[num]]$test_err
    trainErrCVM = trainErrCVM + Errlist[[num]]$train_err
  }
  testErrCVM = testErrCVM/nfolds
  trainErrCVM = trainErrCVM/nfolds
  
  testErrCVMMinIterMax = testErrCVM[iter_tmp,which.min(testErrCVM[iter_tmp,])]
  indminIterMax = which(testErrCVM[iter_tmp,] == testErrCVMMinIterMax)
  
  lcCVIterMax = Para[1,indminIterMax]
  cCVIterMax = Para[2,indminIterMax]
  ParaCVSelIterMax = list(lc = lcCVIterMax, c = cCVIterMax, testErrCVMMin = testErrCVMMinIterMax)
  
  
  testErrCVMMinIterAll = min(testErrCVM)
  indminIterAll = which(testErrCVM == testErrCVMMinIterAll)

  indcol = ceiling(indminIterAll/iter_tmp)
  IterOpt = indminIterAll - (indcol-1)*iter_tmp
  lcCVIterAll = Para[1,indcol]
  cCVIterAll = Para[2,indcol]
  ParaCVSelIterAll = list(lc = lcCVIterAll, c = cCVIterAll, IterOpt = IterOpt, testErrCVMMin = testErrCVMMinIterAll)
  
  CVresults = list(Errlist=Errlist, ParaCVSelIterMax = ParaCVSelIterMax, ParaCVSelIterAll = ParaCVSelIterAll)
  
  return(CVresults)
}