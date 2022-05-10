CVboosting_rescale <- function(train_x, train_y, iter_tmp=500, 
                               control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0), 
                               c_seq=logScale(scale.min=0.1, scale.max=10^2, scale.ratio=1.2),
                               nfolds = 5, parallel = FALSE, loss = "l", ctype, pathtmp, ...)
  
  ## Main function for cross-validation of RBoosting algorithm
  ## train_x, train_y: training data set
  ## iter_tmp: the number of iterations, the default value is 500
  ## nfolds: the training dataset is randomly partitioned into nfold equal size subsamples, the default value is 5

{ 
  num_c <- length(c_seq)
  
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
      test_err <- matrix(data=0,iter_tmp,num_c)
      train_err <- matrix(data=0,iter_tmp,num_c)
      
      for (m in 1:num_c){
        c_tmp <- c_seq[m]
        
        g_rescale<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, loss=loss, 
                                  C_t=c_tmp, ctype=ctype, type2="rescale",type="real", iter = iter_tmp, nu=1, control=control)
        
        
        test_err[,m] = g_rescale$model$err[,3]
        train_err[,m] = g_rescale$model$err[,1]

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
      test_err <- matrix(data=0,iter_tmp,num_c)
      train_err <- matrix(data=0,iter_tmp,num_c)
      
      for (m in 1:num_c){
        c_tmp <- c_seq[m]
        
        g_rescale<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, loss=loss, 
                                       C_t=c_tmp, ctype=ctype, type2="rescale",type="real", iter = iter_tmp, nu=1, control=control)
       
        test_err[,m] = g_rescale$model$err[,3]
        train_err[,m] = g_rescale$model$err[,1]
        
      }
      Errlist[[num]] = list(test_err = test_err, train_err = train_err)
    }
  }
  
  testErrCVM = matrix(data=0,iter_tmp,num_c)
  trainErrCVM = matrix(data=0,iter_tmp,num_c)
  for (num in 1:nfolds){
    testErrCVM = testErrCVM + Errlist[[num]]$test_err
    trainErrCVM = trainErrCVM + Errlist[[num]]$train_err
  }
  testErrCVM = testErrCVM/nfolds
  trainErrCVM = trainErrCVM/nfolds
  
  testErrCVMMinIterMax = testErrCVM[iter_tmp,which.min(testErrCVM[iter_tmp,])]
  indminIterMax = which(testErrCVM[iter_tmp,] == testErrCVMMinIterMax)
  
  cCVIterMax = c_seq[indminIterMax]
  ParaCVSelIterMax = list(c = cCVIterMax, testErrCVMMin = testErrCVMMinIterMax)
  
  
  testErrCVMMinIterAll = min(testErrCVM)
  indminIterAll = which(testErrCVM == testErrCVMMinIterAll)

  indcol = ceiling(indminIterAll/iter_tmp)
  IterOpt = indminIterAll - (indcol-1)*iter_tmp
  cCVIterAll = c_seq[indcol]
  ParaCVSelIterAll = list(c = cCVIterAll, IterOpt = IterOpt, testErrCVMMin = testErrCVMMinIterAll)
  
  CVresults = list(Errlist=Errlist, ParaCVSelIterMax = ParaCVSelIterMax, ParaCVSelIterAll = ParaCVSelIterAll)
  
  return(CVresults)
}