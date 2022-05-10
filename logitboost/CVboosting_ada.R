CVboosting_ada <- function(train_x, train_y, iter_tmp=500,  control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0), 
                      loss='l', nfolds = 5, parallel = FALSE, pathtmp, ...)
  
  ## Main function for cross-validation of LogitBoost algorithm
  ## train_x, train_y: training data set
  ## iter_tmp: the number of iterations, the default value is 500
  ## nfolds: the training dataset is randomly partitioned into nfold equal size subsamples, the default value is 5

{ 
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
      

      g_rescale<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, loss=loss, 
                                type2="ada", type="real", iter=iter_tmp, nu=1, control=control)

      
      test_err = g_rescale$model$err[,3]
      train_err = g_rescale$model$err[,1]

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
       
      g_rescale<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, loss=loss, 
                                     type2="ada", type="real", iter=iter_tmp, nu=1, control=control)
     
      test_err = g_rescale$model$err[,3]
      train_err = g_rescale$model$err[,1]
      
      Errlist[[num]] = list(test_err = test_err, train_err = train_err)
    }
  }
  
  testErrCVM = rep(0,iter_tmp)
  trainErrCVM = rep(0,iter_tmp)
  for (num in 1:nfolds){
    testErrCVM = testErrCVM + Errlist[[num]]$test_err
    trainErrCVM = trainErrCVM + Errlist[[num]]$train_err
  }
  testErrCVM = testErrCVM/nfolds
  trainErrCVM = trainErrCVM/nfolds
  

  testErrCVMMinIterAll = min(testErrCVM)
  IterOpt = which(testErrCVM == testErrCVMMinIterAll)

  ParaCVSelIterAll = list(IterOpt = IterOpt, testErrCVMMin = testErrCVMMinIterAll)
  
  CVresults = list(Errlist=Errlist, ParaCVSelIterAll = ParaCVSelIterAll)
  
  return(CVresults)
}