CVensemble_bagging <- function(train_x, train_y, iter=100, control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0), 
                           nfolds = 5, parallel = TRUE, ...)
  
  ## Main function for cross-validation of Bagging algorithm
  ## train_x, train_y: training data set
  ## iter: the number of iterations, the default value is 100
  ## nfolds: the training dataset is randomly partitioned into nfold equal size subsamples, the default value is 5

{ 
  train_x = data.frame(train_x)
  folds = createFolds(y=train_y, k = nfolds, list = TRUE, returnTrain = FALSE) 
  
  
  Errlist = as.list(seq(nfolds))
 
  for (num in 1:nfolds){
    fold_test_x = train_x[folds[[num]],]
    fold_train_x = train_x[-folds[[num]],]
    fold_test_y = train_y[folds[[num]]]
    fold_train_y = train_y[-folds[[num]]]
    
    
    g_bag <- ensemble_bagging(fold_train_x, fold_train_y, fold_test_x, fold_test_y, 
                              iter=iter, parallel=parallel, control=control)
    

    test_err = g_bag$test_err
    train_err = g_bag$train_err
    
    Errlist[[num]] = list(test_err = test_err, train_err = train_err)
  }

  
  testErrCVM = rep(0,iter)
  trainErrCVM = rep(0,iter)
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