CVboosting_shrinkage <- function(train_x, train_y, iter_tmp=500, 
                               control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0), 
                               nu_seq = seq(0.03,1,length.out=10), loss="l", 
                               nfolds = 5, parallel = FALSE, pathtmp, ...)
  
  ## Main function for cross-validation of RSboosting algorithm
  ## train_x, train_y: training data set
  ## iter_tmp: the number of iterations, the default value is 100
  ## control: tree setting
  ## nfolds:  the training dataset is randomly partitioned into nfold equal size subsamples, the default value is 5
{ 
  num_nu <- length(nu_seq)
  
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
      test_err <- matrix(data=0,iter_tmp,num_nu)
      train_err <- matrix(data=0,iter_tmp,num_nu)
      
      for (m in 1:num_nu){
        nu_tmp <- nu_seq[m]
        
        g_shrinkage<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, loss=loss, 
                                         type2="shrinkage", type="real", iter=iter_tmp, nu=nu_tmp, control=control)
          
        test_err[,m] = g_shrinkage$model$err[,3]
        train_err[,m] = g_shrinkage$model$err[,1]

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
      test_err <- matrix(data=0,iter_tmp,num_nu)
      train_err <- matrix(data=0,iter_tmp,num_nu)
      
      for (m in 1:num_nu){
        nu_tmp <- nu_seq[m]
        
        g_shrinkage<- cvada.modification(x=fold_train_x, y=fold_train_y, test.x=fold_test_x, test.y=fold_test_y, loss=loss, 
                                         type2="shrinkage",type="real", iter=iter_tmp, nu=nu_tmp, control=control)
        
        test_err[,m] = g_shrinkage$model$err[,3]
        train_err[,m] = g_shrinkage$model$err[,1]

      }
      Errlist[[num]] = list(test_err = test_err, train_err = train_err)
    }
  }
  
  testErrCVM = matrix(data=0,iter_tmp,num_nu)
  trainErrCVM = matrix(data=0,iter_tmp,num_nu)
  for (num in 1:nfolds){
    testErrCVM = testErrCVM + Errlist[[num]]$test_err
    trainErrCVM = trainErrCVM + Errlist[[num]]$train_err
  }
  testErrCVM = testErrCVM/nfolds
  trainErrCVM = trainErrCVM/nfolds
  
  testErrCVMMinIterMax = testErrCVM[iter_tmp,which.min(testErrCVM[iter_tmp,])]
  indminIterMax = which(testErrCVM[iter_tmp,] == testErrCVMMinIterMax)
  
  nuCVIterMax = nu_seq[indminIterMax]
  ParaCVSelIterMax = list(nu = nuCVIterMax, testErrCVMMin = testErrCVMMinIterMax)
  
  
  testErrCVMMinIterAll = min(testErrCVM)
  indminIterAll = which(testErrCVM == testErrCVMMinIterAll)

  indcol = ceiling(indminIterAll/iter_tmp)
  IterOpt = indminIterAll - (indcol-1)*iter_tmp
  nuCVIterAll = nu_seq[indcol]
  ParaCVSelIterAll = list(nu = nuCVIterAll, IterOpt = IterOpt, testErrCVMMin = testErrCVMMinIterAll)
  
  CVresults = list(Errlist=Errlist, ParaCVSelIterMax = ParaCVSelIterMax, ParaCVSelIterAll = ParaCVSelIterAll)
  
  return(CVresults)
}