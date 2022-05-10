CVboosting_xgb <- function(train_x, train_y, nrounds = 500, eta_seq=seq(0.1, 1, 0.1), lambda_seq=c(0.1, 1, 10),
                           max_depth=1, objective='binary:logistic', booster = 'gbtree', 
                           nfold = 5, nthread = 5, verbose = TRUE, print_every_n = 100, ...)
  
  # Main function for cross-validation of xgboost algorithm
  # train_x, train_y: training data set
  # eta_seq: the candidate sequence for parameter eta, which controls the learning rate
  # lambda_seq: the candidate sequence for parameter lambda, which controls the l2 regularization term on weights
  # nrounds: max number of boosting iterations
  # nfold: the original dataset is randomly partitioned into nfold equal size subsamples
  # nthread: number of thread used in training, if not set, all threads are used
  
{ 
  dtrain <- xgb.DMatrix(train_x, label=train_y)
  param_base <- list(max_depth=max_depth, nthread=nthread, objective=objective)
  num_lambda <- length(lambda_seq)
  num_eta <- length(eta_seq)
  tuned_para <- matrix(data=0, 2, num_lambda*num_eta)

  testErrCVM <- matrix(data=0, nrounds, num_lambda*num_eta)
  trainErrCVM <- matrix(data=0, nrounds, num_lambda*num_eta)

  count <- 0
  for (m in 1:num_lambda){
    lambda_tmp <- lambda_seq[m]
    for (n in 1:num_eta){
      count <- count + 1
      eta_tmp <- eta_seq[n]
      param <- c(param_base, list(lambda=lambda_tmp, eta=eta_tmp))
      tuned_para[,count] <- c(lambda_tmp, eta_tmp)
      
      cv <- xgb.cv(param, dtrain, nrounds, nfold = nfold, booster = booster,
                   metrics={'error'}, verbose = verbose, print_every_n = print_every_n)
      testErrCVM[,count] <- cv$evaluation_log$test_error_mean
      trainErrCVM[,count] <- cv$evaluation_log$train_error_mean
    }
  }
  
  testErrCVMMinIterMax = testErrCVM[nrounds,which.min(testErrCVM[nrounds,])]
  indminIterMax = which(testErrCVM[nrounds,] == testErrCVMMinIterMax)
  
  lambdaCVIterMax = tuned_para[1,indminIterMax]
  etaCVIterMax = tuned_para[2,indminIterMax]
  ParaCVSelIterMax = list(lambda = lambdaCVIterMax, eta = etaCVIterMax, testErrCVMMin = testErrCVMMinIterMax)
  
  
  testErrCVMMinIterAll = min(testErrCVM)
  indminIterAll = which(testErrCVM == testErrCVMMinIterAll)

  indcol = ceiling(indminIterAll/nrounds)
  IterOpt = indminIterAll - (indcol-1)*nrounds
  
  lambdaCVIterAll = tuned_para[1,indcol]
  etaCVIterAll = tuned_para[2,indcol]
  ParaCVSelIterAll = list(lambda = lambdaCVIterAll, eta = etaCVIterAll, IterOpt = IterOpt, testErrCVMMin = testErrCVMMinIterAll)
  
  CVresults = list(testErrCVM = testErrCVM, trainErrCVM = trainErrCVM, 
                   ParaCVSelIterMax = ParaCVSelIterMax, ParaCVSelIterAll = ParaCVSelIterAll)
  
  return(CVresults)
}