boosting_gbm <- function(train_x, train_y, test_x, test_y, boosting = 'gbdt', nrounds=500, 
                         eta_seq=logScale(scale.min=0.01, scale.max=1, scale.ratio=1.5),
                         lambda_seq=c(0.1, 1, 10, 100), max_depth=1, objective='binary', 
                         nthread=5, eval='binary_error', eval_freq = 100, verbose = 1, ...)
  
  # Main function for lightgbm algorithm
  # train_x, train_y: training data set
  # test_x, test_y: test data set for comparsion of prediction performance
  # boosting: options--gbdt, rf, dart, goss
  # nrounds: max number of boosting iterations
  # eta_seq: the candidate sequence for parameter eta, which controls the learning rate
  # lambda_seq: the candidate sequence for parameter lambda, which controls the l2 regularization term on weights
  # nrounds: max number of boosting iterations
  # nthread: number of thread used in training, if not set, all threads are used

{ 
  dtrain <- lgb.Dataset(train_x, label=train_y)
  dtest <- lgb.Dataset(test_x, label=test_y)
  
  valids = list(train = dtrain, test = dtest)
  
  param_base <- list(max_depth=max_depth, nthread=nthread, objective=objective, boosting=boosting)
  
  num_lambda <- length(lambda_seq)
  num_eta <- length(eta_seq)
  
  test_err <- matrix(data=0, nrounds, num_lambda*num_eta)
  train_err <- matrix(data=0, nrounds, num_lambda*num_eta)
  l1norm <- matrix(data=0, nrounds, num_lambda*num_eta)
  alpha <- matrix(data=0, nrounds, num_lambda*num_eta)
  
  count <- 0
  for (m in 1:num_lambda){
    lambda_tmp <- lambda_seq[m]
    for (n in 1:num_eta)
    {
      count <- count+1
      eta_tmp <- eta_seq[n]
      
      param <- c(param_base, list(lambda=lambda_tmp, eta=eta_tmp))
      
      bst <- lgb.train(param, dtrain, nrounds, valids=valids, eval=eval, verbose=verbose, eval_freq=eval_freq)
      test_err[,count] = unlist(bst$record_evals$test$binary_error$eval)
      train_err[,count] = unlist(bst$record_evals$train$binary_error$eval)
      alpha[,count] = eta_tmp
      l1norm[,count] = cumsum(alpha[,count])
    }
  }
  obj=list(train_err=train_err, test_err=test_err, l1norm=l1norm, alpha=alpha)
  return(obj)
}