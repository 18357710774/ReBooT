boosting_xgb <- function(train_x, train_y, test_x, test_y, nrounds=500, 
                         eta_seq=logScale(scale.min=0.01, scale.max=1, scale.ratio=1.5),
                         lambda_seq=c(0.1, 1, 10, 100), max_depth=1, objective='binary:logistic', 
                         booster = 'gbtree', eval_metric = 'error', nthread=5, 
                         verbose = TRUE, print_every_n = 100, ...)
  # Main function for xgboost algorithm
  # train_x, train_y: training data set
  # test_x, test_y: test data set for comparsion of prediction performance
  # eta_seq: the candidate sequence for parameter eta, which controls the learning rate
  # lambda_seq: the candidate sequence for parameter lambda, which controls the l2 regularization term on weights
  # nrounds: max number of boosting iterations
  # nthread: number of thread used in training, if not set, all threads are used
{ 

  dtrain <- xgb.DMatrix(train_x, label=train_y)
  dtest <- xgb.DMatrix(test_x, label=test_y)
  watchlist = list(train = dtrain, test = dtest)
  param_base <- list(max_depth=max_depth, nthread=nthread, 
                     objective=objective, booster=booster, eval_metric=eval_metric)
  
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

      bst <- xgb.train(param, dtrain, nrounds, watchlist, verbose = verbose, print_every_n = print_every_n)
      test_err[,count] = bst$evaluation_log$test_error
      train_err[,count] = bst$evaluation_log$train_error
      alpha[,count] = eta_tmp
      l1norm[,count] = cumsum(alpha[,count])
    }
  }
  obj=list(train_err=train_err, test_err=test_err, l1norm=l1norm, alpha=alpha)
  return(obj)
}