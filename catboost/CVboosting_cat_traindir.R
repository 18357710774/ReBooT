CVboosting_cat_traindir <- function(train_x, train_y, iterations = 500, learning_rate_seq=seq(0.1, 1, 0.1), 
                           l2_leaf_reg_seq=c(0.1, 1, 10), boosting_type = 'Ordered', max_depth=1, 
                           loss_function='Logloss', custom_loss = c('Logloss', 'Accuracy'), nfolds = 5, 
                           thread_count = 5, logging_level	= 'Silent', verbose = 100, parallel = FALSE, 
                           train_dir = 'train_dir', ...)

  # Main function for cross-validation of CatBoost algorithm
  # train_x, train_y: training data set
  # learning_rate_seq: the candidate sequence for parameter eta, which controls the learning rate
  # l2_leaf_reg_seq: the candidate sequence for parameter lambda, which controls the l2 regularization term on weights
  # iterations: max number of boosting iterations
  # loss_function: the metric to use in training. See the Objectives and metrics section in 
  #                https://catboost.ai/docs/concepts/loss-functions-classification.html
  # custom_loss: metric values to output during training. These functions are not optimized and are 
  #                displayed for informational purposes only. See the Objectives and metrics section in 
  #                https://catboost.ai/docs/concepts/loss-functions-classification.html
  # nfolds: the original dataset is randomly partitioned into nfold equal size subsamples
  # thread_count: number of thread used in training, if not set, all threads are used
  # logging_level: the logging level to output to stdout. Possible values:
  #                Silent ！ Do not output any logging information to stdout.
  #                Verbose ！ Output the following data to stdout: optimized metric, elapsed time of training, and
  #                                                               remaining time of training
  #                Info ！ Output additional information and the number of trees.
  #                Debug ！ Output debugging information.
  # verbose: the frequency of iterations to print the information to stdout
  # train_dir: the directory for storing the files generated during training
  
{ 
  train_dir_base = train_dir

  train_x = data.frame(train_x)
  
  param_base <- list(iterations=iterations, boosting_type=boosting_type, depth=max_depth,
                     loss_function=loss_function, custom_loss=custom_loss, logging_level=logging_level,
                     thread_count=thread_count, verbose=verbose)
  
  num_learning_rate <- length(learning_rate_seq)
  num_l2_leaf_reg <- length(l2_leaf_reg_seq)
  
  tuned_para <- matrix(data=0, 2, num_learning_rate*num_l2_leaf_reg)
  
  count <- 0
  for (m in 1:num_l2_leaf_reg){
    for (n in 1:num_learning_rate){
      count <- count+1
      tuned_para[,count] = c(l2_leaf_reg_seq[m],learning_rate_seq[n])
    }
  }
  
  folds = createFolds(y=train_y, k=nfolds, list=TRUE, returnTrain=FALSE) 
  
  Errlist = as.list(seq(nfolds))

  if (parallel == TRUE){
    no_cores <- detectCores()
    cl <- makeCluster(nfolds)
    registerDoParallel(cl)
    
    Errlist = foreach(num = seq(nfolds), .packages =c('catboost', 'stringr')) %dopar% {
      fold_test_x = train_x[folds[[num]],]
      fold_train_x = train_x[-folds[[num]],]
      fold_test_y = train_y[folds[[num]]]
      fold_train_y = train_y[-folds[[num]]]
      train_pool <- catboost.load_pool(fold_train_x, label=fold_train_y)
      test_pool <- catboost.load_pool(fold_test_x, label=fold_test_y)
      test_err <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
      train_err <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
      
      train_dir <- str_c(train_dir_base, 'fold', as.character(num))
      
      for (m in 1:(num_learning_rate*num_l2_leaf_reg)){
        l2_leaf_reg_tmp <- tuned_para[1,m]
        learning_rate_tmp <- tuned_para[2,m]
        fit_params <- c(param_base, list(l2_leaf_reg=l2_leaf_reg_tmp, learning_rate=learning_rate_tmp, train_dir=train_dir))
        
        model <- catboost.train(learn_pool = train_pool, test_pool = test_pool, params = fit_params)
        train_result_tmp <- read.table(str_c(train_dir, '/learn_error.tsv'), header = T)
        train_err[,m] <- 1 - train_result_tmp$Accuracy
        test_result_tmp <- read.table(str_c(train_dir, '/test_error.tsv'), header = T)
        test_err[,m] <- 1 - test_result_tmp$Accuracy
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
      train_pool <- catboost.load_pool(fold_train_x, label=fold_train_y)
      test_pool <- catboost.load_pool(fold_test_x, label=fold_test_y)
      test_err <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
      train_err <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
      
      train_dir <- str_c(train_dir_base, 'fold', as.character(num))
      
      for (m in 1:(num_learning_rate*num_l2_leaf_reg)){
        l2_leaf_reg_tmp <- tuned_para[1,m]
        learning_rate_tmp <- tuned_para[2,m]
        fit_params <- c(param_base, list(l2_leaf_reg=l2_leaf_reg_tmp, learning_rate=learning_rate_tmp, train_dir=train_dir))
        
        model <- catboost.train(learn_pool = train_pool, test_pool = test_pool, params = fit_params)
        train_result_tmp <- read.table(str_c(train_dir, '/learn_error.tsv'), header = T)
        train_err[,m] <- 1 - train_result_tmp$Accuracy
        test_result_tmp <- read.table(str_c(train_dir, '/test_error.tsv'), header = T)
        test_err[,m] <- 1 - test_result_tmp$Accuracy
      }
      Errlist[[num]] = list(test_err = test_err, train_err = train_err)
    }
  }
  
  testErrCVM = matrix(data=0,iterations,num_learning_rate*num_l2_leaf_reg)
  trainErrCVM = matrix(data=0,iterations,num_learning_rate*num_l2_leaf_reg)
  for (num in 1:nfolds){
    testErrCVM = testErrCVM + Errlist[[num]]$test_err
    trainErrCVM = trainErrCVM + Errlist[[num]]$train_err
  }
  testErrCVM = testErrCVM/nfolds
  trainErrCVM = trainErrCVM/nfolds
  
  testErrCVMMinIterMax = testErrCVM[iterations,which.min(testErrCVM[iterations,])]
  indminIterMax = which(testErrCVM[iterations,] == testErrCVMMinIterMax)
  
  l2_leaf_regCVIterMax = tuned_para[1,indminIterMax]
  learning_rateCVIterMax = tuned_para[2,indminIterMax]
  ParaCVSelIterMax = list(l2_leaf_reg=l2_leaf_regCVIterMax, learning_rate=learning_rateCVIterMax, 
                          testErrCVMMin = testErrCVMMinIterMax)
  
  
  testErrCVMMinIterAll = min(testErrCVM)
  indminIterAll = which(testErrCVM == testErrCVMMinIterAll)
  
  indcol = ceiling(indminIterAll/iterations)
  IterOpt = indminIterAll - (indcol-1)*iterations
  l2_leaf_regCVIterAll = tuned_para[1,indcol]
  learning_rateCVIterAll = tuned_para[2,indcol]
  ParaCVSelIterAll = list(l2_leaf_reg = l2_leaf_regCVIterAll, learning_rate = learning_rateCVIterAll, 
                          IterOpt = IterOpt, testErrCVMMin = testErrCVMMinIterAll)
  
  CVresults = list(Errlist=Errlist, ParaCVSelIterMax = ParaCVSelIterMax, ParaCVSelIterAll = ParaCVSelIterAll)
  
  return(CVresults)
}