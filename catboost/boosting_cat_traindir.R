
boosting_cat_traindir <- function(train_x, train_y, test_x, test_y, iterations = 500, 
                         learning_rate_seq=logScale(scale.min=0.01, scale.max=1, scale.ratio=1.5), 
                         l2_leaf_reg_seq=c(0.1, 1, 10, 100), boosting_type = 'Ordered', max_depth=1,
                         loss_function='Logloss', custom_loss = c('Accuracy', 'Precision', 'Recall'),
                         thread_count = 5, logging_level = 'Silent', verbose = 100, 
                         train_dir = 'train_dir', ...)
  
  # Main function for CatBoost algorithm
  # train_x, train_y: training data set
  # test_x, test_y: testing data set
  # learning_rate_seq: the candidate sequence for parameter eta, which controls the learning rate
  # l2_leaf_reg_seq: the candidate sequence for parameter lambda, which controls the l2 regularization term on weights
  # iterations: max number of boosting iterations
  # loss_function: the metric to use in training. See the Objectives and metrics section in 
  #                https://catboost.ai/docs/concepts/loss-functions-classification.html
  # custom_loss: metric values to output during training. These functions are not optimized and are 
  #                displayed for informational purposes only. See the Objectives and metrics section in 
  #                https://catboost.ai/docs/concepts/loss-functions-classification.html
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
  train_x = data.frame(train_x)
  test_x = data.frame(test_x)
  
  train_pool <- catboost.load_pool(train_x, label=train_y)
  test_pool <- catboost.load_pool(test_x, label=test_y)
  
  param_base <- list(iterations=iterations, boosting_type=boosting_type, depth=max_depth,
                     loss_function=loss_function, custom_loss=custom_loss, logging_level=logging_level,
                     thread_count=thread_count, verbose=verbose, train_dir=train_dir)

  num_learning_rate <- length(learning_rate_seq)
  num_l2_leaf_reg <- length(l2_leaf_reg_seq)
  
  test_err <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
  train_err <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
  l1norm <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
  alpha <- matrix(data=0, iterations, num_learning_rate*num_l2_leaf_reg)
 
  count <- 0
  for (m in 1:num_l2_leaf_reg){
    l2_leaf_reg_tmp <- l2_leaf_reg_seq[m]
    for (n in 1:num_learning_rate)
    {
      count <- count+1
      learning_rate_tmp <- learning_rate_seq[n]
      
      fit_params <- c(param_base, list(l2_leaf_reg=l2_leaf_reg_tmp, learning_rate=learning_rate_tmp))
      model <- catboost.train(learn_pool = train_pool, test_pool = test_pool, params = fit_params)
      train_result_tmp <- read.table(str_c(train_dir, '/learn_error.tsv'), header = T)
      train_err[,count] <- 1 - train_result_tmp$Accuracy
      test_result_tmp <- read.table(str_c(train_dir, '/test_error.tsv'), header = T)
      test_err[,count] <- 1 - test_result_tmp$Accuracy

      alpha[,count] = learning_rate_tmp
      l1norm[,count] = cumsum(alpha[,count])
    }
  }
  obj=list(train_err=train_err, test_err=test_err, l1norm=l1norm, alpha=alpha)
  return(obj)
}