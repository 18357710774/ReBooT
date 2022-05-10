ensemble_bagging <- function(train_x, train_y, test_x, test_y, iter=100, parallel=TRUE, 
                             control=rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0), ...)
  
  ## Main function for bagging algorithm
  ## train_x, train_y: training data set
  ## test_x, test_y: test data set for comparsion of prediction performance
  ## iter: the number of iterations, the default value is 100
{ 
  colnames(train_x) <- paste('x', 1:ncol(train_x), sep = '')
  colnames(test_x) <- paste('x', 1:ncol(test_x), sep = '')
  train_y <- as.factor(train_y)
  test_y <- as.factor(test_y)
 
  data_train <- cbind(data.frame(train_x), data.frame(Y=train_y))
  data_test <- cbind(data.frame(test_x), data.frame(Y=test_y))
  
  g_bag <- bagging.modification(Y ~., data=data_train, mfinal=iter, control=control, par=parallel)
  pred_tr <- predict.bagging.modification(g_bag, newdata=data_train, newmfinal=iter)
  pred_te <- predict.bagging.modification(g_bag, newdata=data_test, newmfinal=iter)
  train_err <- pred_tr$error
  test_err <- pred_te$error
  
  obj=list(train_err=train_err,test_err=test_err)
  return(obj)  
}