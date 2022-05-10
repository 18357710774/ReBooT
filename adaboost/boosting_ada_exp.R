boosting_ada_exp <- function(train_x, train_y, test_x, test_y, iter_tmp=500, 
                             control = rpart.control(cp = -1, maxdepth = 1, minsplit = 0),
                             loss = 'e', ...)

  ## Main function for AdaBoost algorithm
  ## train_x, train_y: training data set
  ## test_x, test_y: test data set for comparsion of prediction performance
  ## iter_tmp: the number of iterations, the default value is 500
{ 
  # browser()
  test_err <- rep(0,iter_tmp)
  train_err <- rep(0,iter_tmp)
  l1norm <- rep(0,iter_tmp)
  alpha <- rep(0,iter_tmp)

  ## call real adaboost 
  g_rescale<- ada.modification(train_x, train_y, test.x=test_x, test.y=test_y, loss=loss,
                               type2="ada", type="real",iter = iter_tmp, nu=1, bag.frac=1, control=control)
  

  train_err = g_rescale$model$err[,1]
  test_err = g_rescale$model$err[,3]
  
  infos = l1normComputation(g_rescale)
  
  l1norm = infos$l1norm
  alpha = infos$alpha

  obj=list(train_err=train_err,test_err=test_err,l1norm=l1norm,alpha=alpha)
  return(obj)  
}