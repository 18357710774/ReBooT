boosting_truncated <- function(train_x, train_y, test_x, test_y, iter_tmp=500,
                               control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0),
                               c_seq=logScale(scale.min=1, scale.max=10^3, scale.ratio=1.2), loss="l", ...)
  
  ## Main function for RTboosting algorithm
  ## train_x, train_y: training data set
  ## test_x, test_y: test data set for comparsion of prediction performance
  ## iter_tmp: the number of iterations, the default value is 500
  ## control: tree setting
  
{ 
  num_c <- length(c_seq)
  test_err <- matrix(data=0,iter_tmp,num_c)
  train_err <- matrix(data=0,iter_tmp,num_c)
  l1norm <- matrix(data=0,iter_tmp,num_c)
  alpha <- matrix(data=0,iter_tmp,num_c)
  
  for (i in 1:num_c)
  {
    c_tmp <- c_seq[i]

    g_truncated<- ada.modification(train_x, train_y, test.x=test_x, test.y=test_y, 
                                   loss=loss, type2="truncated", type="real",iter=iter_tmp, 
                                   C_t=c_tmp, nu=1, bag.frac=1, control=control)

    test_err[,i] = g_truncated$model$err[,3]
    train_err[,i] = g_truncated$model$err[,1]
    
    infos = l1normComputation(g_truncated)
    
    alpha[,i] = infos$alpha
    l1norm[,i] = infos$l1norm
    
  }
  obj=list(train_err=train_err,test_err=test_err,l1norm=l1norm, alpha=alpha)
  return(obj) 
}