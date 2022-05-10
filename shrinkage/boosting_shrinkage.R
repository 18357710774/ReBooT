boosting_shrinkage <- function(train_x, train_y, test_x, test_y, iter_tmp=500,
                               control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0),
                               nu_seq = seq(0.03,1,length.out=10), loss="l", ...)
  
  ## Main function for RSboosting algorithm
  ## train_x, train_y: training data set
  ## test_x, test_y: test data set for comparsion of prediction performance
  ## iter_tmp: the number of iterations, the default value is 500
  ## control: tree setting
  
  { 
  num_nu <- length(nu_seq)
  test_err <- matrix(data=0,iter_tmp,num_nu)
  train_err <- matrix(data=0,iter_tmp,num_nu)
  l1norm <- matrix(data=0,iter_tmp,num_nu)
  alpha <- matrix(data=0,iter_tmp,num_nu)
  
  for (i in 1:num_nu)
  {
    nu_tmp <- nu_seq[i]
    
    g_shrinkage <- ada.modification(train_x, train_y, test.x=test_x, test.y=test_y, 
                                    loss=loss, type2="shrinkage", type="real", iter=iter_tmp, 
                                    nu=nu_tmp, bag.frac=1, control=control)
    
    train_err[,i] <- g_shrinkage$model$err[,1]
    test_err[,i] <- g_shrinkage$model$err[,3]
    
    infos = l1normComputation(g_shrinkage)
    
    l1norm[,i] = infos$l1norm
    alpha[,i] = infos$alpha
  }
  obj=list(train_err=train_err,test_err=test_err,l1norm=l1norm, alpha=alpha)
  return(obj)
}