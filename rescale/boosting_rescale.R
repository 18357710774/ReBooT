boosting_rescale <- function(train_x, train_y, test_x, test_y, iter_tmp=500,
                             control = rpart.control(cp = -1, maxdepth = 1, minsplit = 0),
                             c_seq=logScale(scale.min=0.1, scale.max=10^2, scale.ratio=1.2), 
                             loss = "l", ctype, ...)
  ## Main function for RBoosting algorithm
  ## train_x, train_y: training data set
  ## test_x, test_y: test data set for comparsion of prediction performance
  ## iter_tmp: the number of iterations, the default value is 500
{ 
  num_c <- length(c_seq)
  test_err <- matrix(data=0,iter_tmp,num_c)
  train_err <- matrix(data=0,iter_tmp,num_c)
  l1norm <- matrix(data=0,iter_tmp,num_c)
  alpha <- matrix(data=0,iter_tmp,num_c)
  for (i in 1:num_c)
  {
    c_tmp <- c_seq[i]

    g_rescale<- ada.modification(train_x, train_y, test.x=test_x, test.y=test_y, loss=loss, C_t=c_tmp,
                                 type2="rescale", type="real", iter = iter_tmp, ctype=ctype, nu=1, bag.frac=1, control=control)
  
    
    test_err[,i]= g_rescale$model$err[,3]
    train_err[,i] = g_rescale$model$err[,1]
    
    infos = l1normComputation(g_rescale)
    
    alpha[,i] = infos$alpha
    l1norm[,i] = infos$l1norm
    
  }
  obj=list(train_err=train_err,test_err=test_err,l1norm=l1norm,alpha=alpha)
  return(obj) 
}