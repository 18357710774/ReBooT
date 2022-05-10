boosting_rescale_truncated <- function(train_x, train_y, test_x, test_y, ltype=1, iter_tmp=500,
                                       control = rpart.control(cp = -1 , maxdepth = 1 , minsplit = 0),
                                       c_seq=logScale(scale.min=1, scale.max=10^3, scale.ratio=1.2), 
                                       lc_seq = c(1,5,10,50,100,500), loss="l", ctype=1, ...)
  ## Main function for ReBooT algorithm
  ## train_x, train_y: training data set
  ## test_x, test_y: test data set for comparsion of prediction performance
  ## iter_tmp: the number of iterations, the default value is 500
  ## c_seq: the parameter in alpha_k = c/(k+c) 
  ## lc_seq: the parameter in h_k = lc*logk
  ## control: tree setting
{ 
  num_c <- length(c_seq)
  num_lc <- length(lc_seq)
  test_err <- matrix(data=0,iter_tmp,num_c*num_lc)
  train_err <- matrix(data=0,iter_tmp,num_c*num_lc)
  l1norm <- matrix(data=0,iter_tmp,num_c*num_lc)
  alpha <- matrix(data=0,iter_tmp,num_c*num_lc)
  uu <- 0
  for (jj in 1:num_lc){
    lc_tmp <- lc_seq[jj]
    for (ii in 1:num_c)
    {
      uu <- uu+1
      c_tmp <- c_seq[ii]

      g_rescale_truncated<- ada.modification(train_x, train_y, test.x=test_x, test.y=test_y, 
                                             loss=loss, C_t=c_tmp, lc=lc_tmp, ltype=ltype, 
                                             type2="rescale_truncated", type="real", iter = iter_tmp, 
                                             ctype=ctype, nu=1, bag.frac=1, control=control)
      
      
      test_err[,uu] = g_rescale_truncated$model$err[,3]
      train_err[,uu] = g_rescale_truncated$model$err[,1]
      
      infos = l1normComputation(g_rescale_truncated)
      
      alpha[,uu] = infos$alpha
      l1norm[,uu] = infos$l1norm
    }
  }
  obj=list(train_err=train_err,test_err=test_err,l1norm=l1norm,alpha=alpha)
  return(obj)
}