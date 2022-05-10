l1normComputation <- function (obj){
  # browser()
  type2 = obj$type2
  alpha = obj$model$alpha
  rescale = obj$model$rescale
  
  L = length(alpha)
  l1norm = rep(0,L)
  
  l1norm[1] = abs(alpha[1])
  for (m in 2:L){
    if (type2=="ada" | type2=="shrinkage" | type2=="truncated"){
      l1norm[m] = l1norm[m-1] + abs(alpha[m])
    }
    if (type2=="rescale" | type2=="rescale_truncated"){
      l1norm[m] = l1norm[m-1]*rescale[m] + abs(alpha[m])
    }
  }
  infos = list(type2 = type2, alpha = alpha, l1norm = l1norm, rescale = rescale)
  return(infos)
}
