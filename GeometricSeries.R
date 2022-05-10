GeometricSeries <- function (beginVal, endVal, Num){
  # browser()
  aa = log(beginVal)
  bb = log(endVal)
  cc = seq(aa, bb, (bb-aa)/(Num-1))
  dd = exp(cc)
  return(dd)
}
