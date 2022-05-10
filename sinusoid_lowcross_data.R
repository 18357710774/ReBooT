## The main function for generating low-cross data

sinusoid_lowcross_data <- function(positive_size, negative_size, noise_feature=0){
	f<-function(x,a,b,d)a*(x-b)^2+d
	x1<-runif(positive_size,0,4)
	y1<-f(x1,-1,2,2.3)+runif(positive_size,-1,1)
	x2<-runif(negative_size,2,6)
	y2<-f(x2,1,4,-2.3)+runif(negative_size,-1,1)
	y<-c(rep(1,positive_size),rep(2,negative_size))
	noise_mat<-matrix(rnorm(((positive_size+negative_size)*noise_feature),sd=4),ncol=noise_feature)
	x1=c(x1,x2)
	x2=c(y1,y2)
	if(noise_feature==0)
	  data_x <- cbind(x1,x2)
	else
  	data_x <- cbind(x1,x2,noise_mat)
  data_y <- y
  obj=list(data_x=data_x, data_y=data_y)
  return(obj)
}
