"cvada.machine.modification" <-
function(x,y,test.x,test.y,iter=50,nu=1,lossObj,C_t, ctype, lc, ltype, 
         type2=c("truncated", "rescale","rescale_truncated","ada","shrinkage"),
         FM=700, weps=1e-5,oldObj=NULL,na.action=na.action,...)
  # modified from the original R script "ada.machine.R" in the ada package for implementing 
  # AdaBoost, LogitBoost, RTboosting, RSboosting, RBoosting, ReBooT
  {
  kapstat<-function(tab=diag(2) ){
    if(dim(tab)[1]==1){
      return(0)
    }
    if(dim(tab)[1]==dim(tab)[2]){
      rs<-apply(tab,2,sum)
      cs<-apply(tab,1,sum)
      N<-sum(rs)
      E<-sum(rs*cs)/N^2
      O<-sum(diag(tab))/N
      return( (O-E)/(1-E) )
    }else{
      return(0.5)
    }
  }
  tmp<-function(i){
    a1<-sample(which(y==1),1)
    a2<-sample(which(y==-1),1)
    ind<-c(a1,a2)
    return(c(sample(setdiff(1:n,ind),n-val-2,FALSE),ind))
  }
  
  n=dim(x)[1]
  fit=list()
  y<-as.numeric(y)
  dat<-data.frame(y=y,x)
  
  w=rep(1,n)/n
  fits=rep(0,n)
  
  atmp=alpha=vector(length=iter)
  train.err<-rep(0,iter)
  train.kap<-rep(0,iter)
  start=0
  
  fit=oldObj$model$trees
  test.err<-rep(0,iter)
  test.kap<-rep(0,iter)
  test.n<-dim(test.x)[1]
  fits.test<-rep(0,test.n)
  
  start<-start +1
  wfun=lossObj$wfun
  coefs=lossObj$coefs
  method=lossObj$method
  predict.type=lossObj$predict.type
  shift=lossObj$shift
  f1<-f2<-0
  
  M <- seq(0,by=0,length=iter)
  rescale <- seq(0,by=0,length=iter)
  
  IndPos = which(y==1)
  IndNeg = which(y==-1)
  PosNum = length(IndPos)
  NegNum = length(IndNeg)

  for (m in start:iter){
    fit[[m]] =rpart(y~.,data=dat,weights=w,method=method,x=FALSE,y=TRUE,na.action=na.action,...)
    f<-predict.type(fit[[m]],dat)

    errm=sum(w*(sign(f)!=y))
    if( (1-errm)==1 | errm==1 ){
      errm=(1-errm)*0.0001+errm*.9999
    }
    alp=0.5*log( (1-errm)/errm)
    
    alpha[m]=coefs(w,f,y,alp)
    
    if(type2=="ada"){
      fits<-fits+alpha[m]*f
    }
    
    if (type2=="shrinkage"){
      alpha[m]=nu*alpha[m]
      fits<-fits+alpha[m]*f
    }
    
    if(type2=="truncated"){
      M[m] <-  C_t*m^{-2/3}
      if(abs(alpha[m])>M[m]){ 
        alpha[m] <- sign(alpha[m])*M[m]
      } 
      fits<-fits+alpha[m]*f
    }
    

    if(type2=="rescale"){ 
      if (ctype == 1){
        rescale[m] <- 1-C_t/(m+C_t)
      }else{
        rescale[m] <- 1-2/(m+C_t)
      }
      fits<-rescale[m]*fits+alpha[m]*f
    }

    if(type2=="rescale_truncated"){ 
      if (ctype == 1){
        atmp <- C_t/(m+C_t)
      }else{
        atmp <- 2/(m+C_t)
      }
      
      if (ltype == 1)
        ltmp <- lc*log(m+1)
      if (ltype == 2)
        ltmp <- log(log(m+1))
      if (ltype == 3)
        ltmp <- m
      if (ltype == 4)
        ltmp <- m^2
      
      M[m] <- atmp*ltmp
      
      if (abs(alpha[m])>M[m]){ 
        alpha[m] <- sign(alpha[m])*M[m]
      }
      
      rescale[m] <- 1-atmp
      fits<-rescale[m]*fits+alpha[m]*f
    }
    
    if(shift){
      f1=(1-nu)*f+fits
      atmp[m]=1-nu+alpha[m]
    }
    

    indCut = which(abs(fits)>FM)
    fits[indCut] = sign(fits[indCut])*FM
    w=wfun(y,fits)
    w=w/sum(w)
    
    wPos = w[IndPos]
    wNeg = w[IndNeg]
    wSumPos = sum(wPos)
    wSumNeg = sum(wNeg)
    
    if (wSumPos<weps){ 
      if (wSumPos==0){ 
        wPos = weps/PosNum
      }else{ 
        IndPos1 = which(wPos>0) 
        wPos[IndPos1] = wPos[IndPos1]*weps/wSumPos
      }
      IndNeg1 = which(wNeg>0)
      wNeg[IndNeg1] = wNeg[IndNeg1]*(1-weps)/wSumNeg
    }
    
    if (wSumNeg<weps){ 
      if (wSumNeg==0){ 
        wNeg = weps/NegNum
      }else{ 
        IndNeg1 = which(wNeg>0) 
        wNeg[IndNeg1] = wNeg[IndNeg1]*weps/wSumNeg
      }
      IndPos1 = which(wPos>0)
      wPos[IndPos1] = wPos[IndPos1]*(1-weps)/wSumPos
    }
    w[IndPos] = wPos
    w[IndNeg] = wNeg
    
    tab<-table(sign(fits),y)
    train.err[m]<-1-sum(diag(tab))/n
    train.kap[m]<-1-kapstat(tab)
    

    fit1<-predict.type(fit[[m]],test.x)
    if (type2=="ada" | type2=="shrinkage" | type2=="truncated"){
      fits.test<-fits.test + alpha[m]*fit1
    }
    if (type2=="rescale" | type2=="rescale_truncated"){
      fits.test<-rescale[m]*fits.test+alpha[m]*fit1
    }
    
    
   if(shift){
      f2=(1-nu)*fit1+fits.test
    }
    tab<-table(sign(fits.test),test.y)
    test.err[m]<- 1-sum(diag(tab))/test.n
    test.kap[m]<-1-kapstat(tab)
    
    if (m%%1000 == 0){
      print(c(m,train.err[m],test.err[m],sum(wPos),sum(wNeg),errm))
    }
  }
  
  if(shift){
    alpha=atmp
    fits<-f1
    fits.test<-f2
  }
  a1=(fits==0)
  if(sum(a1)>0)
    fits[a1]<-sample(c(-1,1),sum(a1),TRUE,c(.5,.5))
  
  ans<-list()
  ans[[1]]=fits 
  ans[[2]]=fits.test  
  
  errs<-cbind(train.err,train.kap)
  errs<-cbind(errs,test.err,test.kap)

  obj=list(trees=fit,alpha=alpha, rescale=rescale, Fval=ans, errs=errs,shift=shift,lossObj=lossObj)
  return(obj)
}
