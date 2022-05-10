
predict.bagging.modification <- function (object, newdata, newmfinal=length(object$trees), ...) {   
# browser()
  
if (newmfinal > length(object$trees) | newmfinal < 1) 
    stop("newmfinal must be 1<newmfinal<mfinal")

formula <- object$formula
n <- length(newdata[, 1])

vardep.summary<-attributes(object)$vardep.summary

nclases <- length(vardep.summary)

pred <-as.data.frame(sapply (object$trees[1:newmfinal], predict, newdata, type="class"))
classfinaltmp <- array(0, c(n, nclases, newmfinal))
classfinal <- array(0, c(n, nclases, newmfinal))
for (i in 1:nclases) {
    classfinaltmp[, i, ] <- matrix(as.numeric(pred == names(vardep.summary)[i]), nrow = n)
    classfinal[, i, ] <- t(cumsum(as.data.frame(t(as.data.frame(classfinaltmp[, i, ])))))
}


if(sum(names(newdata)==as.character(object$formula[[2]]))==0)
{
    tabla <- NULL
    error <- NULL
    probs <- list()
    predclass <- list()
    
    for (m in 1:newmfinal){
        predclasstmp <- rep("O", n)
        predclasstmp[] <- apply(classfinal[, , m], 1, FUN=select, vardep.summary=vardep.summary)
        predclass[[m]] <- predclasstmp
        probs[[m]] <- classfinal[, , m]/apply(classfinal[, , m], 1, sum)
    }
}
else{
    vardep <- newdata[, as.character(object$formula[[2]])]
    
    tabla <- list()
    error <- rep(0, 1, newmfinal)
    probs <- list()
    predclass <- list()
    
    for (m in 1:newmfinal){
        predclasstmp <- rep("O", n)
        predclasstmp[] <- apply(classfinal[, , m], 1, FUN=select, vardep.summary=vardep.summary)
        predclass[[m]] <- predclasstmp
        probs[[m]] <- classfinal[, , m]/apply(classfinal[, , m], 1, sum)
        
        tabla[[m]] <- table(predclasstmp, vardep, dnn = c("Predicted Class",  "Observed Class"))
        error[m] <- 1 - sum(predclasstmp == vardep)/n
      
    }
}

output<- list(formula=formula, votes=classfinal, prob=probs, class=predclass, confusion=tabla, error=error)
}

