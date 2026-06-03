# Example of predicting Cognition (Cattell performance) from multiple brain measures
# rik.henson@mrc-cbu.cam.ac.uk March 2025

## Set working directory to source directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(dplyr)     # for data wrangling
library(stringr)   # for data wrangling
library(glmnet)    # for ridge regression
library(caret)     # for ML
library(ggplot2)   # for plotting

noise_ceiling = 100*0.747^2 # From independent CamCAN Validation dataset, correlating odd/even trials 

df = read.csv('cog_res_data.csv')
nrow(df)

##########################################
# Cross-validated stacking

set.seed(123)    # for reproducibility

nFold = 10
nPerm = 20

# Confounds: Sex, PolyGenic Score (PGS) and Arm (since different sequences in different Arms - assume offset only)
X_con = df[,c("Arm", "PGS", "Sex")] # Age and TIV excluded because assumed to act through brain (though PGS could too?)

X_GMV = df %>% select(starts_with("ROI"))      # Gray-Matter volume (98 HOA ROIs)
X_WMI = df %>% select(starts_with("MK_ROI"))   # White-Matter Integrity (48 Julich ROIs)
X_rFC = df %>% select(contains("Rest"))        # Resting-state Functional Connectivity (153 values within and between 17 Yeo Networks)

all_folds = seq(1,nFold)

R2 = matrix(nrow = nPerm, ncol = 6,)           # Variance explained

for (i in 1:nPerm) {
  print(i)
  
  folds=createFolds(seq(1,nrow(X_con)), k = 10, list = TRUE, returnTrain = FALSE)
  fR2 = matrix(nrow = nFold, ncol = 6,)

  for (f in 1:nFold) {
    inTrain = setdiff(all_folds,f)
    inTrain = unlist(folds[inTrain])
    
    y.train = df$Cattell[inTrain]
    y.test  = df$Cattell[-inTrain]
    
    # Confounds (con)
    x.train = X_con[inTrain,]
    fit.con = train(x.train, y.train, method="glmnet")
    pre.con = predict(fit.con, x.train)
    pre.con.test = predict(fit.con, X_con[-inTrain,])
    fR2[f,1] = cor(y.test, pre.con.test)^2
    
    # GMV
    x.train = X_GMV[inTrain,]
    fit.GMV = train(x.train, y.train, method="glmnet")
    pre.GMV = predict(fit.GMV, x.train)
    pre.GMV.test = predict(fit.GMV, X_GMV[-inTrain,])
    #fR2[f,2] = cor(y.test, pre.GMV.test)^2

    # Stacking GMV+con
    x.train = data.frame(pre.con, pre.GMV)
    fit.all = train(x.train, y.train, method="glm")
    x.test  = data.frame(pre.con.test, pre.GMV.test)
    names(x.test) <- c('pre.con', 'pre.GMV') # need to rename to match trained variables
    pre.all.test = predict(fit.all, x.test)
    fR2[f,2] = cor(y.test, pre.all.test)^2 #- fR2[f,1]
    
    
    # WMI
    x.train = X_WMI[inTrain,]
    fit.WMI = train(x.train, y.train, method="glmnet")
    pre.WMI = predict(fit.WMI, x.train)
    pre.WMI.test = predict(fit.WMI, X_WMI[-inTrain,])
    #fR2[f,3] = cor(y.test, pre.WMI.test)^2
    
    # Stacking WMI+con
    x.train = data.frame(pre.con, pre.WMI)
    fit.all = train(x.train, y.train, method="glm")
    x.test  = data.frame(pre.con.test, pre.WMI.test)
    names(x.test) <- c('pre.con', 'pre.WMI') # need to rename to match trained variables
    pre.all.test = predict(fit.all, x.test)
    fR2[f,3] = cor(y.test, pre.all.test)^2 #- sR2[f,1]
    
    
    # rFC
    x.train = X_rFC[inTrain,]
    fit.rFC = train(x.train, y.train, method="glmnet")
    pre.rFC = predict(fit.rFC, x.train)
    pre.rFC.test = predict(fit.rFC, X_rFC[-inTrain,])
    #fR2[f,4] = cor(y.test, pre.rFC.test)^2
    
    # Stacking rFC+con
    x.train = data.frame(pre.con, pre.rFC)
    fit.all = train(x.train, y.train, method="glm")
    x.test  = data.frame(pre.con.test, pre.rFC.test)
    names(x.test) <- c('pre.con', 'pre.rFC') # need to rename to match trained variables
    pre.all.test = predict(fit.all, x.test)
    fR2[f,4] = cor(y.test, pre.all.test)^2 #- sR2[f,1]
    
    
    # Stacking GMV+WMI+con
    x.train = data.frame(pre.con, pre.GMV, pre.WMI)
    fit.all = train(x.train, y.train, method="glm")
    x.test  = data.frame(pre.con.test, pre.GMV.test, pre.WMI.test)
    x.train = data.frame(pre.con, pre.GMV, pre.WMI)
    fit.all = train(x.train, y.train, method="glm")
    x.test  = data.frame(pre.con.test, pre.GMV.test, pre.WMI.test)
    names(x.test) <- c('pre.con', 'pre.GMV', 'pre.WMI') # need to rename to match trained variables
    pre.all.test = predict(fit.all, x.test)
    fR2[f,5] = cor(y.test, pre.all.test)^2 #- sR2[f,1]

    
    # Stacking GMV+WMI+rFC+con
    x.train = data.frame(pre.con, pre.GMV, pre.WMI, pre.rFC)
    fit.all = train(x.train, y.train, method="glm")
    x.test  = data.frame(pre.con.test, pre.GMV.test, pre.WMI.test, pre.rFC.test)
    names(x.test) <- c('pre.con', 'pre.GMV', 'pre.WMI', 'pre.rFC') # need to rename to match trained variables
    pre.all.test = predict(fit.all, x.test)
    fR2[f,6] = cor(y.test, pre.all.test)^2 #- sR2[f,1]
  }
  
  R2[i,] = colMeans(fR2)
}

#toplot = seq(1,6)
toplot = c(1,2,5,6)

names = c("con", "GMV+con", "WMI+con", "rFC+con", "WMI+GMV+con", "rFC+WMI+GMV+con")

pdat <- data.frame(
  name=names[toplot],
  value=100*colMeans(R2[,toplot]),
  sd=100*apply(R2[,toplot], 2, sd)
)
pdat

ggplot(pdat) + 
  geom_bar(aes(x=factor(name, names[toplot]), y=value), stat="identity", fill="skyblue", alpha=0.7) +
  geom_errorbar( aes(x=name, ymin=value-sd, ymax=value+sd), width=0.4, colour="orange", alpha=0.9, size=1) +
    xlab("") +
    ylab(paste("% Variance (nPerm=",nPerm,")",sep="")) +
    geom_hline(yintercept=noise_ceiling) +
    ylim(0,noise_ceiling+5)

ggsave("var_explained_cons.png", width=2000, height=1000, units="px", dpi=300)

#save.image(file='cog_reserve.RData')



