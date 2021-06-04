
fun.model <- function(formula, MyData, fileName) {
  
  lm_mini_model <- lm(formula, data = MyData)
  par(mfrow = c(2, 2))
  plot(lm_mini_model)
  
  estimates_mini_model <- coef(summary(lm_mini_model))
  estimates_mini_model <- data.frame(estimates_mini_model)
  estimates_mini_model <- estimates_mini_model[,c(1*(ncol(estimates_mini_model)/4), 2*(ncol(estimates_mini_model)/4), 3*(ncol(estimates_mini_model)/4), 4*(ncol(estimates_mini_model)/4))]
  colnames(estimates_mini_model)<- c('Estimate', 'SD', 't', 'p')
  
  unit <-as.data.frame(array(1, dim = c(length(estimates_mini_model$Estimate),1)))
  names(unit)<-'Unit'
  estimates_mini_model.change <-((exp(estimates_mini_model$Estimate * unit)-1) * 100)
  names(estimates_mini_model.change) <- 'change'
  estimates_mini_model.change_LL <- (exp((estimates_mini_model$Estimate - 1.96 * estimates_mini_model$SD) 
                                         * unit )-1) * 100
  names(estimates_mini_model.change_LL) <- 'change LL'
  estimates_mini_model.change_UL <- (exp((estimates_mini_model$Estimate + 1.96 * estimates_mini_model$SD)
                                         * unit )-1) * 100
  names(estimates_mini_model.change_UL) <- 'change UL'
  estimates_mini_model.estimates.mini_model <- cbind(rownames(estimates_mini_model),estimates_mini_model.change, estimates_mini_model.change_LL, estimates_mini_model.change_UL, estimates_mini_model)
  print(estimates_mini_model.estimates.mini_model)
  library("writexl")
  write_xlsx(estimates_mini_model.estimates.mini_model,
             paste('C:/Users/sahar.behzadi/Desktop/Noise2Nako/Code/R_Code/Output/LinearRegression/', fileName))
  
}



setwd('C:/Users/sahar.behzadi/Desktop/Noise2Nako/Data/KORA_S3_S4')
MyData <- read.table('KORA_Noise_noMissing_median.csv', sep = ',', header = TRUE)

MyData$sex <- factor(MyData$sex, levels = c(0,1), labels=c(0,1))

MyData$smoking <- factor(MyData$smoking, levels = c(1:3), labels=c(1:3))

MyData_scaled <- MyData[c('age', 'sex', 'smoking', 'bmi', 'lden_org', 'bp_syst')]
MyData_scaled$age <- scale(MyData_scaled$age)
MyData_scaled$bmi <- scale(MyData_scaled$bmi)
MyData_scaled$lden_org <- scale(MyData_scaled$lden_org)
MyData_scaled$bp_syst <- scale(MyData_scaled$bp_syst)
summary(MyData_scaled)

variable.mini_model <- c('age', 'sex', 'smoking', 'bmi', 'lden_org')
mini_model <- paste('age', 'sex', 'smoking', 'bmi', 'lden_org', sep='+')

# estimates_mini_model <- lm(MyData$bp_syst~ MyData$age + MyData$sex +
#                  MyData$smoking + MyData$bmi + MyData$lden_org)

formula <- as.formula(paste("bp_syst ~", mini_model))

fileName <- 'regression_estimates.xlsx'

fun.model(formula, MyData, fileName)
