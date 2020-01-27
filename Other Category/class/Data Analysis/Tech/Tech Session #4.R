# 11. 24

### Econometrics ###

Steps in Empirical Economic Analysis
  1. Specify an economic model.
  2. Specify an econometric model.
  3. Gather data.
  4. Analyze data according to econometric model.
  5. Draw conclusions about your economic model.


#1.
lm(formula, data)
  formula : a symbolic description of the model to be fitted
  data : a data frame, list or environment containing the variables in the model

  reg1 = lm(prestige~ education+income+women, data=Prestige)
  summary(reg1)


Multiple Rsquared :0.7982 # About 80% of varaince can be explained by the variables




#2. Factor Variable regression with no interactions

reg2 = lm(prestige~education+income+type, data=Prestige)

#3. with interactions

reg3 = lm(prestige ~ income+type*education, data=Prestige)

#4. Logistic Regression for Binary Dependent Variable
glm(formula, data, family="binomial")

#5. Poisson Regression for count dependent variable
glm(formula, data, family="poisson")











###### HOMEWORK ######

a. write code
b. write the name
c. attach code
d. code
e. code
f. explain
g. code
h. screen capture
i. 
