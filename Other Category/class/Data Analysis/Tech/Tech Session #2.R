a = read.csv(file, header=FALSE, sep=",")
str(a)  # show structure
View(a) # csv는 데이터프레임으로 들어온다.

# Missing Variable을 정해주지 않으면 factor타입이 되어버린다.
c = read.csv("c.csv", T, ",", na.strings=c("No Score"))
str(c)  # factor가 아닌 integer로 나오는 것을 알 수 있다.

# Export to CSV
write.csv(c, file="d.csv", row.names=TRUE, sep=",")

# TXT Files
read.table(file, header=, sep=",")
write.table()

# Import from Excel
install.packages("readxl")
library(readxl) #라이브러리 불러오기

read_excel(path, sheet = 1)
xls1 = read.excel("a.xlsx", sheet=1)


# 복습
# 행 붙이기 (행은 서로 다른 데이터인 데이터프레임)
add_row = data.frame()
rbind(example, add_row)
rbind(example, name = "", gender ="", int, int)
# 열 붙이기 (열은 같은 데이터인 벡터)
cbind()
example$newdata = c()
# 일부 출력
example[c(1,3)]

# 기준 정의에 따라 구별
hw <-heightweight
hw$weightGroup <- cut(hw$weightLb, breaks = c(-Inf, 100, Inf), labels = c("< 100", ">= 100"))


# Missing Values
age = c(23,16,NA)
mean(age) -> error
mean(age, na.rm=TRUE)  ==   mean(age[!is.na(age)])
complete.cases(person) #NA 포함 여부에 따라 T/F반환

person[complete.cases(person), ]
== na.omit(person)
person[ age>20, ]

# Outliers
# boxplot함수는 default로 tukey의 box and whisker 방법을 씀
# upper whisker = ( 1.5 * (3rd q - 1st q) + 3rd q ) round to nearest lower observation

x <- c(1:10, 20, 30)
boxplot(x)

boxplot.stats(ex$score)$out
ex[ex$score>40 , ]


#Obvious Inconsistency
#명백하게 잘못된 데이터들

install.packages("editrules")
library(editrules)
age_rule = editset(c("age>=0", "age<=100"))
violatedEdits(age_rule, people)
summary(violatedEdits(age_rule,people))
drop1 = violatedEdits(age_rule, people)[,1] | violatedEdits(age_rule,people)[,2]
people[!drop1, ]
drop2 = apply(violatedEdits(age_rule, people), 1, any)
people[!drop2, ]




# Simple Transformation
# deducorrect라는 패키지를 사용
install.packages("deducorrect")
library("deducorrect")

# 먼저 규칙이 적힌 txt파일을 만든다.
height_rule = correctionRules("transform rule.txt")

trans2 = correctwithRules(height_rule, trans1)
trans2$corrected



#Data Transformation
split(x,f) # x를 f의 규칙에 따라 나눔

split(people, people$agegroup)
subset(people, height>5) == people[people$height >5, ]
subset(people, status=="married" & yearsmarried>2)

# Merge : 마스터 테이블, using table
merge(x,y,by) : Merge two data frames by common columns

merged = merge(x,y,by=c("id"))


# Sort and Order
z = c(20,11,33,50,47)
sort(z)
sort(z, decreasing=TRUE)

order(z)
order(z, decreasing=TRUE)

merged = merged[order(names(merged))]


# Group-Level Transformation

summaryBy(formula, data, FUN)
install.packages("doBy")
library(doBy)
# calculate mean of score and age by gender group of the data frame merged
summaryBy(score+age~gender, merged, FUN=mean)
summaryBy(score~id, merged, FUN=max)
summaryBy(score~semester, merged, FUN=c(max,min,mean))
