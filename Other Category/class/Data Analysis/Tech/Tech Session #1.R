# Vector : a collection of ordered homogeneous elements
# Matrix : a vector with two-dimensional shape information
# Data.frame : a set of heterogeneous vectors of equal length
# List : a vector with possible heterogeneous elements

1. Vector
x <- c("a", "b", "c")
y <- c(1,2,3,4)
length(x)
length(y)

score <- c(math=100, eng=30, science=70)
sort(score)
names(score)

x <- c(1,2,3)
y <- c(5,0,1)
x+1
x-y
x^y

var()
cor()
sum()
max()
min()
mean()
summary()

seq(from, to, by)
rep(x, each, times)

v = seq(-3,3)
v
tf<- v >0
tf

c1&c2 : intersection("and")
c1|c2 : union("or")
!c1 : negation of c1

# NA : not available
x = c(1:3, NA)
is.na(x)

v = c('a', 'b', 'c', 'd', 'e')
J = c(1,3,5)

v[1,3,5] (X)
v[c(1,3,5)]
v[J]
v[1:3]
v[2:length(v)]

v[-J]
v[-1]
v[-length(v)]

v <- c('a', 'b', 'c', 'd', 'e')
L <- c(TRUE, FALSE, TRUE, FALSE, TRUE)
v[L]
x = seq(-3,3)
x >= 0
x[x>=0]

x = seq(-1,1)
names(x) = c("N1", "N2", "N3")
x[c("N1", "N3")]

z =seq(1,4)
z[1] = 0
z[z<=2] = 10

w = c(1:3, NA, NA)
w[is.na(w)] <- 0

Exercise
#1
Eric = c(Math = 90, Physics = 85, Biology = NA)

#2
mean(Eric[!is.na(Eric)])

#3
mean(Eric[c("Math", "Physics")])


2. Matrix

x = matrix(0, nrow = 3, ncol = 4)
nrow(x)
ncol(x)
dim(x)

Y = matrix(1:12, nrow=3, ncol=4)
Y = matrix(1:12, nrow=3, ncol=4, byrow=TRUE)

x = 1:15
dim(x) = c(3,5)
z = x[1:2, 3:4]

Y = matrix(1:6, nrow=3, ncol=2)
r = matrix(1:6, nrow=3, ncol=2)

Y*r (그냥 인자들의 곱)
Y%*%r (오류)

t(r)
Y%*%t(r)

diag(Y%*%t(r)) : diagonal vector 반환

## aaply과도 같다!!
Extracting statistics from the rows or columns of a matrix
If X is a matrix, apply(X,1,f) is the result of applying f to each row of X; apply(X,2,f) to the columns

Y = matrix(1:12, nrow=3, ncol=4)
apply(Y,1,min)
apply(Y,2,mean)

Y = matrix(1:12, nrow=3, ncol=4)
colnames(Y) = c("X1", "X2", "X3", "X4")

Example
Y = matrix(1:12, nrow=3, ncol=4)
rownames(Y) = c("obs1", "obs2", "obs3")
colnames(Y) = c("var1", "var2", "var3", "var4")

max_Y = apply(Y,2,max)
names(max_Y) = paste("max", colnames(Y), sep="_")




3. Data.frame

BMI <- data.frame (
        gender = c("Male", "Male", "Female"),
        height = c(152, 171.5, 165),
        weight = c(81, 93, 78)
)

BMI[1,2]
names(BMI)
rownames(BMI) = c("Eric", "James", "Babel")

str() : display the internal structure of an R object

BMI$Height : components as vectors

BMI$age = c(42,38,26) # 새로운 컴포넌트 추가

tel = c("SK", "KT", "LG")
BMI = cbind(BMI, tel) # 새로운 컴포넌트 추가 다른 방법

BMI = BMI[1:3, 1:3]
add_row = data.frame(gender="Male", height=160, weight=50)  #데이터 종류 다르니까 df
rownames(add_row) = "Sanchez"
BMI = rbind(BMI, add_row)

BMI[5,] = c("Male", 170, 60)
rownames(BMI)[5] = "Yoon"

#정말 중요한 함수!!!
ddply(ToothGrowth, c("supp", "dose"), function(sub){ data.frame(length=mean(sub$len))})

4.
vec_n = c(1:10)
vec_c = c("Eric", "James")
mat = matrix(1:10, nrow=5, ncol=2)
df = data.frame(gender = c("Male", "Male", "Female"),
                height = c(152, 171.5, 165),
                weight = c(81,93,78)
                )
lst_ex = list(vec_n, vec_c, mat, df)

lst_ex[i] : i번째 object째로 반환
lst_ex[[i]] : i번째 object 내용물 반환


Lst = list(
      name = "Joe",
      height = 182,
      no.children = 3,
      child.ages = c(5,7,10)
)

Lst[[2]]
Lst[["height"]]
Lst$height
