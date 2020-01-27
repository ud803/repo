## dplyr 패키지 ##

install.packages(c("dplyr", "hflights"))
library(dplyr)
library(hflights)

hf = tbl_df(hflights)



## 1. filter ##
# filter(dataframe, cond1, cond2, cond3-1 | cond3-2)

## 2. arrange ##
# default is ascending order. use desc() otherwise.
# arrange(dataframe, Crit1, Crit2, Crit3)
# arrange(dataframe, Crit1, desc(Crit2))

## 3. select ##
# select(dataframe, col1, col2, col3)
# above is same as select(dataframe, col1:col3)
# select(hf, -(Year:DayOfWeek)) :

## 4. mutate ##
# similar to transform
# mutate(dataframe, newColName = newColFunc, newColName2 = newColName1/2)
# can use the name spontaneously

## 5. summarise ##
# summarise(dataframe, delay = mean(DepDelay, na.rm = TRUE))

## 6. group_by ##
# group_by(dataframe, colName)

planes = group_by(hflights_df, TailNum)
delay = summarise(planes, count=n(), dist=mean(Distance, na.rm=T), delay=mean(ArrDelay, na.rm=T))
delay = filter(delay, count>20, dist<2000)
ggplot(delay, aes(dist, delay)) + geom_point(aes(size=count), alpha=0.5) + geom_smooth() + scale_size_area()


## 7. chain ##
# chain() or %>%
a1 <- group_by(hflights, Year, Month, DayofMonth)
a2 <- select(a1, Year:DayofMonth, ArrDelay, DepDelay)
a3 <- summarise(a2, arr = mean(ArrDelay, na.rm = TRUE), dep = mean(DepDelay, +na.rm = TRUE))
a4 <- filter(a3, arr > 30 | dep > 30)

# chain function can abbreviate above commands as follows
hf %>% group_by(Year, Month, DayofMonth) %>% summarise(arr=mean(ArrDelay, na.rm=T), dep=mean(DepDelay, na.rm=T))%>% filter(arr>30 | dep>30)

hff = hflights %>% filter( TailNum == c(sample(names(table(hflights$TailNum)), 4, replace=F))) %>% select( TailNum, ActualElapsedTime)







### aaply, adply, alply ##
### Split array, apply function and return in an array/ dataframe / list

aaply(ozone, 1, mean) # 1 row, 2 column, c(1,2) by row&col
aaply(ozone, 1, mean, .drop=FALSE)
aaply(ozone, 3, mean)
aaply(ozone, c(1,2), mean)
aaply(ozone, 3, each(min,max)) # 여러 함수는 each로 감싼다.


### filter, slice ##
### filter는 조건에 의한 선별, slice는 위치에 의한 선별
# filter(dataframe, filter con1, filter con2 ...)
# slice(dataframe, from, to)

filter(Cars93_1, Type==c("Compact"), Max.Price <=20 )
slice(Cars93_1, 6:10) #6~10번째 행의 데이터를 선별


### arrange, order, sort, rank, reorder ##
# sort(a, decreasing =T)  / 벡터
# rank(a), rank(-a)  / 순위의 색인
# order(a) / 순위의 색인
# reorder(a) /
# arrange(dataframe, order criterion1, ...)
# 모두 디폴트는 오름차순
# order 은 -가 내림차순

만약 a = c(15, 1, 10, 3) 이라면
order(a) = a를 가지고 a의 인덱스로 순위를 매긴 것  = c(2,4,3,1)
rank(a)  = a를 그대로 둔 채로 순위를 매긴 것      = c(4,1,3,2)

arrange(Cars93_1, desc(MPG.highway), asc(Max.price))

Cars93[ order(-Cars93_1$MPG.highway, Cars93_1$Max.Price), ]


### select ##
### 선별하고자 하는 변수 추출
# select(dataframe, VAR1, VAR2 , ...)

select(Cars93_1, -(Manufacturer:Price))
select(Cars93_1, Manufacturer, Max.Price, MPG.highway)
select(Cars93_1, Manufacturer:Price)
select(Cars93_1, 1:5)

select(Cars93_1, starts_with("MPG"))
select(Cars93_1, ends_with("xx_name"))
select(Cars93_1, contains("xx_string"))
select(Cars93_1, matches(".xx_string."))

subset(Cars93_1, select=c(Manufacturer:Price))


### rename() ##
### 이름 변경하기
# rename(dataframe, new_var1 = old_var1, ...)

Cars93_2 = rename(Cars93_1, New=Old, New2=Old2)
