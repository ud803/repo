### geom_bar : 비율을 표현한다 ###
###            stat ="count" 를 default로 사용한다. (count 센다는 말)
### geom_col : 값을 표현한다  ###
###            stat ="identity"를 사용한다. (값을 그대로 쓴다는 말)


barplot(BOD$demand, names.arg = BOD$Time)
ggplot(mpg, aes(class)) + geom_bar()
ggplot(mpg, aes(class)) + geom_bar(aes(weight=time)) # 가중치를 두어 그림


barplot(table(mtcars$cyl))
ggplot(pg_mean, aes(x= group, y= weight)) + geom_bar(stat="identity", fill="lightblue", colour = "black")

# x값을 숫자로 인식
ggplot(BOD, aes(x = Time, y = demand)) + geom_bar(stat = "identity")
ggplot(BOD, aes(x = factor(Time), y = demand)) + geom_bar(stat = "identity")

# 이 둘의 차이는 x값을 연속적으로 보느냐, 아니냐의 차이

# x값은 요인으로 변환
qplot(as.factor(BOD$Time), BOD$demand, geom="bar", stat="identity")
ggplot(BOD, aes(x = factor(Time), y=demand)) + geom_bar(stat = "identity")


# 막대 색상 채우기/테두리 설정하기 (fill : 채우기/ colour(or color) : 테두리)
ggplot(pg_mean, aes(x = group, y = weight)) + geom_bar(stat = "identity", fill = "lightblue", colour = "black")

# 막대 묶어서 표현하기(나누어 표현하고 싶은 변수를 색상으로 지정)
# dodge는 막대를 분리한다!
ggplot(cabbage_exp, aes(x = Date, y = Weight, fill = Cultivar)) + geom_bar(stat = "identity", position = "dodge") + scale_fill_brewer(palette = "Pastel1")
