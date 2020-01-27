library(ggplot2)
library(gcookbook)
library(plyr)
library(reshape2)

#가장 기초적인 scatter plot
plot(mtcars$wt, mtcars$mpg)
qplot(mtcars$wt, mtcars$mpg)  #ggplot의 plot형태. 더 예쁨.
ggplot(mtcars, aes(x= wt, y=mpg)) + geom_point() #축을 정하고, 점을 찍음

ylim(0, max(BOD$demand)) #그래프의 ylim 설정해줌


# Scatter + Line Plot
plot(pressure$temperature, pressure$pressure, type = "l")
points(pressure$temperature, pressure$pressure)

qplot(temperature, pressure, data = pressure, geom = c("line", "point"))

ggplot(pressure, aes(x = temperature, y = pressure)) + geom_point() + geom_line()



#Line Plot
ggplot(BOD, aes(x = Time, y = demand)) + geom_line() + ylim(0, max(BOD$demand))
ggplot(BOD, aes(x = Time, y = demand)) + geom_line() + expand_limits(y = 0)
    #expand_limits(x=c(), y=) 현재 그래프에서 범위를 넓혀줌
    #expand_limits(y = c(0,10))

# 축을 추가하고, col=에 해당하는 자료에 따라 색을 구분
# 숫자 자료를 넣을 경우 범위에 따라 농도가 달라지는 그래프로 자동 변경
ggplot(tophit, aes(x= avg, y=name, col=lg)) + geom_point()

ggplot(heightweight, aes(x = ageYear, y= heightIn, color =sex)) +geom_point()

ggplot( aes(shape=sex, color=sex)) + geom_point() #모양, 색으로 구분

ggplot(heightweight, aes(x = ageYear, y = heightIn, color = sex, shape = sex)) + geom_point() + scale_shape_manual(values = c(1,2)) + scale_color_brewer(palette = "Set1") #scale_sape_manual은 모양 정하는 것

ggplot(data = yearly_counts, aes(x = year, y = n, group = species_id, colour = species_id)) + geom_line()

## 측면보기 Faceting!!
ggplot(data = yearly_counts, aes(x = year, y = n, group = species_id, colour = species_id)) + geom_line() + facet_wrap(~ species_id)



# 이산형 자료의 이동 적용 - overplotting 방지?
ggplot(ChickWeight, aes(x = Time, y = weight)) + geom_jitter()




# 특정 값에 특정 단어로 라벨 붙이기
sp + annotate("text", x = 4350, y = 5.4, label = "Canada") +
  annotate("text", x = 7400, y = 6.8, label = "USA")
# 데이터 값을 라벨로 붙이기
sp + geom_text(aes(label = Name), size = 4)
# 라벨의 위치를 데이터값보다 조금 크게 설정
sp + geom_text(aes(y = infmortality + 0.1, label = Name), size = 4, vjust = 0)

# 특정 값만 라벨 붙이기 ! 똑똑한 방법!
cdat <- subset(countries, Year == 2009 & healthexp > 2000)
cdat$Name1 <- cdat$Name
idx <- cdat$Name %in% c("Andorra", "France", "Canada")
cdat$Name1[!idx] <- NA
ggplot(cdat, aes(x = healthexp, y = infmortality)) + geom_point() + geom_text(aes(y = infmortality + 0.1, label = Name1), size = 4, vjust = 0)



# 정렬 / 격자 없애기 / 수평선 점선으로 바꾸기
ggplot(tophit, aes(x= avg, y=reorder(name,avg))) + geom_point(size=3) + theme_bw() + theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed"))

# x, y축 바꿔서 그치기 & 그래프 격자 없애기 & 수직선 점선으로 바꾸기 & X축 값 정의 및 회전
ggplot(tophit, aes(x = reorder(name, avg), y = avg)) + geom_point(size = 3) + theme_bw() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_line(color = "grey60", linetype = "dashed"))





# 적합된 회귀선 추가하기
sp <- ggplot(ChickWeight, aes(x = Time, y = weight))
sp + geom_point(color = "blue") + stat_smooth(method = lm, se = TRUE, color = "red")

#그룹 별 회귀선 추가하기
sps <- ggplot(heightweight, aes(x = ageYear, y = heightIn, color = sex)) +
  geom_point() + scale_color_brewer(palette = "Set1")
sps + geom_smooth()





#geom_segment(x, y, xend, yend) : 한 선분을 긋는다

# 격자 선이 그래프의 끝에서 끝까지 횡단하지 않고, 점까지만 가도록 표현 [ geom_segment ]
ggplot(tophit, aes(x = avg, y = name)) +
  geom_segment( aes(yend=name),xend = 0, color = "grey50") + geom_point(size = 3, aes(color = lg)) +
  scale_color_brewer(palette = "Set1", limits = c("NL", "AL")) + theme_bw() +
  theme(panel.grid.major.y = element_blank(), legend.position = c(1, 0.55), # 범례를 그래프 안쪽으로 옮김
        legend.justification = c(1, 0.5))




# 그룹 별 그래프 분할
ggplot(tophit, aes(x = avg, y = name)) +
geom_segment(aes(yend = name), xend = 0, color = "grey50") + geom_point(size = 5, aes(color = lg)) +
scale_color_brewer(palette = "Set1", limits = c("NL", "AL"), guide = FALSE) + theme_bw() +
theme(panel.grid.major.y = element_blank()) + facet_grid(lg~., scales = "free_y", space = "free_y")
  # facet_grid = 기준에 따라 그래프를 나눔
