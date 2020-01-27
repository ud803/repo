# melt함수는 테이블,데이터프레임을 녹여낸다.
hec <- melt(hec, value.name = "count")

# 기준 정의에 따라 구별
hw <-heightweight
hw$weightGroup <- cut(hw$weightLb, breaks = c(-Inf, 100, Inf), labels = c("< 100", ">= 100"))



# 거품그래프(balloon plot)
hec <- HairEyeColor[,,"Male"] + HairEyeColor[,,"Female"]
hec <- melt(hec, value.name = "count")
ggplot(hec, aes(x = Eye, y = Hair)) + geom_point(aes(size = count), shape = 21, color = "black", fill = "cornsilk") +
  scale_size_area(max_size = 20, guide = FALSE) +
  geom_text(aes(y = as.numeric(Hair)-sqrt(count)/22, label = count), vjust = 1, color = "grey60", size = 4)


# 산점도 행렬 만들기
c2009 <- subset(countries, Year == 2009, select = c(Name, GDP, laborrate, healthexp, infmortality))
plot(c2009[,2:5])


# 이산형 변수값에 따른 구분
tg <- ddply(ToothGrowth, c("supp", "dose"), summarize, length = mean(len))
# 색상으로 구분
ggplot(tg, aes(x = dose, y = length, color = supp)) + geom_line()
ggplot(tg, aes(x = factor(dose), y = length, color = supp, group = supp)) + geom_line() #group이 없으면 데이터를 어떻게 묶을지 모른다!!


# 두 선이 겹칠때 하나의 선을 옆으로 이동시켜 표현
ggplot(tg, aes(x = dose, y = length, shape = supp)) + geom_line(position = position_dodge(0.1)) +
  geom_point(position = position_dodge(0.1), size = 4)


# 선 형태 바꾸기 [ linetype ]
ggplot(tg, aes(x = dose, y = length, color = supp)) + geom_line(linetype = "dashed") +
  geom_point(shape = 22, size = 3, fill = "white")

# 점 형태 바꾸기 [ shape ]
ggplot(tg, aes(x = dose, y = length, fill = supp)) + geom_line(position = pd) +
  geom_point(shape = 21, size = 5, position = pd) +
  scale_fill_manual(values = c("black", "white"))



# sample Data
sunspotyear <- data.frame(Year     = as.numeric(time(sunspot.year)),
                          Sunspots = as.numeric(sunspot.year))
# 음영 영역 그래프 그리기
ggplot(sunspotyear, aes(x = Year, y = Sunspots)) + geom_area()
    #geom_area는 점 밑의 부분을 색칠해준다.
    #geom_ribbon은 지정된 영역을 색칠
# 음영 투명도 설정하기 [ alpha ]
ggplot(sunspotyear, aes(x = Year, y = Sunspots)) + geom_area(color = "black", fill = "blue", alpha = 0.5)


# 누적 영역 그래프 그리기
ggplot(uspopage, aes(x = Year, y = Thousands, fill = AgeGroup)) + geom_area()


# 영역색상 그라데이션 넣기
ggplot(uspopage, aes(x = Year, y = Thousands, fill = AgeGroup)) + geom_area(color = "black", size = 0.2, alpha = 0.4) +
  scale_fill_brewer(palette = "Blues", breaks = rev(levels(uspopage$AgeGroup)))


# 데이터 순서정렬하기 & 양쪽 테두리 지우기
ggplot(uspopage, aes(x = Year, y = Thousands, fill = AgeGroup, order = desc(AgeGroup))) +
  geom_area(color = NA, alpha = 0.4) + scale_fill_brewer(palette = "Blues") + geom_line(position = "stack", size = 0.2)


#  비율 누적 영역 그래프 그리기
uspopage_prop <- ddply(uspopage, "Year", transform, Percent = Thousands / sum(Thousands) * 100)
ggplot(uspopage_prop, aes(x = Year, y = Percent, fill = AgeGroup)) + geom_area(color = "black", size = 0.2, alpha = 0.4) +
  scale_fill_brewer(palette = "Blues", breaks = rev(levels(uspopage$AgeGroup)))


# 그래프에 신뢰 영역 추가하기
clim <- subset(climate, Source == "Berkeley", select = c("Year", "Anomaly10y", "Unc10y"))
# 신뢰영역 음영으로 표현
ggplot(clim, aes(x = Year, y = Anomaly10y)) +
  geom_ribbon(aes(ymin = Anomaly10y - Unc10y, ymax = Anomaly10y + Unc10y), alpha = 0.2) + geom_line()


# 신뢰영역 점선으로 표현
ggplot(clim, aes(x = Year, y = Anomaly10y)) +
  geom_line(aes(y = Anomaly10y - Unc10y), linetype = "dotted") +
  geom_line(aes(y = Anomaly10y + Unc10y), linetype = "dotted") +
  geom_line()
