## ggplot Basic ##

1. data 인자를 사용해서 특정 데이터프레임과 그래프 플롯을 묶는다.
ggplot(data = dataframe)

2. 미학요소(aesthetics, aes)를 정의해서 데이터의 변수와 그래프 플롯의 축, 형태 등을 매핑한다.
ggplot(data=df, aes(x=xcomp, y=ycomp))

3. geom를 추가한다. 데이터의 그래픽 표현(점, 선, 막대 등)
ggplot(data=df, aes(x=xcomp, y=ycomp)) + geom_point()

이러한 방식 덕분에 이미 존재하는 ggplot 객체를 + 연산을 이용하여 쉽게 변형시킬 수 있다.

ggplot() + geom_point(alpha =0.1, aes(color=species_id))


ggplot() + geom_boxplot() + geom_gitter(alpha=0.3, color="tomato")
