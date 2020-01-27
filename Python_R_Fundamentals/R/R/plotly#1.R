plotlyly는 웹 기반 시각화 툴이다.

기본적으로 크기가 매우 큰 데이터 프레임들은
dplyr 이라는 패키지가 제공하는 tibble (혹은 tbl_df)함수에 의해 또 다른 데이터 프레임으로 변한다. 이는 데이터 프레임을 실수로 호출했을 때 전체 행을 호출하는 것을 막아준다.

plot_ly()   : transforms data into a plotly object
ggplotly()  : transforms a ggplot object into a plotly object

library(plotly)
txhousing
