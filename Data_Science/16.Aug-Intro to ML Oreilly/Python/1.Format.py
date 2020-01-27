print("Sammy ate {0:.3f} {pr} of a {1:16}!".format(75.765367, "pizza", pr="percent"))

# {field_name:conversion} 형태에서 field_name은 인덱스를, conversion은 자료형을 나타냄
# .3f는 float으로 나타내고, 소수점 셋째 자리까지 나타내라는 뜻, 반올림함 / d는 정수형
# {1:16}은 padding을 만든다. 텍스트는 왼쪽 정렬, 숫자는 오른쪽 정렬
# {1:^16}, < 왼쪽 정렬, ^ 가운데 정렬, > 오른쪽 정렬
# {:*^20s}".format("Sammy") 빈 공백을 *로 채워서 총 20칸을 구성, 가운데 정렬

for i in range(3,13) :
	print("{:6d} {:6d} {:6d}".format(i, i*i, i*i*i))

str2 = "boy"

print("I am a {}".format(str2))



'''
파이썬을 부분실행하고 싶을 땐
sed -n '13,15p' filename | python
을 사용한다!
'''

'''
[	startpoint	:	endpoint	:	step	]

[:,:]
앞의 :는 1차원 배열을
뒤의 :는 2차원 배열을 의미
'''
