from sklearn.linear_model import Perceptron

# 논리적 AND 연산 샘플과 정답이다. 
X = [[0,0],[0,1],[1,0],[1,1]]		# 항상 2차원 배열이어야 한다. 
y = [0, 0, 0, 1]

# 퍼셉트론을 생성한다. tol는 종료 조건이다. random_state는 난수의 시드이다.
clf = Perceptron(tol=1e-3, random_state=0)

# 학습을 수행한다. 
clf.fit(X, y)

# 테스트를 수행한다. 
print(clf.predict(X))