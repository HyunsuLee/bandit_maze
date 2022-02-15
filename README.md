# bandit_maze
test maze environment like two-armed bandit

* 2개의 상태가 있고, 왼쪽으로 가면 -10 보상(종료), 오른쪽으로 가면 bandit arm을 pull 할 수 있는데, 평균 -1 보상이며 계속 pull할 수 있다. 
* 당연히 agent는 평균 -1인 step을 계속 밟으면 왼쪽 -10보상의 종료보다 안 좋은 선택이다. 약간의 학습을 거치면 왼쪽 -10보상 종료를 먼저 선택하게 된다.