---
sort: 5
---

# InductiveBias    

## 1. Inductive Bias  
ML/DL 모델이 갖는 Generalization Problem은 Brittle, Spurious이 있다. 모델이 주어진 데이터에 대해서 잘 일반화한 것인지, 혹은 주어진 데이터에만 잘 맞게 된 것인지 모르기 때문에 발생하는 문제이다. 이러한 문제를 해결하기 위한 것이 바로 Inductive Bias이다. Inductive Bias는 주어지지 않은 입력의 출력을 예측하는 것이다. 일반화의 성능을 높이기 위해서 만약의 상황에 대한 추가적인 가정(Additional Assumptions)이라고 할 수 있다.  

- Brittle : 같은 분류상의 데이터도 조금만 변하면 모델의 정확도가 낮아지는 문제  
- Spurious : 데이터 분석을 제대로 하지 못해 모델의 결과에 큰 편향이 생기는 문제  

위 두가지 이유로 모델이 학습하는 과정에서 학습 데이터이외의 데이터들까지도 정확한 출력에 가까워지도록 추측하기 위해서는 추가적인 가정이 필요하다. 대부분의 일반화가 잘 된 모델은 일반화하기 위해 만들어진 가정인 Inductive Bias의 유형을 가지고 있다. Inductive Bias는 보지 못한 데이터에 대해서도 귀납적 추론이 가능하도록 하는 알고리즘이 가지고 있는 가정의 집합이라고 할 수 있다.  

## 2. On Deep Learning  
Computer Vision Tasks들을 다루는 모델들은 모두 CNN을 사용한다. CNN은 이미지를 다루기에 적합한 Inductive Bias를 갖고 있다.  

FCN(Fully Connected Neural Network)은 가장 일반적인 블록의 형태로, 가중치와 편향으로 각 층의 요소들이 서로 모두 연결되어 있고, 모든 입력의 요소가 어떤 출력 요소던지 영향을 미칠 수 있다. 이러한 이유로 Inductive Bias가 매우 약하다.  

CNN(Convolutional Neural Network)은 Convolution Filter가 입력을 Window Sliding 한다. CNN은 FCN과 비교했을때, Entities 간의 관계가 약하다는 차이점이 있다. 이러한 특징으로 CNN은 FCN과 다르게 Locality & Translation Invariance의 Relational Inductive Biases를 갖는다.  

- Localitiy : 입력에서 각 Entities간의 관계가 서로 가까운 요소들에 존재한다는 것  
- Translation Invariance : 입력과 동일하게 계속해서 관계가 유지된다는 것  

이는 어떤 특징을 가지는 요소들이 서로 모여있는지가 중요한 문제에서 좋은 성능을 보여준다는 것을 의미한다.  

RNN에서는 CNN의 Locality & Translation Invariance와 유사한 개념으로 Sequential & Temporal Invariance의 Relational Inductive Biases를 갖는다. GNN(Graph Neural Network)은 이러한 개념을 그래프로 가져간 것으로, Permutational Invarianced의 Relational Inductive Biases를 갖는다.  

- Sequential : 입력이 시계열의 특징을 갖는다고 가정  
- Temporal Invariance : 동일한 순서로 입력이 들어오면 출력도 동일하다는 것  

## 3. Vision Transformer  
Transformer는 NLP분야 뿐만아니라 최근에는 Computer Vision에서도 강력한 성능을 보여주며, 주목을 받고 있다. CNN은 이미지가 지역적으로 얻을 정보가 많다는 것을 가정하고 만들어진 모델이다. 반면에, Transformer는 Positional Embedding과 Self-Attention을 사용해 모든 정보를 활용한다. 즉, Transformer는 CNN에 비해 Inductive Biases가 부족하다고 볼 수 있다. 결과적으로 Global한 정보가 필요한 경우에는 Transformer가 좋지만, Inductive Biases가 잘 맞는 이미지가 지역적으로 얻을 정보가 많은 경우에는 CNN이 적합하다.  