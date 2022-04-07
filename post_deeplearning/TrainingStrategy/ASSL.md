---
sort: 7
---

# ASSL  
Aligned Structural Sparsity Learning  

## 1. Abstract  
Lightweight SR 모델의 대부분은 구조를 설계하는대 중점을 두고 중복된 파라미터를 제거하는데 관심이없다. KD 기법은 학습과정에서 상당한 메모리와 계산 자원을 요구한다. 이를 위해 Pruning 구조를 사용해야하는데 Residual 구조로 인하여 많은 어려움이 있다. 이를 해결하기 위한 Aligned Structured Sparsity Learning (ASSL)을 제안한다.  

## 2. Inctoduction  
Deep Neural CNN은 SRCNN을 시작으로 EDSR, RCAN등 여러가지 모델이 발표되었다. 그러나 이 모델들은 무거운 모델 크기로 인하여 사용대상이 되는 device에서 직접적으로 사용하지 못하는 결점이 있다. 즉, 별도의 NPU가 없다면 실제 동작이 힘들다는 것이다. CARN, IMDN등 다양한 Lightweight Model을 통해 해결하려 했지만 학습과정에서의 필요로하는 많은 컴퓨팅자원, Sparsity, 중복된 Convolution Layer를 무시하고 있다. 더 효과적이고, 적은 자원으로 사용이 가능한 Lightweight SR Model이 필요한 시점이다.  

Convolution 커널의 중복성과 모델의 복잡성을 줄이기 위해 Neural Network Pruning 기법을 사용한다. 대부분의 연구자들은 모델 가속을 위해 weight-element pruning 기법보다 filter pruning 기법을 선호하지만 Residual이라는 구조적인 문제가 있다. 이 문제를 해결하기 위해 ASSL을 제안한다.  

ASSL은 filter pruning기반 효과적인 regularization기법이다. 본 논문에서는 각각의 convolution layer 후에 normalization layer를 추가한다. 그리고 weight normalization의 파라미터 개수를 조절하기 위해 sparsity-inducing L2 regularization을 사용한다. Residual SR 모델의 pruning과정에서 중요한 문제는, 서로 다른 layer에서 sparsity structure가 정렬된다는 것이다. pruned filter를 서로 다른 later에 동일하게 적용하기 위하여 새로운 sparsity structure alignment regularization term을 제안한다.  