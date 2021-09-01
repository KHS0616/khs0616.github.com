---
sort: 1
---

# StyleGAN  
A Style-Based Generator Architecture for Generative Adversarial Networks  
![StyleGAN 메인 이미지](../../static/StyleGAN-main.png)
## 1. 개요
PGGAN 구조에서 Style Transfer 개념을 적용하여 Generative Adversarial Networks을 위한 alternative generative architecture을 제안한다.  
새로운 기능으로는 자동으로 학습되고 높은 수준의 속성(pose, identity 등)과 생성된 이미지(주근깨, 머리카락 등)의 Stochastic variation(확률적 변동)을 분리하고 직관적으로 scale별 제어를 가능하게 한다.  
새로운 Generator는 traditional distribution quality metrics 측면에서 SOTA을 향상시키고 더 나은 interpolation properties을 입증하며 latent factor of variation을 더 잘 disentangled 하게 한다.  
interpolation quality와 disentanglement을 정량화(quantify) 하기 위해 모든 generator 아키텍처에 사용할 수 있는 두가지 새로운 자동화된 방법을 제안한다.

## 2. 기존 연구 및 문제점
### 1. PGGAN  
PGGAN은 점진적으로 낮은 해상도부터 높은 해상도까지 차근차근 점진적으로 생성하는 대표적인 생성 모델로 StyleGAN의 base가 되는 모델로 latent vector z가 Normalize를 거쳐 모델에 바로 입력이 되는 형태로 학습이 진행된다. 하지만 이렇게 z가 Generator에 바로 입력이 들어가면 GAN은 latent space가 무조건 학습 데이터 셋의 확률 분포와 비슷한 형태로 만들어 지도록 학습을 하게되면서 entangle하게 만들어 지게 된다.  
GAN을 이용하여 이미지를 생성할 때 latent variable 기반의 모델은 random noise를 입력으로 사용하게 된다. 이 때 entangle하게 만들어지는 가장 큰 단점이 부각된다. 억지로 끼워맞추기 하는 형식으로 mapping이 이루어지다보니 wrapping이 발생하게 된다. wrapping이 발생하게 되면 예측할 수 없을 정도로 급진적으로 변하게 되는 특징이 존재한다.  

## 3. StyleGAN
### 1. Disentangle
![StyleGAN-Disentangle](../../static/StyleGAN-entangle.png)  
이미지에 style을 적용하여 색다른 이미지를 생성하고 싶은데 Generator에 latent vector z가 바로 입력되기 때문에 entangle하게 되어서 불가능 하다는 단점이 있었다.  
이를 해결하기 위해 각각 다른 style을 여러 scale에 추가해서 학습시키는 방식을 도입했다.  
하지만 근본적인 문제인 latent vector z가 직접적으로 입력이 되는 문제는 해결되지 않았고 이를 해결하기위해 Mapping Network를 사용하여 mapping된 W를 각 scale에 브로드캐스트 하는 방식으로 해결하였다.  
이를 통해 매핑된 latent space의 W는 정확하지는 않지만 학습 데이터 셋의 확률 분포와 비슷한 모양으로 매핑이되었고 disentangle하게 된다.

### 2. AdaIN  
![AdaIN 수식](../../static/StyleGAN-AdaIN.png)  
Neural Network에서 각 layer를 지나가며 scale, variance의 변화가 생기는 일이 빈번하게 발생하며 이는 점점 학습이 불안정 해지는 현상을 발생한다. 따라서 이를 방지하기 위해 Batch Normalization 방법같은 normalization 기법을 각 layer에 사용하므로써 해결한다.  
StyleGAN에서는 Mapping network를 거쳐서 나온 W가 latent vector의 style로 각 scale을 담당하는 layer에 입력으로 들어가게 된다. style에 영향을 주면서 동시에 normalization 해주는 방법으로 AdaIN을 사용하게 된다.  
수식을 해석하면 x라는 결과에서 평균을 빼고 표준편차로 나누는 모습이다. 이러한 결과에 y 
s,i 와 y b, i를 Affine Transformation을 거쳐서 shape을 맞추고 style을 입혀주게 된다. 여기서 y s,i는 linear cofficient를 의미한다.  
전체 과정을 정리하면 W가 AdaIN을 통해 Style을 입혀야 하는데 Shape이 안맞기 때문에 Affine Transform을 거쳐 맞춰준다. Style을 입히는 개념은 y s,i를 곱하고 y b,i를 더하는 과정을 의미한다. AdaIN에서 정규화를 할 때 한번에 하나씩만 W가 기여하게 되므로 하나의 style이 각각의 style에서만 영향을 끼칠 수 있도록 분리해주는 효과를 갖는다.

### 3. Generator Network  
![StyleGAN-Generator 구조](../../static/StyleGAN-Generator.png)
StyleGAN의 Generator Network 구조는 기본적으로 PGGAN과 동일하게 점진적으로 낮은 해상도부터 높은 해상도 까지 차근차근 Feature Map을 생성하는 구조다.  기존 PGGAN과 다른점은 latent vector z를 바로 사용하지 않고 Mapping Network를 사용하여 W code를 생성하고 이를 각 scale별로 브로드 캐스트한다는 점이다.  
또한 Style의 편향 및 Style의 localization을 위해 mixing regularization를 사용 하는데 이 방식은 두 개의 임의의 latent code를 사용하는 방식이다. StyleGAN에서는 두 개의 latent code z를 이용하여 w1, w2가 style을 제어하도록 하고 하나는 AdaIN이전, 하나는 AdaIN연산에 입력된다.  
최종적으로는 1024*1024 이미지가 생성된다. (총 18개 layer)

### 4. Discriminator Network  
Discriminator 구조는 Generator구조와는 반대로 높은해상도에서 점진적으로 낮은해상도로 변하며 판별을 한다.