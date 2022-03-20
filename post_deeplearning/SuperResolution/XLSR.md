---
sort: 7
---

# XLSR  
Extremely Lightweight Quantization Robust Real-Time Single-Image Super Resolution for Mobile Devices  

## 1. 개요  
SR은 컴퓨터 비전분야에서 오래전부터 연구되어왔고 최근에는 SRCNN을 시작으로 딥러닝 기술을 이용하여 성능이 비약적으로 상승했다. 현재까지 GAN을 비롯한 여러가지 학습기법을 이용한 모델이 나오고있지만, 여전히 int8 및 모바일기기에서의 성능은 많이 제한되는 상황이다.  
이를 해결하기위해 자체적인 양자화 및 Cliped ReLU 방식등을 사용한 XLSR 모델을 제안한다.  

## 2. 관련 연구  
모바일 장치에 딥러닝 모델을 성공적으로 배포하면 더 넓은 응용 영역이 열리고 학술적 기여와 함께 사용성이 증가한다.  

### 1. Network  
딥러닝 구축에 초점을 맞춘 하드웨어는 이후 Qualcomm과 Arm의 노력으로 시작되었으며, 여러 공급업체의 전문 AI 실리콘으로 지속되었다. 그리고 모바일 구축에 초점을 맞춘 많은 연구자들은 모바일 친화적 모델에 대한 다양한 아이디어를 생각해낸다.
대표적으로 MobileNetV1은 depthwise separable convolutions을 사용하며 SqueezeNet보다 더 나은 성능을 달성한다. ShuffleNet은 group convolutions and channel shuffling operator를 활용하여 더 가벼우면서 성능을 더욱 향상시킨다. ShuffleNet과 유사하게 DeepRoots는 channel shuffling 대신 1x1 convolution의 사용을 제안한다. 얼굴 확인을 위해 MobileFaceNets는 depthwise separable convolutions과 bottleneck layers를 사용하여 애플리케이션별 경량 네트워크를 구축한다. 이미지 초고해상도의 경우, IMDN은 channel splitting을 사용하여 더 가벼운 네트워크를 구축한다.  

### 2. Quantize, Pruning  
연구자들은 네트워크 자체에 초점을 맞추는 것 외에도, 양자화에도 초점을 맞추고 있다. 왜냐하면 대부분의 경우 양자화는 옵션이 아니라 하드웨어 성능상 필수이기 때문이다. Knowledge Distillation등을 사용하며, 더 나아가 양자화를 결합하는 방법도 제안되었다.. 경량 모델을 구축하는 동안 pretrained된 복합적인 모델을 사용하는 또 다른 방법론은 channel sparsification and pruning이다.
이 방법론을 사용해 필터에서 불필요한 채널을 제거함으로써 복잡한 모델을 얇게 만들 수 있다.  

### 3. Summary  
앞에서 언급한 방법의 대부분은 SISR이 아닌 다른 영역에 대해 설계되었지만 실시간 성능 SISR 모델을 구축하는 동안 여전히 매우 유용할 수 있다. 모바일 장치의 성능 딥러닝 방법을 실행/설계하는 방법론은 다음과 같이 요약할 수 있다.  

- Hand-Designed Architectures
- Efficient Building Block Design
- Network Pruning / Sparsification
- Network Quantization
- Network Architecture Search (NAS) • Knowledge Distillation  

본 논문에서는 Mobile AI 2021 Real-Time 단일 이미지 초해상도 챌린지에 제출하는 동안 첫 번째와 두 번째 방법론을 따랐다. 이 접근법의 이유와 동기는 고려해야 할 플랫폼(Synaptics Dolphin NPU)의 다양한 하드웨어 한계가 있었기 때문이다. 또한, 과제는 모델의 전체 int8 양자화를 필요로 했다. 전체 uint8 양자화 요구 사항은 당면한 모델이 적절하게 설계/훈련/양자화되지 않은 경우 SISR 문제가 양자화 작업에 의해 심각하게 영향을 받기 때문에 문제를 더욱 복잡하게 만든다.

## 3. Network Architecture  
본 문단에서는 제안된 네트워크의 세부 사항과 설계 아이디어에 대한 동기를 설명하면서 이를 문헌 및 하드웨어 한계와 연결한다. 이전에 언급했듯이, 본 논문에서는 Backbone 모델없이 제안된 아카이브를 설계했고, SISR 문제를 위해 효율적인 구성 요소를 채택했다.  

### 1. Building Block Selection  
Group Convolutions은 AlexNet에서 처음 사용되지만 GPU 하드웨어 제한으로 인해 이러한 방법론이 불가피하다. 잘 사용하면 Group Convolutions이 정확도를 높이는 동시에 계산 비용을 줄일 수 있다. 이러한 특성 때문에 모바일 중심 네트워크에서 자주 사용된다. Skip Connection과 함께 ResNet에서, 계단식 Channel shuffling이 있는 ShuffleNet에서, 계단식 1x1 Convolution이 있는 DeepRoots에서 사용된다.  

SISR 문제와 하드웨어 제한의 관점에서, reshape 및 transpose 작업이 최적화되지 않았기 때문에 channel shuffling을 사용하는 것은 불가능하다. Skip Connection과 residual 구조는 모델 수렴을 돕고 더 깊은 아키텍처를 허용하며 많은 최신 네트워크에 사용된다. 하지만 동시 다발적인 Skip connection들은 느리고 요소간 덧셈은 최적화 되지않았다. 따라서 ResNet block은 하드웨어어에 최적화 되지 않는다.  

반면에 channel shuffling 연산자가 1x1 convolution으로 완화되거나 Resnet 블록에서의 Skip Connection등의 특성을 이용하여 설계를 진행한다. Group convolution 대신 Depthwise convolution도 빌딩 블록의 후보가 될 수 있다. 그러나 depthwise convolution은 특별한 예방 조치 없이 사용할 경우 큰 정량화 오류를 일으킬 수 있으며, 이는 또한 실험에서 경험적으로 관찰한 결과다. 제안된 모델을 설계하는 동안 우리가 고려한 중요한 측면은 elementise wise operation이 배치 하드웨어에서 최적화되지 않았다는 것이다. 따라서 모든 addition, scaling 연산을 피하고 필요할 때 concatenation operation을 사용했으며 입력 데이터 scaling링 및 normalization를 사용하지 않았다.  
