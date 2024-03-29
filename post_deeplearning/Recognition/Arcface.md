---
sort: 1
---

# ArcFace  
ArcFace: Additive Angular Margin Loss for Deep Face Recognition  

## 1. Abstract   
컨볼루션 신경망은 차별적 특징을 학습하는 큰 수용량으로 인해 최근 몇 년 동안 얼굴 인식의 성능을 크게 향상했다. Softmax loss의의 차별적 힘을 강화하기 위해, multiplicative angular margin와 additive cosine margin는 각각 손실 함수에 angular margin와 cosine margine을 통합한다. 본 논문에서는 지금까지 제안된 supervisor signal보다 기하학적 해석이 우수한 새로운 supervisor signal인 additive angular margin(ArcFace)을 제안한다. 구체적으로, 제안된 ArcFace cos(α + m)는 L2 normalised weights와 features에 기초한 angular(arc) 공간에서 decision boundary를 직접 최대화한다. multiplicative angular margin 비용(m²)과 additive cosine margin 비용 β - m에 비해, ArcFace는 보다 차별적인 심층 특징을 얻을 수 있다. 또한 심층 얼굴 인식 문제에서 네트워크 설정과 데이터 개선의 중요성을 강조한다. LFW, CFP 및 AgeDB라는 몇 가지 관련 얼굴 인식 벤치마크에 대한 광범위한 실험은 제안된 ArcFace의 효과를 입증한다.  가장 중요한 것은 메가페이스 챌린지에서 최첨단 성능을 완전히 재현할 수 있다는 점이다. 데이터, 모델 및 교육/테스트 코드를 공개한다.  

## 2. Introduction  
Deep convolutional network embedding을 통한 얼굴 표현은 face verification, face clustering, and face recognition을 위한 최첨단 방법으로 간주된다. 심층 컨볼루션 네트워크는 일반적으로 pose normalisation step 후 얼굴 이미지를 embedding feature vector에 매핑하여 동일한 사람의 피처에는 작은 거리가 있고 다른 사람의 feature에는 상당한 거리가 있도록 하는 역할을 한다. 심층 컨볼루션 네트워크 임베딩에 의한 다양한 얼굴 인식 접근 방식은 세 가지 주요 속성에 따라 다르다.  

첫 번째 속성은 모델을 교육하는 데 사용되는 학습 데이터다. VGG-Face, VGG2-Face, CAISAWebFace, UMDFaces, MS-Celeb-1M 및 MegaFace와 같은 training data의 ID 수는 수 천에서 50만 개에 이른다. MS-Celeb-1M과 MegaFace는 상당한 수의 identities를 가지고 있지만 annotation noises and long tail distributions로 어려움을 겪고 있다. 그에 비해 private training data of Google는 심지어 수백만 개의 identities를 가지고 있다. 얼굴인식 FRVT의 최근 실적보고서를 보면 알 수 있듯이, 중국 출신의 스타트업 Yitu는 18억 개의 얼굴 비공개 영상을 기준으로 1위를 차지하고 있다. Training Data 규모의 순서 차이로 인해 업계의 얼굴 인식 모델이 학계의 모델보다 훨씬 더 나은 성능을 보인다. 훈련 데이터의 차이로 인해 일부 심층 얼굴 인식 결과는 완전히 재현되지 않는다.  
 
두 번째 속성은 네트워크 아키텍처 및 설정이다. ResNet 및 Inception-ResNet과 같은 대용량 심층 컨볼루션 네트워크는 VGG 네트워크 및 Google Inception V1 네트워크에 비해 더 나은 성능을 얻을 수 있다. 심층 얼굴 인식의 다른 응용 프로그램은 속도와 정확도 사이의 상이한 절충을 선호한다. 모바일 장치의 얼굴 확인을 위해서는 원활한 고객 만족을 위해 실시간 실행 속도와 소형 모델 크기가 필수적이다. 10억 수준의 보안 시스템에서 높은 정확도는 효율성만큼이나 중요하다.  

세 번째 속성은 손실 함수의 설계이다.  
(1) Euclidean margin based loss
Softmax 분류 계층은 알려진 ID 집합에 대해 훈련된 다음 feature vector는 네트워크의 중간 계층에서 가져와 훈련에 사용되는 ID 집합을 넘어 인식을 일반화하는 데 사용된다. Centre loss Range loss, Marginal loss는 내부 분산을 압축하거나 거리 간을 확대하여 인식률을 향상하도록 추가 페널티를 추가하지만, 모든 손실은 여전히 소프트맥스를 결합하여 인식 모델을 훈련시킨다. 그러나 분류 기반 방법은 ID 수가 백만 수준으로 증가하면 분류 계층에서 GPU 메모리 소모가 커지며, 각 ID에 대해 균형 있고 충분한 훈련 데이터를 선호한다. contrastive loss, the Triplet loss는 pair training strategy를 사용한다. contrastive loss 함수는 양의 쌍과 음의 쌍으로 구성된다. 손실 함수의 기울기는 양의 쌍을 끌어당기고 음의 쌍을 밀어낸다. Triplet loss은 앵커와 양성 샘플 사이의 거리를 최소화하고 앵커와 다른 ID로부터의 음성 샘플 사이의 거리를 최대화한다. 그러나 대비 손실 및 삼중수소의 훈련 절차는 효과적인 훈련 샘플 선택으로 인해 까다롭다.  

(2) 각도 및 코사인 마진 기준 손실  
기능 차별을 개선하기 위해 각 아이덴티티에 곱셈 각도 제약 조건을 추가하여 큰 여백 소프트맥스(LSOFTmax)를 제안했다. Sphere Face cos(m²)는 가중치 정규화를 통해 심층 얼굴 인식에 L-Softmax를 적용한다. 코사인 함수의 비 일원 성 때문에, Sphere Face에서 조각 별 함수를 적용하여 단조성을 보장한다. Sphere Face 훈련 중 Softmax 손실은 수렴을 촉진하고 보장하기 위해 결합된다. 최적화 어려움을 극복하기 위해 추가 코사인 여백 cos(α)-m은 각도 여백을 코사인 공간으로 이동시킨다. cosine margin의 구현과 최적화는 Sphere Face보다 훨씬 쉽고, 수월히 재현할 수 있으면서도 MegaFace에서 최첨단 성능을 달성한다. 
Enclidean margin based loss와 비교하여 angular and cosine margin based loss는 인간의 얼굴이 manifold에 놓여 있는 이전과 본질적으로 일치하는 hypershpere manifold에 차별적 제약을 추가한다.  

위에서 언급한 세 가지 속성인 데이터, 네트워크 및 Loss가 얼굴 인식 모델의 성능에 높은 영향을 주는 것은 잘 알려져 있다.  

본 논문에서는 이 세 가지 특성 모두에서 심층 얼굴 인식을 개선하는 데 기여한다.  

Data. 가장 큰 공개 교육 데이터인 MS-Celeb-1M을 자동 및 수동 방식으로 개선했다. Resnet-27 네트워크를 통해 정교한 MS1M 데이터 세트의 품질과 NIST 얼굴 인식 상 챌린지 2의 한계 손실을 점검했다. 메가 페이스 백만 개의 산란기와 FaceScrub 데이터 세트 사이에 수백 개의 겹치는 얼굴 이미지가 있다는 것을 발견하여 평가 결과에 상당한 영향을 미친다. MegaFace 산란기에서 이러한 겹치는 얼굴 이미지를 수동으로 찾는다. 교육 데이터와 테스트 데이터의 정교함은 모두 공개된다.  

Network. VGG2를 교육 데이터로 사용하여 컨볼루션 네트워크 설정과 관련된 광범위한 대조 실험을 수행하고 LFW, CFP 및 AgeDB에 대한 검증 정확도를 보고한다. 제안된 네트워크 설정은 큰 포즈 및 연령 변화 하에서 견고하게 확인되었다. 가장 최근의 네트워크 구조를 기반으로 속도와 정확성 사이의 절충을 탐구한다.  

Loss. 강력한 얼굴 인식을 위한 높은 차별적 특징을 학습하기 위해 새로운 손실 함수인 가산 각도 여유(ArcFace)를 제안한다. 그림 1에서 보듯이, 제안된 손실 함수는 L2 정규화된 가중치와 형상에 기초한 각도(arc) 공간에서 결정 경계를 직접 최대화한다. ArcFace가 보다 명확한 기하학적 해석을 가지고 있을 뿐만 아니라 곱셈 각도 여유와 첨가 코사인 여유와 같은 기준 방법을 능가한다는 것을 보여준다. 반하드 샘플 분포의 관점에서 아크페이스가 소프트맥스, 스피어페이스 및 코사인페이스보다 나은 이유를 혁신적으로 설명한다.  

Performance. 제안된 ArcFace는 메가페이스 챌린지에서 최첨단 결과를 달성하는데, 이는 100만 명의 얼굴을 가진 가장 큰 대중 얼굴 벤치마크이다. 데이터, pretrained Model과 Train/Test Code를 통해 이러한 결과를 재현할 수 있도록 한다.  

## 3. From Softmax to Arcface  
### 1. Softmax  
가장 널리 사용되는 분류 손실 함수인 소프트맥스 손실은 다음과 같이 제시된다.  

그러나 소프트맥스 손실 함수는 positive pairs에 대해 더 높은 유사성 점수를 가지도록 features를 명시적으로 최적화하지 않으며, negative pair에 대해서는 더 낮은 유사성 점수를 가지므로 성능 차이가 발생한다.  

### 2. Weights Normalisation  
Sphere Face의 실험에서 L2 weight normalization는 성능만 거의 개선하지 않는다.  

### 3. Multiplicative Angular Margin  
추가 예정  

### 4. Feature Normalisation  
추가 예정  

### 5. Additive Cosine Margin  
본 논문에서는 cosine margin을 0.35로 설정하였다. Sphere Face와 비교하여 additive cosine margin(Cosine-Face)은 다음과 같은 세 가지 이점을 제공한다.  
- 까다로운 초 매개 변수 없이 구현하기 매우 쉽다.  
- 소프트맥스 감독 없이 보다 명확하고 수렴할 수 있다.  
- 명백한 성능 개선.  

### 6. Additive Angular Margin  
Facecnn v1, Additive margin softmax for face verification의 cosine margin은 cosine 공간에서 angular 공간으로의 일대일 매핑을 가지고 있지만, 이 두 margin 사이에는 여전히 차이가 있다. 사실, angular margin은 cosine margin에 비해 기하학적 해석이 더 명확하며, angular 공간의 margin은 hypersphere manifold의 arc distance에 해당한다.  

Sphere Face와 CosineFace에 비해 본 논문의 방법은 기하학적 해석이 가장 우수하다. 다른 색상 영역은 서로 다른 클래스의 특징 공간을 나타낸다. ArcFace는 형상 영역을 압축할 수 있을 뿐만 아니라 하이퍼스피어 표면의 측지 거리에도 해당된다.  

### 7. Comparison under Binary Case
소프트맥스에서 제안된 Arcface 까지의 프로세스를 더 잘 이해하기 위해 표와 이진 분류 사례에 따라 결정 경계 그림을 제공한다.  

가중치와 특징 정규화에 기초하여이러한 방법의 주된 차이점은 여백을 어디에 두느냐이다.  

표 이진 분류 사례 아래의 클래스 1에 대한 결정 경계입니다. βi는 Wi와 x 사이의 각도이고, s는 하이퍼스피어 반지름이고, m은 여백이다.  

그림 이진 분류 사례에서 서로 다른 손실 함수의 결정 여백. 파선은 결정 경계를 나타내고 회색 영역은 결정 여백이다.  

### 8. Target Logit Analysis  
추가 예정  

## 4. Experiments  
본 논문에서는 얼굴 식별 및 검증 벤치마크 중 가장 큰 MegaFace Challenge에서 완전히 재현 가능한 방식으로 최첨단 성능을 얻는 것을 목표로 한다. 야생에서 라벨링 된 얼굴(LFW), 전방 프로필의 유명인(CFP), 연령 데이터베이스(AgeDB)를 검증 데이터 세트로 사용하고 네트워크 설정과 손실 함수 설계와 관련된 광범위한 실험을 수행한다. 제안된 ArcFace는 이 네 개의 데이터 세트 모두에서 최첨단 성능을 달성한다.  

### 1. Data
#### 1. Training data
 교육 데이터로 VGG2와 MS-Celeb-1M이라는 두 개의 데이터 세트를 사용한다.
VGG2. VGG2 데이터 세트에는 8,631개의 ID를 가진 교육 세트(3,141,890개의 이미지)와 500개의 ID를 가진 테스트 세트(169,396개의 이미지)가 포함되어 있다. VGG2는 자세, 연령, 조명, 민족성 및 직업에서 큰 차이를 보인다. VGG2는 고품질 데이터 세트이므로 데이터를 정제하지 않고 직접 사용한다.  

MS-Celeb-1M 원본 MS-Celeb-1M 데이터 세트에는 1,000만 개의 이미지를 가진 약 100k 개의 ID가 포함되어 있다. MS-Celeb-1M의 노이즈를 줄이고 고품질 훈련 데이터를 얻기 위해 ID 센터까지의 거리에 따라 각 ID의 모든 얼굴 이미지를 순위를 매긴다. 특정 아이덴티티의 경우, 피처 벡터가 아이덴티티의 피처 센터에서 너무 멀리 떨어져 있는 얼굴 이미지는 자동으로 제거된다. 또한 각 ID에 대한 첫 번째 자동 단계의 임계값 주변의 얼굴 이미지를 수동으로 확인한다. 마지막으로 3을 포함하는 데이터 세트를 얻는다. 85k 개의 고유 ID를 가진 8M 개의 이미지. 다른 연구자들이 본 논문의 모든 실험을 재현할 수 있도록, 정제된 MS1M 데이터 세트를 이진 파일 내에서 공개하지만, 이 데이터 세트를 사용할 때는 원본 용지를 인용하고 원래 라이선스를 따른다. 여기서 기억해야 할 점은 본 논문은 단지 훈련 데이터 개선일 뿐이지 공개가 아니다.  

#### 2. Validation data
야생에서 라벨링 된 얼굴(LFW), 전방 프로필의 유명인(CFP) 및 연령 데이터베이스(AgeDB)를 검증 데이터 세트로 사용한다.  

LFW. LFW 데이터 세트에는 5749개의 서로 다른 ID에서 13,233개의 웹 수집 이미지가 포함되어 있으며 포즈, 표현 및 조도에 큰 변화가 있다. 라벨이 부착된 외부 데이터에 제한되지 않는 표준 프로토콜을 따라, 6000개의 얼굴 쌍에 대한 검증 정확도를 제공한다.  

CFP 데이터 세트는 각각 10개의 정면 영상과 4개의 종단 영상을 가진 500개의 피사체로 구성된다. 평가 프로토콜에는 정면(FF) 및 전면 프로파일(FP) 얼굴 검증이 포함되며, 각각 10개의 폴더가 있고 350개의 동일한 사람 쌍과 350개의 다른 사람 쌍이 있다. 본 논문에서는 성능을 보고하기 위해 가장 까다로운 부분 집합인 CFP-FP만 사용한다.  

AgeDB. AgeDB 데이터 세트는 포즈, 표현, 조명 및 연령에서 큰 변화를 가진 야생 데이터 세트이다. AgeDB는 배우, 여배우, 작가, 과학자, 정치인 등 440여 개의 주제를 담은 12,240개의 이미지를 담고 있다. 각 이미지에는 ID, 연령 및 성별 속성에 대한 주석이 달려 있다. 최소 연령과 최대 연령은 각각 3세와 101세이다. 각 과목의 평균 연령대는 49세이다. 서로 다른 연도 간격(각각 5년, 10년, 20년, 30년)을 갖는 네 가지 테스트 데이터 그룹이 있다. 각 그룹에는 10개의 얼굴 이미지가 분할되어 있으며, 각 분할에는 300개의 긍정적인 예와 300개의 부정적인 예가 포함된다. 얼굴 검증 평가 메트릭은 LFW와 동일하다. 본 논문에서는 성능을 보고하기 위해 가장 까다로운 하위 집합인 AgeDB-30만 사용한다.  

#### 3. Test data
MegaFace 메가페이스 데이터 세트는 최대 공개 테스트 벤치마크로서 공개되며, 이는 얼굴 인식 알고리듬의 성능을 백만 개의 분산 장치 규모로 평가하는 것을 목표로 하고 데이터 세트에는 갤러리 세트 및 프로브 세트가 포함된다. 
야후의 플리커 사진 중 일부인 갤러리 세트는 690k의 각기 다른 개인으로부터 백만 개 이상의 이미지로 구성되어 있다. testset는 두 개의 기존 데이터베이스이다.  

FaceScrub은 530명의 고유한 개인 100k 사진을 포함하는 공개 데이터 세트이며, 55,742개의 이미지가 남성이고 52,076개의 이미지가 여성이다. FGNet은 82개의 ID에서 1002개의 이미지를 가진 얼굴 노화 데이터 세트이다. 각각의 정체성은 다른 나이(1에서 69까지의 범위)에 여러 개의 얼굴 이미지를 가지고 있다.  

메가페이스의 데이터 수집은 매우 힘들고 시간이 많이 소요되므로 데이터 노이즈가 불가피하다는 것은 충분히 이해할 수 있다. FaceScrub 데이터 세트의 경우 특정 ID의 모든 얼굴 이미지는 동일해야 한다. 100만 명의 산란기의 경우 FaceScrub ID와 겹치지 않아야 한다. 그러나 FaceScrub 데이터 세트에는 노이즈가 있는 얼굴 이미지가 있을 뿐만 아니라 100만 개의 산만 장치에도 존재한다는 것을 발견하여 성능에 상당한 영향을 미친다.  

모든 면의 아이덴티티 센터까지의 코사인 거리에 따라 순위를 매긴다. 사실, 얼굴 이미지 221과 136은 아론 에크하트가 아니다. 
FaceScrub 데이터 세트를 수동으로 정리하고 마침내 605개의 노이즈가 있는 얼굴 이미지를 찾는다. 
테스트 중에 노이즈가 있는 얼굴을 다른 오른쪽 얼굴로 변경하여 식별 정확도를 약 1% 높일 수 있다.  

### 2. Network Settings  
먼저 VGG2를 Training Data로, softmax를 loss function으로 사용하여 서로 다른 네트워크 설정을 기반으로 face verification 성능을 평가한다. 본 논문의 모든 실험은 MxNet에 의해 구현된다.  배치 크기를 512로 설정하고 4개 또는 8개의 NVIDIA Tesla P40(24GB) GPU에서 모델을 학습한다. learning rate는 0.1에서 시작하여 100k, 140k, 160k 반복에서 10으로 나눈다. 총 iteration step은 200k로 설정된다. 
momentum을 0.9로 설정하고 weight decay를 5e - 4로 설정했다.  

#### 1. Input setting
다음으로, 얼굴 이미지를 normalize하기 위해 similarity transform을 위한 5개의 얼굴 랜드마크(눈 가운데, 코끝 및 입 가름)를 사용한다. 얼굴 이미지는 crope되고, 112 × 112로 크기가 조절되며, RGB 이미지는 127.5를 뺀 다음 128로 나누어 정규화된다.  

대부분의 컨볼루션 네트워크는 Image-Net 분류 작업을 위해 설계되었기 때문에 입력 이미지 크기는 일반적으로 224 × 224 이상으로 설정된다. 참고로 face crops 크기는 112 ×112에 불과하다. 더 높은 feature map resolution을 보존하기 위해, conv7 × 7과 stread = 2를 사용하는 대신 첫 번째 컨볼루션 레이어에서 conv3 × 3과 stread = 1을 사용한다. 이 두 설정에서, 컨볼루션 네트워크의 출력 크기는 각각 7 × 7과 3 × 3이다.  

#### 2. Output setting  
마지막으로 여러 계층에서 임베딩 설정이 모델 성능에 어떤 영향을 미치는지 확인하기 위해 몇 가지 다른 옵션을 조사할 수 있다. 옵션 A의 임베딩 크기는 마지막 컨볼루션 레이어의 채널 크기에 따라 결정되므로 모든 feature embedding 채널 수는 옵션 A에 대해 512로 설정된다.  

- Option-A: Use global pooling layer(GP).
- Option-B: Use one fully connected (FC) layer after GP.
- Option-C: Use FC-Batch Normalisation (BN) after GP.
- Option-D: Use FC-BN-Parametric Rectified Linear Unit (PReLu) after GP.
- Option-E: Use BN-Dropout -FC-BN after the last convolutional layer.

#### 3. Block Setting  
원래 ResNet unit 외에도 얼굴 인식 모델의 훈련을 위한 보다 발전된 residual 유닛 설정도 조사한다.  

#### 4. Backbones  
모델 구조 설계에 대한 최근의 발전을 기반으로, 심층 얼굴 인식을 위해 MobileNet, InceptionResnet-V2, Densely connected convolutional networks(DenseNet), SE(Squeeze and excitation Network), DPN(Dual Path Network)도 탐구한다. 본 논문에서는 정확도, 속도 및 모델 크기 측면에서 이러한 네트워크 간의 차이를 비교한다.  

#### 5. Network Setting Conclusions  
"L"의 설정을 포함하거나 포함하지 않은 두 개의 네트워크를 비교한다. 첫 번째 컨볼루션 레이어로 conv3 × 3과 stread = 1을 사용할 때, 네트워크 출력은 7 ×7이다. 대조적으로, 만약 첫 번째 컨볼루션 레이어로 conv7 ×7과 stread=2를 사용한다면, 네트워크 출력은 3 × 3에 불과하다. 훈련 중에 더 큰 feature map을 선택하는 것이 더 높은 검증 정확도를 얻는다는 것은 명백하다. 옵션 E(BN-Dropout-FC-BN)는 최상의 성능을 제공한다. 본 논문에서는 드롭아웃 매개변수가 0.4로 설정된다. Dropout over-fitting을 방지하고 심층 얼굴 인식을 위한 더 나은 일반화를 얻기 위해 정규화를 효과적으로 작용할 수 있다. 원래 residual 단위와 개선된 residual 단위를 비교 결과에서 알 수 있듯이 제안된 BN-Conv(strid=1)-BN-PReLu-Conv(strid=2)-BN 유닛은 검증 성능을 분명히 개선할 수 있다. LFW의 성능이 거의 포화 상태이기 때문에, 이러한 네트워크 백본을 비교하기 위해 더 어려운 테스트 세트인 CFP-FP와 AgeDB-30에 초점을 맞춘다. Inception-Resnet V2 네트워크는 긴 실행 시간(53.6ms)과 가장 큰 모델 크기(642MB)로 최고의 성능을 얻는다. 반면 모바일 넷은 112MB 모델로 4.2ms 이내에 face feature embedding을 마칠 수 있고 성능은 소폭 하락할 뿐이다. ResNet-100, Inception-Resnet-V2, DenseNet, DPN 및 SE-Resnet-100과 같은 대규모 네트워크 사이의 성능 차이는 상대적으로 작다. ccuracy, speed and model size 간의 절충을 기반으로, 메가 페이스 과제에 대한 실험을 수행하기 위해 LResNet100 E-IR을 선택한다. Weight decay. SE-LResNet50 E-IR 네트워크를 기반으로 중량 감소(WD) 값이 검증 성능에 어떤 영향을 미치는지도 살펴본다. weight decay 값이 5e - 4로 설정되면 검증 정확도가 가장 높은 지점에 도달한다. 따라서 다른 모든 실험에서 weight decay를 5e - 4로 고정한다.  

### 3. Loss Setting
margin parameter m은 제안된 ArcFace에서 중요한 역할을 하므로 먼저 최상의 angular margin을 검색하기 위한 실험을 수행한다. 0.2에서 0.8까지 m을 변화시킴으로써 LMobileNetE 네트워크와 ArcFace 손실을 사용하여 정교한 MS1M 데이터 세트에 대한 모델을 교육한다. 성능은 모든 데이터 세트에서 m = 0.2보다 일관되게 향상되고 m = 0.5에서 포화 상태가 된다. 그러면 검증 정확도가 m = 0.5에서 감소한다. 본 논문에서는 가산 각도 여백 m을 0.5로 수정한다. LresNet100 E-IR 네트워크와 정교한 MS1M 데이터 세트를 기반으로 소프트맥스, 스피어 페이스, 코사인 페이스 및 아크 페이스와 같은 다양한 손실 기능의 성능을 비교한다. 표 7에서는 LFW, CFP-FP 및 AgeDB-30 데이터 세트에 대한 자세한 검증 정확도를 제공한다. LFW가 거의 포화상태에 달해 성능 향상이 뚜렷하지 않다.  

- (1) Softmax, SphereFace, CosineFace 및 ArcFace에 비해 특히 큰 포즈 및 연령 변화 하에서 성능이 확실히 개선된다.
- (2) CosineFace와 ArcFace는 훨씬 더 쉬운 구현으로 SphereFace를 능가한다. 코사인 페이스와 아크 페이스는 소프트맥스의 추가 감독 없이도 쉽게 융합할 수 있다. 이와는 대조적으로, Speace Face는 훈련 중 분산을 피하기 위해 Softmax의 추가 감독이 필수적이다. 
- (3) ArcFace는 CosineFace보다 약간 낫다. 그러나 ArcFace는 그림 1과 같이 보다 직관적이며 하이퍼 스피어 매니폴드에 대한 기하학적 해석이 보다 명확하다.