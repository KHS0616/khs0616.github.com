---
sort: 5
---

# Yolo4  
YOLOv4: Optimal Speed and Accuracy of Object Detection  

## 1. Introduction  
최신 Neural Networks들은 높은 정확도를 가지지만, 낮은 FPS, 큰 mini batch size로 인해 학습하는데 많은 GPU가 필요한 단점이 있다. Yolo v4는 1개의 GPU를 사용하는 일반적인 학습환경에서 학습이 가능하다. 또한 BOF, BOS 기법 등 다양한 최신 기법을 사용한 효율적이고 강력한 Object Detection 모델이다.  

## 2. Related Work  
### 1. Object Detection  
최신 detector는 주로 백본(Backbone)과 헤드(Head)라는 두 부분으로 구성된다. 백본은 입력 이미지를 feature map으로 변형시켜주는 부분이다. ImageNet 데이터셋으로 pre-trained 시킨 VGG16, ResNet-50 등이 대표적인 Backbone이다. 헤드는 Backbone에서 추출한 feature map의 location 작업을 수행하는 부분이다. 헤드에서 predict classes와 bounding boxes 작업이 수행된다.  

헤드는 크게 Dense Prediction, Sparse Prediction으로 나뉘는데, 이는 Object Detection의 종류인 1-stage인지 2-stage인지와 직결된다. Sparse Prediction 헤드를 사용하는 Two-Stage Detector는 대표적으로 Faster R-CNN, R-FCN 등이 있다. Predict Classes와 Bounding Box Regression 부분이 분리되어 있는 것이 특징이다.  

Dense Prediction 헤드를 사용하는 One-Stage Detector는 대표적으로  YOLO, SSD 등이 있다. Two-Stage Detector과 다르게, One-Stage Detector는 Predict Classes와 Bounding Box Regression이 통합되어 있는 것이 특징이다.  

넥(Neck)은 Backbone과 Head를 연결하는 부분으로, feature map을 refinement(정제), reconfiguration(재구성)한다. 대표적으로 FPN(Feature Pyramid Network), PAN(Path Aggregation Network), BiFPN, NAS-FPN 등이 있다.  

### 2. Bag of Freebies (BOF)  
BOF는 inference cost의 변화 없이 성능을 향상(better accuracy)시킬 수 있는 딥러닝 기법들이다. 대표적으로 데이터 증강(CutMix, Mosaic 등)과 BBox(Bounding Box) Regression의 loss 함수(IOU loss, CIOU loss 등)이 있다.  

### 3. Bag of Specials (BOS)  
BOS는 BOF의 반대로, inference cost가 조금 상승하지만, 성능 향상이 되는 딥러닝 기법들이다. 대표적으로 enhance receptive filed(SPP, ASPP, RFB), feature integration(skip-connection, hyper-column, Bi-FPN) 그리고 최적의 activation function(P-ReLU, ReLU6, Mish)이 있다.  

## 3. Methodology  
본 논문의 주된 목적은 빠르게 작동하는 Neural Network, 병렬 처리에 대한 최적화가 목표이다.  

### 1. Selection of Architecture  
목표는 입력 network resolution, convolution layer number, parameter number(filter_size^2 * filter * channel/group) 및 layer output number(filter) 사이의 최적의 균형을 찾는 것이다.  
예를 들어, CSPResNext50이 ILSVRC2012(ImageNet) 데이터 세트의 객체 분류 측면에서 CSPDarknet53에 비해 상당히 우수하다는 것을 보여준다. 그러나, 반대로, CSPDarknet53은 MS COCO 데이터 세트의 객체 탐지 측면에서 CSPResNext50에 비해 더 우수하다.  

다른 목표는 다양한 Detection Level(FPN, PAN, ASFF, BiFPN)에 대한 Backbone Level에서 최선의 parameter 연산 방식을, receptive field를 증가시키기 위해 추가 블록을 선택하는 것이다.  

Classification에 최적인 model이 항상 Detection에 최적인 것은 아니다. Classification과 달리 Detection에는 다음이 필요하다.  
- 다양한 작은 Objects들을 Detecting하기 위한 더 큰 Network size(Resolution)  
- Input Network Size의 증가를 대비한 더 큰 receptive field를 위한 더 많은 layers  
- 한 장의 이미지에서 서로다른 크기의 Objects들을 Detect하기 위한 모델의 큰 수용량을 위한 더 많은 parameters  

위 3가지 요건을 고려하면, 더 큰 receptive field 크기(더 많은 수의 convolution layers 3 × 3)와 더 많은 수의 parameters를 가진 모델을 Backbone으로 선택해야 한다. CSPResNext50은 16개의 convolution layers 3 × 3, 425 receptive field 및 20.6M parameters만 포함하고 있는 반면, CSPDarknet53은 29개의 convolution layer 3 × 3, 725 × 725 receptive field 및 27.6M parameters를 포함한다. 이러한 이론적 정당성은 수많은 실험과 함께 CSPDarknet53 Network가 Detection의 BackBone으로서 둘 중 최적의 모델임을 보여준다. receptive field 크기에 따른 영향력은 아래와 같다.  

- Object Size 까지의 receptive field는 Object 전체 정보를 이용할 수 있다.  
- Network Size 까지의 receptive field는 Object 주변 정보까지 이용할 수 있다.  
- Network Size를 초과할 경우 Image Point와 Final Activation 사이의 연결점을 증가시켜준다.  

Yolo v4에서는 SPP 블록을 CSPDarknet53에 추가했는데, 이는 receptive field를 크게 늘리고 가장 중요한 context features를 분리하며 Network 작동 속도를 거의 감소시키지 않기 때문이다. YOLOv3에 사용되는 FPN 대신 다양한 Detection Level에 대한 다양한 Backbone Level에서 parameter aggregation 방법으로 PANet을 사용한다. YOLOv4의 Architecture로 CSPDarknet53 Backbone, SPP add modules, PANet path-aggregation neck 및 YOLOv3(Anchor Based) head를 선택한다. 향후 우리는 이론적으로 일부 문제를 해결하고 정확도를 높일 수 있는 Detector를 위해, Bag of Freebies의 content를 크게 확장하고 각 기능의 영향을 순차적으로 확인할 계획이다. Cross-GPU Batch Normalization(CGBN 또는 SyncBN)나 값비싼 전문 기기를 사용하지 않는다. 이를 통해 누구나 GTX 1080Ti 또는 RTX 2080Ti와 같은 기존 그래픽 프로세서에서 최첨단 결과를 재현할 수 있다.  

### 2. Selection of BoF and BoS  
CNN에서 사용하는 주된 기법은 아래와 같다.  
- Activation: ReLU, leaky-ReLU, pReLU, ReLU6, SELU, Swish, or Mish  
- Bounding box regression loss: MSE, IoU, GIoU, CIoU, DIoU  
- Regularization method: DropOut, DropPath, Spatial DropOut, or DropBlock  
- Normalization of the Network activations by their mean and variance: Batch Normalization(BN), Cross-GPU Batch Normalization(CGBN or SyncBN), Filter Response Normalization(FBN), Cross-Iteration Batch Normalization(CBN)  
- Skip-connections: Residual connections, Weighted residual connections, Multi-input weighted residual connections, or Cross stage partial connections (CSP)  

Activation Function에 대해서는 PRLU와 SELU가 학습하기 더 어렵고, ReLU6는 Quantize Network를 위해 특별히 설계되었으므로 후보 목록에서 제외한다. Regularization 방법으로는 DropBlock을 선택한다. normalization 방법의 선택에 대해서는 GPU 하나만 사용하는 학습 전략에 중점을 두기 때문에 SyncBN은 고려되지 않는다.  

### 3. Additional improvements  
하나의 GPU에서 적합한 학습을 위해 아래와 같은 내용을 추가했다.  
- 새로운 Data augmentation 방식인 Mosaic, 그리고 Self-Adversarial Training(SAT)를 추가하였다.  
- Genetic 알고리즘을 적용하는 동안 optimal hyper-parameters를 선택한다.  
- 기존 모델을 본 논문의 목적에 맞게 수정하여 사용한다.(modified SAM, modified PAN, Cross mini-batch normalization CmBN)  

Mosaic은 4개의 학습 이미지를 혼합하는 새로운 Data augmentation 방법이다. 따라서 CutMix는 2개의 입력 영상만 혼합하는 반면, Mosaic은 4개의 다른 컨텍스트가 혼합된다. 이는 정상적인 context를 벗어난 물체의 detection을 낮춘다. 또한 batch normalization은 각 레이어에 있는 4개의 서로 다른 이미지에서 activation statistics를 계산한다. 따라서 큰 mini batch size가 필요하지 않다.  

Self-Adversarial Training(SAT)은 또한 2개의 순전파, 역전파 단계에서 작동하는 새로운 Data augmentation 기술을 나타낸다. 1단계에서 Network는 network weights 대신 원본 이미지를 변경한다. 이러한 방식으로 network는 이미지에 원하는 물체가 없다는 속임수를 만들기 위해 원래 이미지를 변경함으로써 자신에 대한 적대적 공격을 실행한다. 두 번째 단계에서 network는 이 수정된 이미지의 물체를 정상적인 방법으로 감지하도록 학습된다.  

본 논문에서는 SAM을 기존의 spatial-wise attention 방식에서 point-wise attention 방식으로 변경한다. PAN에서는 shortcut connection 구조를 concatenation 구조로 변경한다.  

### 4. Yolo v4  
Yolo v4는 아래와 같이 구성된다.  
- Backbone: CSPDarknet53  
- Neck: SPP, PAN  
- Head: YOLOv3  

BackBone Network 성능 향상을 위한 BoF, BoS는 아래와 같다.  
- BoF: CutMix and Mosaic data augmentation, DropBlock regularization, Class label smoothing  
- BoS: Mish activation, Cross-stage partial connections (CSP), Multi-input weighted residual connections (MiWRC)  

Detect Network 성능 향상을 위한 BoF, BoS는 아래와 같다  
- BoF: CIoU-loss, CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training, Eliminate grid sensitivity, Using multiple anchors for a single groundtruth, Cosine annealing scheduler, Optimal hyper parameters, Random training shapes  
- BoS: Mish activation, SPP-block, SAM-block, PAN path-aggregation block, DIoU-NMS  

## 4. Experiments  
### 1. Training Strategy  
ImageNet image classification 학습에 관련된 hyper parameter는 아래와 같다.  
- training steps: 8000000 iteration  
- 128 batch size, 32 mini batch size  
- momentum을 사용하며 weight 값은 각각 0.9, 0.005  
- learning rate는 0.1부터 시작해서 1000번 마다 감소  
- BoS,BoF 방식의 hyper parameters는 위와 같고 BoF는 50%의 training steps  

MS COCO object detection 학습에 사용된 hyper parameters는 아래와 같다.  
- training steps: 500500 iteration  
- learning rate는 0.01부터 시작해서 400000, 450000 steps에 0.1만큼 곱해져서 적용된다.  
- momentum을 사용하며 weight 값은 각각 0.9, 0.005  
- 64 batch size, 8 mini batch size or 4 mini batch size  

YOLOv3-SPP를 사용한 Genetic 알고리즘 실험에서는 위와는 다른 hyper parameters를 설정했다. hyper parameters는 아래와 같다.  
- GIoU loss 사용  
- 최소 5000iteration을 300epoch 학습  
- learning rate 0.00261 설정  
- momentum 0.949  
- IoU threshold 0.213  
- loss normalizer 0.07  

### 2. Influence of different features on Classifier training  
Class label smoothing 영향, 서로 다른 Data augmentation의 영향, MixUp, CutMix 및 Mosaic, Leaky-ReLU, Swish, Miss와 같은 다양한 Activation의 영향을 연구한다.  
표 2에 설명된 것처럼 본 실험에서, CutMix 및 Mosaic Data augmentation, Class label smoothing, Mish Activation 같은 기능을 도입함으로써 Classifier의 정확도가 향상된다. 따라서 Classifier 학습을 위한 BOF Backbone(Bag of Freebies)는 CutMix 및 Mosaic Data augmentation 및 Class label smoothing. 또한 표 2와 표 3에 표시된 바와 같이 보완적인 옵션으로 Mish Activation을 사용합니다.  

### 3. Influence of different features on Detector training  
본 연구는 표 4와 같이 Detector Training accuracy에 대한 다양한 BoF Detector(Bag-of-Freebies)의 영향에 관한 것이다. FPS에 영향을 주지 않고 Detector accuracy를 높이는 다양한 특징을 연구하여 BoF 목록을 크게 확장한다.  
- S: 격자 감도를 제거한다. 방정식 bx = ((ty)+cy,by =ty(ty)+cy,where (cx 및 cy 변수는 전체 갯수를 의미한다)는 객체 좌표를 평가하기 위해 YOLOv3에서 사용된다. 위 식에 따르면 cx 또는 cx + 1 값에 접근하는 bx 값에 매우 높은 tx 절대값이 필요하다. 이 문제를 해결하기 위해 Sigmoid에 1.0을 초과하는 계수를 곱하여 물체를 감지할 수 없는 격자의 영향을 제거한다.  
- M: Moasic Data Augmentation - 학습 중 단일 이미지 대신 4개의 이미지 mosaic을 사용한다.  