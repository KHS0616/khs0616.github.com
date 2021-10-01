---
sort: 1
---

# DepthWise-Separable-Conv  
DepthWise-Separable-Convolution  

## 1. DepthWise Convolution  
### 1. 개요  
![DepthWise-Convolution 연산](../../static/DepthWise-Separable-Conv/DepthWise-Conv.png)  
기존 CNN은 Feature Map을 생성하는데 Kernel Size * Kernel Size * Input Channel(입력 이미지 개수)의 Parameter를 사용한다.  
반면에 DepthWise Convolution 방식은 각 채널마다 Feature Map을 계산하여 합하게 되는데 이러한 과정을 통해 Input Channel이 줄어드므로 연산량이 감소되는 이점이 있다.  
또한 각 필터에 대한 연산 결과가 독립적일 필요가 있을 경우에 큰 장점이 된다.  

### 2. Pytorch Code  
```{.python}
class depthwise_conv2d(nn.Module):
    def __init__(self, input_channel):
        super(depthwise_conv2d, self).__init__()
        self.depthwise = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, groups=input_channel)

    def forward(self, x):
        out = self.depthwise(x)
        return out
```  

## 2. Pointwise Convolution(1x1 convolution)  
### 1. 개요  
ㅇㅇ

### 2. Pytorch Code  

### 3. Depthwise Separable Convolution  
### 1. 개요  
ㅇㅇ

### 2. Pytorch Code  
ddd