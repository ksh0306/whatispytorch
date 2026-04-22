# MNIST 분류 PyTorch 예제 코드 설명

본 문서는 [mnisttorch.py](mnisttorch.py) 코드를 한 줄씩 분석하여 PyTorch로 MNIST 손글씨 숫자 분류기를 구현하는 과정을 상세하게 설명합니다.

MNIST는 0~9 사이의 손글씨 숫자 이미지 데이터셋으로, 각 이미지는 `28 x 28` 픽셀의 흑백 이미지입니다. 이 코드는 MNIST 데이터를 불러와 신경망을 학습시키고, 테스트 데이터셋에 대한 정확도를 측정합니다.

---

## 1. 라이브러리 import

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

| 모듈 | 역할 |
|------|------|
| `torch` | PyTorch의 핵심 모듈. 텐서 연산과 자동 미분(autograd) 기능을 제공합니다. |
| `torch.nn` | 신경망 계층(Layer), 손실 함수 등 모델 구성 요소를 담고 있는 모듈. |
| `torch.nn.functional` (`F`) | 함수형 API. `nn`이 클래스 기반이라면 `F`는 함수 기반 (예: `F.relu`). |
| `torch.optim` | SGD, Adam과 같은 최적화 알고리즘을 제공합니다. |
| `torchvision.datasets` | MNIST, CIFAR 등 표준 비전 데이터셋을 쉽게 사용할 수 있게 해줍니다. |
| `torchvision.transforms` | 이미지 전처리 도구 (Tensor 변환, 정규화 등). |
| `DataLoader` | 데이터셋을 배치(batch) 단위로 효율적으로 꺼내주는 이터레이터. |

---

## 2. HTTP User-Agent 설정

```python
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
```

MNIST 데이터를 다운로드할 때 일부 서버가 기본 Python User-Agent를 차단하기 때문에, **브라우저처럼 보이도록** User-Agent를 `Mozilla/5.0`으로 바꿔주는 우회 코드입니다. 다운로드 403 오류를 방지하기 위한 조치입니다.

---

## 3. 하이퍼파라미터 설정

```python
batch_size = 64
learning_rate = 0.01
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
```

- **batch_size = 64**: 한 번에 모델에 입력할 이미지 개수. 작으면 메모리는 적게 쓰지만 학습이 느려지고, 크면 빠르지만 메모리 부담이 커집니다.
- **learning_rate = 0.01**: 가중치 업데이트의 보폭. 아래에서 Adam을 사용할 때는 실제로 `0.001`이 쓰입니다 (이 변수는 SGD용으로 남겨둔 값).
- **epochs = 5**: 전체 학습 데이터를 몇 번 반복해서 학습할지를 뜻합니다.
- **device**: GPU(`cuda`)가 사용 가능하면 GPU, 아니면 CPU를 선택합니다. 모든 텐서와 모델은 이 장치로 보내져야 연산이 가능합니다.

---

## 4. 데이터셋 로드 및 전처리

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081),)
])
```

- **`transforms.ToTensor()`**: PIL 이미지나 numpy 배열을 PyTorch 텐서로 변환하고, 픽셀 값을 `[0, 255]` → `[0.0, 1.0]`으로 스케일링합니다.
- **`transforms.Normalize((0.1307,), (0.3081,))`**: MNIST 학습셋의 실제 평균(0.1307)과 표준편차(0.3081)로 정규화합니다. 즉 `(x - mean) / std`. 입력 분포를 평균 0에 가깝게 만들어 학습을 안정적으로 만듭니다.

```python
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

- 학습셋(60,000장)과 테스트셋(10,000장)을 `./data` 폴더에 내려받고, 위에서 정의한 `transform`을 적용합니다.
- 학습 로더는 **`shuffle=True`**로 매 에폭마다 데이터 순서를 섞어 편향을 줄이고, 테스트 로더는 순서를 그대로 둡니다 (재현성 목적).

---

## 5. 신경망 정의

### 5.1 SimpleNet (완전연결 신경망, 주석 처리되어 미사용)

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

이미지를 `784차원` 벡터로 펼친 뒤 `784 → 128 → 64 → 10`으로 줄여가는 단순한 MLP입니다. 마지막 출력 10은 숫자 0~9에 대한 분류 점수(logit)입니다.

- `x.view(-1, 28*28)`: 배치 차원을 유지한 채 이미지를 1차원 벡터로 평탄화. `-1`은 배치 크기를 자동 계산합니다.
- `torch.relu`: 음수를 0으로, 양수는 그대로 통과시키는 활성화 함수.

### 5.2 CNNNet (합성곱 신경망, 실제 학습에 사용)

```python
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
```

구성 요소:

| 레이어 | 설명 |
|--------|------|
| `conv1` | 입력 채널 1(흑백) → 출력 채널 32. 3x3 커널, `padding=1`이므로 크기는 그대로 유지 (28x28). |
| `conv2` | 32 → 64 채널로 확장. 동일하게 3x3 커널, padding=1. |
| `pool` | 2x2 MaxPooling. 가로·세로 크기를 절반으로 줄입니다. |
| `dropout1` | 25% 확률로 뉴런을 무작위로 0으로. 과적합 방지. |
| `dropout2` | 50% 확률로 뉴런을 무작위로 0으로. FC 층 사이에서 더 강한 정규화. |
| `fc1` | `64 채널 × 7 × 7 = 3136` 차원을 128로 압축. |
| `fc2` | 128 → 10 (클래스 개수). |

### 5.3 forward 흐름

```python
def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))  # (B,1,28,28) → (B,32,14,14)
    x = self.pool(F.relu(self.conv2(x)))  # → (B,64,7,7)
    x = self.dropout1(x)
    x = x.view(-1, 64 * 7 * 7)            # Flatten → (B, 3136)
    x = F.relu(self.fc1(x))               # → (B, 128)
    x = self.dropout2(x)
    x = self.fc2(x)                       # → (B, 10)
    return x
```

**텐서 모양 변화**:
- 입력: `(B, 1, 28, 28)` (B는 배치 크기)
- conv1 + pool 후: `(B, 32, 14, 14)`
- conv2 + pool 후: `(B, 64, 7, 7)`
- Flatten 후: `(B, 3136)`
- fc1 후: `(B, 128)`
- fc2 후: `(B, 10)` ← 최종 로짓

> 주의: 출력에 softmax를 직접 적용하지 않습니다. `CrossEntropyLoss`가 내부에서 LogSoftmax를 포함하기 때문입니다.

---

## 6. 모델, 손실 함수, 옵티마이저 준비

```python
# model = SimpleNet().to(device)
model = CNNNet().to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

- **`.to(device)`**: 모델 파라미터를 GPU 또는 CPU 메모리로 이동시킵니다. 이후 입력 텐서도 같은 device에 있어야 연산이 가능합니다.
- **`CrossEntropyLoss`**: 다중 클래스 분류에서 표준으로 쓰는 손실 함수. `LogSoftmax + NLLLoss`를 합친 형태입니다. 입력은 원시 로짓, 타깃은 클래스 인덱스 (0~9)면 됩니다.
- **`optim.Adam`**: 적응적 학습률(Adaptive Learning Rate) 알고리즘. 초기 수렴이 빠르고 대부분의 문제에 무난히 잘 작동합니다. 학습률 `0.001`은 Adam의 일반적인 기본값입니다.

---

## 7. 학습 루프

```python
model.train()
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(f'Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')
```

**한 스텝(iteration) 내부에서 일어나는 일**:

1. **`model.train()`**: 학습 모드로 전환. `Dropout`과 `BatchNorm` 같은 층의 동작을 학습용으로 세팅합니다.
2. **`data.to(device), target.to(device)`**: 배치 데이터를 모델과 같은 장치로 이동.
3. **`optimizer.zero_grad()`**: 이전 스텝에서 누적된 기울기를 0으로 초기화. PyTorch는 기본적으로 `.backward()`가 호출될 때마다 기울기를 **누적**하므로, 매 스텝마다 리셋해야 합니다.
4. **`output = model(data)`**: 순전파. 내부적으로 `forward()`가 호출됩니다.
5. **`loss = criterion(output, target)`**: 손실 계산.
6. **`loss.backward()`**: 역전파. 손실을 기준으로 각 파라미터에 대한 기울기를 자동 미분(autograd)으로 계산합니다.
7. **`optimizer.step()`**: 계산된 기울기를 이용해 파라미터를 실제로 업데이트합니다.

**로깅**: `batch_idx`가 500의 배수일 때마다 현재 손실을 출력합니다. MNIST 학습셋은 60,000개이고 배치 크기가 64이므로, 에폭당 약 938 iteration이 돌아가며 1 epoch당 약 2번의 로그가 찍힙니다.

---

## 8. 모델 평가(테스트)

```python
model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100 * correct / len(test_loader.dataset)
```

**핵심 포인트**:

- **`model.eval()`**: 평가 모드. Dropout은 꺼지고 BatchNorm은 학습 중 수집된 통계를 사용합니다.
- **`torch.no_grad()`**: 기울기 계산을 끕니다. 테스트에선 파라미터 업데이트가 없으므로 메모리 사용량과 연산 속도를 크게 개선합니다.
- **`output.argmax(dim=1, keepdim=True)`**: 클래스 차원(`dim=1`, 길이 10)에서 가장 큰 로짓을 가진 인덱스를 고릅니다. 이것이 모델의 예측 레이블입니다.
- **`pred.eq(target.view_as(pred))`**: 예측과 정답이 같은지 비교하여 Boolean 텐서를 얻고, `.sum().item()`으로 맞힌 개수를 누적합니다.

> ⚠️ **작은 주의점**: `test_loss`는 배치별 평균 손실을 더한 뒤 `len(test_loader.dataset)`(=샘플 수, 10,000)로 나누고 있습니다. `CrossEntropyLoss`의 기본 reduction이 `'mean'`이므로, 엄밀하게는 배치 개수로 나누는 것이 배치 평균에 더 맞지만, 단순 참고용 지표로 사용하는 수준에서는 큰 문제는 없습니다.

---

## 9. 결과 출력

```python
print(f'\n[Result of test]')
print(f'Loss:{test_loss:.4f}')
print(f'Accuracy:{correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
```

최종 손실과 정확도를 출력합니다. CNN 모델은 보통 MNIST에서 5 에폭 학습만으로도 **98~99% 정확도**를 쉽게 달성합니다.

---

## 10. 전체 흐름 요약

```
[MNIST 데이터 로드 + 정규화]
          ↓
[DataLoader로 배치 단위 공급]
          ↓
[CNNNet: Conv → Pool → Conv → Pool → Dropout → FC → Dropout → FC]
          ↓
[CrossEntropyLoss + Adam]
          ↓
[5 epochs 학습: forward → loss → backward → step]
          ↓
[model.eval() + no_grad로 테스트셋 평가 → 정확도 출력]
```

---

## 11. 추가로 실험해볼 만한 것

- `SimpleNet` ↔ `CNNNet` 주석을 바꿔가며 MLP와 CNN의 성능 차이 확인
- `epochs`를 10~20으로 늘렸을 때의 과적합 관찰 (테스트 정확도가 떨어지는 시점)
- `optim.Adam` 대신 `optim.SGD(momentum=0.9)` 사용해보기
- `transforms`에 `RandomRotation`, `RandomAffine` 같은 데이터 증강 추가
- `BatchNorm2d`를 Conv 층 사이에 끼워 학습 안정성 비교
- 학습된 모델을 `torch.save(model.state_dict(), 'mnist.pth')`로 저장하고 재사용