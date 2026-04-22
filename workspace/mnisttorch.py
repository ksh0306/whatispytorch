import torch # pytorch를 쓰겠다.
import torch.nn as nn # Neural Network 추상화를 쓰겠다.
import torch.nn.functional as F # 활성화 함수를 사용하겠다.(sigmoid, relu 같은) 가중치가 없는 순수 함수 연산이다.
import torch.optim as optim # DNN 최적화 기법을 사용하겠다.(SGD, Adam)
from torchvision  import datasets, transforms # 영상처리 관련 DNN 모듈에서 데이터셋과 변환툴을 사용하겠다.
from torch.utils.data import DataLoader # 데이터셋을 학습과정에 밀어넣는  DataLoader를 사용하겠다.

#파이토치의 데이터셋을 관리하는 주소에서 브라우저 등을 통하지 않는 접속의 경우 보안상의 문제로 거부하는 경우가 있어. 이를 우회한다.
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# 학습에 필요한 hyper paramer를 지정한다.
batch_size = 64 # 1회 역전파 학습시 데이터의 개수 = 64개
lr = 0.001 # 미분의 결과에 따라 데이터를 이동하는 값의 크기
epochs = 5 # 반복 학습의 횟수

# cuda가 있으면 학습에 이용한다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f'Using device: {device}')

# 필요한 변환 도구를 설정하여 조합한다.
# ToTensor() : 가로세로채널 -> 파이토치용 채널가로세로, 0~255 --> 0~1.0으로 스케일 변경
# transforms.Normalize(평균튜플, 표준편차튜플) --> 정규화하여 연산시 통계값이 발산해버리는 걸 방지한다.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081),)])

# 학습용과 검증용 데이터셋을 로딩한다.
# 파이토치에서 제공해주는 mnist를 가져온다.
# 데이터 포맷과 스케일링, 정규화는 상기 지정된 transform을 이용한다.
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

#학습과 검증을 진행할 로더를 설정한다.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# DNN 구성한다.nn.Module을 상속하여 만들면 된다.
# 레이어와 relu 함수 층을 하나 더 추가해 보았다.
class KSHNet(nn.Module):
    def __init__(self):
        super(KSHNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 196)
        self.fc2 = nn.Linear(196, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28) # 이 DNN은 우선 28*28 이미지를 1열로 펴는 것부터 한다.
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# nn class와 device를 조합하여 학습할 모델을 설정한다.
model1 = KSHNet().to(device)

# 손실함수와 최적화 함수를 설정한다.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=lr)

# 모델을 학습모드로 설정한다.
model1.train()

# 정해진 epoch 수 만큼
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # 배치 사이즈만큼 쪼개서 넘긴다.

        optimizer.zero_grad() # 최적화 함수의 기울기를 초기화

        # output에는 model1에 접근할 수 있는 정보가 포함되어있다.
        output = model1(data) # 모델에 데이터를 넣어서
        # 단순히 손실함수 계산값을 만드는 것에서 끝나는 게 아니라, output에서 model1의 값을 넘겨 받는다.
        loss = criterion(output, target) # 나온 값과 목적값을 비교하여 손실함수 값을 만든다.

        # model1에 연결된 optimizer에 기울기값을 산출 등록한다고 보면 된다.
        loss.backward()
        # 위 기울기값은 model에 적용한다.
        optimizer.step()

        if batch_idx % 500 == 0:
            print(f'Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')

print("Training complete")

# 모델을 검증모드로 변경한다.
model1.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model1(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100 * correct / len(test_loader.dataset)

print(f'\n[Result of test]')
print(f'Loss:{test_loss:.4f}')
print(f'Accuracy:{correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n\n\n\n')


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Drop out 추가...
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 392)
        self.fc2 = nn.Linear(392, 98)
        self.fc3 = nn.Linear(98, 49)
        self.fc4 = nn.Linear(49, 10)

    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.dropout1(x)
        
        # Flatten: 1차원으로 펼치기
        x = x.view(-1, 64 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout2(x)
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

model2= CNNNet().to(device) 

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model2.parameters(), lr=0.001)

model2.train()
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model2(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(f'Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')

print("Training complete")

model2.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model2(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100 * correct / len(test_loader.dataset)

print(f'\n[Result of test]')
print(f'Loss:{test_loss:.4f}')
print(f'Accuracy:{correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')