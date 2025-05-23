import torch
from torchvision import models, transforms
from PIL import Image #이미지 파일을 열고 조작하기 위한 라이브러리리
import urllib.request #인터넷에서 파일 다운
import os

# 전처리 함수 정의 (ImageNet 표준)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 사전학습된 모델 로드
model = models.resnet50(pretrained=True)
model.eval()

# 클래스 레이블 로드 (ImageNet 1000개 클래스)
LABELS_PATH = "imagenet_classes.txt"
if not os.path.exists(LABELS_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        LABELS_PATH
    )

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 이미지 분류 함수
def classify_image(img_path):
    if not os.path.exists(img_path):
        print("이미지 파일을 찾을 수 없습니다.")
        return

    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    top3 = torch.topk(probs, 3)
    print("분석 결과 (Top 3):")
    for i in range(3):
        label = labels[top3.indices[i]]
        score = round(top3.values[i].item() * 100, 2)
        print(f"{i+1}. {label} ({score}%)")

# 실행
classify_image("test.jpg")

# 202415014 강미소
#CNN은 이미지 처리에 적합한 구조를 가진 신경망으로 
#Convolution Layer(필터를 이용해 이미지의 특징추출),Pooling Layer(크기를 줄이고 중요한 정보 추출),
#Fully Connected Layer(추출된 특징을 바탕으로 최종 분류)와 같은 과정이 있다.

#Transformer은 원래 자연어처리(NLP)에 사용되던 Transformer 구조를 이미지에 적용한 방식
#이미지를 작은 패치로 나누고 각 패치를 1차원 벡터로 변환하는 과정이 있다.
#성능은 Transformer가 좋을 수 있으나 데이터 효율성 측면에서는 CNN이 적은 데이터로도 학습이 가능하다.