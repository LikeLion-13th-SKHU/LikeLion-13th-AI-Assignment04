import torch                                
# 딥러닝의 핵심 라이브러리(파이토치 라이브러리 로드, 텐서 조작, 모델 로딩, 추론 등에 사용됨.)
from torchvision import models, transforms  
# 토치비젼에서 모델과 전처리(transform) 도구를 가져옴.
# models : AI 모델을 불러올 때 사용, transforms : 이미지를 모델이 이해할 수 있도록 바꿔줌(전처리)
from PIL import Image                       
# PIL : 이미지 파일을 열고 처리할 수 있는 라이브러리리
import urllib,requests
import os
# urllib, requests, os : 파일 다운로드, 경로 확인 등에 쓰임

# 전처리 함수 정의 (ImageNet 표준)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Resize : ResNet-50은 입력 크기 224x224을 기대하므로 크기 조정
# Totensor : [0, 225] 이미지 -> [0.0, 1.0] 텐서로 변환, 픽셀값을 [0,1]로 정규화
# Normalize : imageNet 데이터셋의 평균, 표준편차로 정규화(학습된 환경과 일치시키기 위함.)


# 사전학습된 모델 로드
model = models.resnet50(pretrained=True)
model.eval()
# resnet50(pretrained=True) : ImageNet에서 학습된 ResNet-50 모델을 로딩. pretrainded=True로 ImageNet에서 학습된 가중치 사용.
# eval() : 추론 모드로 전환 (Dropout, BatchNorm 등 학습 중만 작동하는 레이어 비활성화) -> 비활성화되어 일관된 예측을 보장함.

# 클래스 레이블 로드 (ImageNet은 1000개 클래스)
LABELS_PATH = "imagenet_classes.txt"
if not os.path.exists(LABELS_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        LABELS_PATH
    )
## 로컬에 레이블 파일이 없으면 인터넷에서 다운로드 받아 저장함

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]
# ImageNet은 1000개의 클래스를 가지고 있으며, 이 텍스트 파일에는 각 라벨이 순서대로 저장되어 있음.
# 로컬에 없으면 URL에서 자동 다운로드합니다.

# 이미지 분류 함수
def classify_image(img_path):                               
# 로컬 경로의 이미지 분류를 수행하는 메인 함수(이미지를 열고, 전처리 한 뒤, AI에게 물어보고, 가장 가능성 높은 3가지를 보여주는 역할)
    if not os.path.exists(img_path):
        print("이미지 파일을 찾을 수 없습니다.")
        return

    image = Image.open(img_path).convert("RGB")
    # 이미지 열고 RGB 모드로 변환 (채널 부족/흑백 이미지 방지).
    input_tensor = preprocess(image).unsqueeze(0)
    # 전처리 적용 후 배치 차원 추가 (unsqeeze(0)) -> 모델은 [B, C, H, W] 형태 기대.

    with torch.no_grad():
        # torch.no_grad(): 추론 시 불필요한 연산(graph 저장 등) 비활성화
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        # softmax로 확률값으로 변환

    top3 = torch.topk(probs, 3)
    # 확률이 가장 높은 상위 3개 클래스 추출
    print("분석 결과 (Top 3):")
    for i in range(3):
        label = labels[top3.indices[i]]
        score = round(top3.values[i].item() * 100, 2)
        print(f"{i+1}. {label} ({score}%)")

# 실행
classify_image("./test.jpg")

# 202114014 이창현
# --- 작성 ---
# CNN이 고정된 크기의 입력을 받는다는 것 : ResNet-50은 224x224 크기의 이미지를 기대.
# 정규화의 중요성 : 평균과 표준편차로 정규화 -> 학습 안정성과 성능 향상에 필수.
# 이는 CNN이 픽셀값 분포에 민감하다는 것을 보여줌.
# -> CNN  입력 데이터는 반드시 정제된 형태로 제공되어야 한다는 것을 알 수 있음.
# probs = torch.nn.functional.softmax(output[0], dim=0)
# CNN의 최종 출력은 로짓(logit), 이를 softmax를 통해 확률 분포로 바꿈.
# -> CNN의 최종 출력이 각 클래스에 대한 예측 확률이라는 걸 직관적으로 이해하게 됨.
# Transfer Learning 개념
# "이미 학습된 모델의 지식을 새로운 문제에 재사용하는 것"
# 이 코드는 CNN을 처음부터 학습하지 않고, 사전 학습된 가중치를 불러와서 사용함
# -> CNN은 학습 비용이 높기 때문에 전이 학습이 매우 중요하다는 점을 자연스헙게 체험할 수 있음.

