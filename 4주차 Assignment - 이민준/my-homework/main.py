import torch  # PyTorch 딥러닝 프레임워크
from torchvision import models, transforms  # 사전학습 모델 및 이미지 전처리 도구
from PIL import Image  # 이미지 열기 및 처리용 라이브러리
import urllib.request  # 인터넷에서 파일 다운로드용
import os  # 파일 경로 및 존재 여부 확인용

# 전처리 함수 정의 (ImageNet에 맞는 표준화 방식)
# 이미지 크기를 224x224로 조정하고, 텐서로 변환한 뒤 정규화
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # RGB 채널 평균값 (ImageNet 기준)
        std=[0.229, 0.224, 0.225]    # RGB 채널 표준편차 (ImageNet 기준)
    )
])

# 사전학습된 ResNet-50 모델 로드
# ImageNet 데이터셋에서 학습된 가중치를 불러옴
model = models.resnet50(pretrained=True)
model.eval()  # 평가 모드로 설정 (dropout, batchnorm 비활성화)

# ImageNet 클래스 이름(1000개)을 담은 텍스트 파일 경로
LABELS_PATH = "imagenet_classes.txt"

# 파일이 없으면 인터넷에서 다운로드
if not os.path.exists(LABELS_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        LABELS_PATH
    )

# 클래스 레이블을 리스트로 로드
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]  # 각 줄의 문자열을 리스트에 저장

# 이미지 분류 함수 정의
def classify_image(img_path):
    # 이미지 파일 존재 여부 확인
    if not os.path.exists(img_path):
        print("이미지 파일을 찾을 수 없습니다.")
        return

    # 이미지를 열고 RGB 형식으로 변환
    image = Image.open(img_path).convert("RGB")
    
    # 전처리 수행 및 배치 차원 추가 (1 x 3 x 224 x 224 형태)
    input_tensor = preprocess(image).unsqueeze(0)

    # 모델 추론 (grad 계산 비활성화)
    with torch.no_grad():
        output = model(input_tensor)  # 모델 출력 (로짓)
        probs = torch.nn.functional.softmax(output[0], dim=0)  # 확률로 변환

    # 확률이 높은 Top 3 클래스 추출
    top3 = torch.topk(probs, 3)
    print("분석 결과 (Top 3):")
    
    # 결과 출력
    for i in range(3):
        label = labels[top3.indices[i]]  # 예측 클래스 이름
        score = round(top3.values[i].item() * 100, 2)  # 확률(%)로 변환
        print(f"{i+1}. {label} ({score}%)")

# 함수 실행 - 'test.jpg' 파일을 분류
classify_image("test.jpg")



# 202014068 이민준
# --- 작성 ---
# 여기에 본인이 이해한 CNN 개념을 작성해주세요
# '합성곱 신경망'을 의미하는 CNN에 대해서 이미지와 같은 2차원 데이터들을 처리하기에 매우 적합한 딥러닝 모델이라는 것을 다시 한번 깨달을 수 있었습니다.
# (선택) 본인이 이해한 Transformer 개념을 작성해주세요!
# '자연어 처리'를 의미하는 NLP 분야에서 주로 사용되고 있다는 것을 직접 느낄 수 있었으며 이미지를 작은 패치들로 나눈 후 시퀀스로 변환한 뒤 처리한다는 것을 알 수 있었습니다.