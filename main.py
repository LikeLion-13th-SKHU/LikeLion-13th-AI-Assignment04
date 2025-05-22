import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
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


# 202014057 이하늘늘
# --- 작성 ---
# 이미지를 필터(커널)을 통해 특성을 추출 분석하는 모델이다. 
# 필터(커널)은 이미지를 순회?하면서 이미지의 특성을 추출한다. 
# 이 특징들을 풀링(pooling) 으로 요약한 후, 완전 연결층(Fully Connected Layer) 을 통해 최종적으로 분류 등의 작업을 수행하는 모델입니다