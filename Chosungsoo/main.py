import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import os

# 전처리 함수 정의 (ImageNet 표준)
# 전처리는 왜 필요할까?
# 컴퓨터는 우리의 언어를 이해하지 못하기 때문에 컴퓨터가 이해할 수 있는 과정이 필요합니다. 즉, 이미지를 일정한 형태로 만들어주는 것이지요.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 사전학습된 모델 로드

# pretrained는 무엇인가?
# 우리 같은 학생이 많은 데이터를 모으고 모델을 학습시키는 것은 쉽지 않은 일입니다.
# pretrained = True를 사용하면 이미 학습된 좋은 모델을 사용할 수 있기 때문에 적은 자원으로도 이러한 과제를 수행할 수 있습니다.
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

    # 이 부분은 왜 필요한가?
    # 이미 학습된 모델로 예측을 하는 것이기 때문에 기울기 계산은 건너뛰는 것입니다. (불필요한 계산 스킵킵)
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

# 202214156 조성수
# CNN은 주로 이미지나 영상과 같은 시각 데이터를 처리하고 분석하는 데 사용되는 인공 신경망입니다.
# 예를 들어 골든 리트리버의 사진을 본다고 하면, 작은 부분의 특징을 찾아내어 이미지를 분류해내는 것이지요.
# 왜 사용할까요? 이미지 데이터는 매우 크기 때문에 모든 픽셀을 다 분석하는 것은 힘든 일입니다. 특징만 뽑아내서 분석한다면 훨씬 효율적이겠지요.

# Transfomer는 전체를 넓게 보고 관계를 파악하는 모델입니다. 각 이미지 조각들이 어떤 관계를 가지는지 파악합니다.
# 왜 필요할까요? 이미지 조각들의 관계를 파악하여 이미지를 분류하는데 쓰이는 것입니다.