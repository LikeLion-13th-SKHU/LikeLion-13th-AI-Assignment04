import torch
from torchvision import models, transforms
from PIL import Image#pil : 
import urllib.request# 인터넷에서 파일을 다운로드.
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

# 2202415003 홍다보미
# --- 작성 ---
# 여기에 본인이 이해한 CNN 개념을 작성해주세요!
# CNN은 이미지나 영상등등을 분석하는 것. 
# 과정은 전처리를 하고 필터?를 써서 특징을 추출하고 음수가 나오면 0으로 바꾸고 특징을 간단하게 만들기까지 반복을 하고다음 1차원 벡터로 바꾸고 ㄱㅖ산을 해서 ... 분석을 하는게.. 맞나요..?
# 이미지 위에 필터를 올려서 특징을 추출하는 과정에서 그 필터에 있는 101010101 같은 값은 모델이 자기가 학습해서 나오는 값이면..그걸 어떻게 학습을 하는지 모르겠습니다..!



# (선택) 본인이 이해한 Transformer 개념을 작성해주세요!