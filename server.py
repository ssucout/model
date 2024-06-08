import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
#모델
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 기존 fc 레이어의 출력 크기를 확인하여 새로운 레이어 추가
        num_ftrs = self.resnet.fc.in_features
        # 원래 ResNet50의 마지막 fc 레이어를 제거하고 새로운 레이어 추가
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


app = Flask(__name__)
CORS(app)

# 모델 불러오는 부분
# 모델 경로 설정해주기
num_classes = 7
model = CustomResNet(num_classes=num_classes)
model.load_state_dict(torch.load('./myModel/model.pth', map_location=torch.device('cpu')))
model.eval()

# 사진 전처리(224 x 224로 만들어주기)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



@app.route('/face', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item() + 1

        print(f'Predicted class: {predicted_class}')
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333)