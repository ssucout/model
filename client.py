import requests

def test_flask_server(image_path, server_url='http://54.180.25.63:3333/face'):
    # 이미지 파일을 읽어서 전송


    with open(image_path, 'rb') as img_file:
        files = {'file': img_file}
        response = requests.post(server_url, files=files)
    print(img_file)
    # 서버 응답 출력
    if response.status_code == 200:
        print('Prediction:', response.json())
    else:
        print('Error:', response.json())

# 테스트할 이미지 파일 경로
image_path = './clientTestData/image.png'

# Flask 서버 테스트 실행
test_flask_server(image_path)
