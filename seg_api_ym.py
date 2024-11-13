import base64
import requests
import json
from PIL import Image
import io

# 서버 URL
url = "http://localhost:5000/api/segmentation"

# 테스트할 이미지 로드 및 Base64 인코딩
image_path = "E:\\ComfyUI_windows_portable\\1111_api\\img\\test.png"
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# 좌표 데이터 설정 (리스트 형식으로 직접 전달)
coordinates_positive = [{'x': 50, 'y': 100}, {'x': 150, 'y': 200}]
coordinates_negative = [{'x': 70, 'y': 90}]

# JSON 데이터 구성
data = {
    "image": base64_image,
    "coordinates_positive": coordinates_positive,
    "coordinates_negative": coordinates_negative
}

# POST 요청 보내기
response = requests.post(url, json=data)

# 응답 처리
if response.status_code == 200:
    response_data = response.json()
    if response_data['status'] == 'success':
        # 마스크 이미지 디코딩
        mask_base64 = response_data['mask']
        mask_image_data = base64.b64decode(mask_base64)
        mask_image = Image.open(io.BytesIO(mask_image_data))  # BytysIO -> BytesIO로 수정

        # 마스크 이미지 표시
        mask_image.show()
    else:
        print("Error:", response_data.get('message', 'Unknown error'))
else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)  # 에러 메시지 상세 출력