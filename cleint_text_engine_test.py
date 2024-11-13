import base64
import requests
import json
from PIL import Image
import io
# 서버 URL
url = "http://127.0.0.1:5000/"
# 테스트할 원본 이미지
image_path = "tera.jpg"
# 좌표 데이터 설정 (예제 좌표)
target = "bottle"
# 테스트 이미지 base64 인코딩
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
# JSON 데이터 구성
data = {
    "image": base64_image,
    "target": target
}
# 작업 상태 확인 함수
def check_segmentation_status():
    response = requests.get(url + "api/segmentation/state")
    print(response)
    if response.status_code == 200:
        status = response.json().get('status')
        print(status)
        if status == "1":
            print("Segmentation engine is currently busy.")
            return True  # 작업 중
        elif status == "0":
            print("Segmentation engine is available.")
            return False  # 작업 중 아님
        else:
            print("Unexpected response:", status)
            return True  # 알 수 없는 경우, busy 상태로 간주
    else:
        print("Error:", response.status_code, response.text)
        return True  # 요청 실패 시 busy 상태로 간주


# 상태 체크 및 POST 요청 보내기
if not check_segmentation_status():
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url + "api/segmentation/text_start", data=json.dumps(data), headers=headers)

    with open('./seg_text_input_json.json', 'w') as f:
        json.dump(data, f)

    # 응답 처리
    if response.status_code == 200:
        response_data = response.json()

        with open('./seg_text_output_json.json', 'w') as f:
            json.dump(response_data, f)

        if response_data.get('status') == 'success':
            # 마스크 이미지 디코딩
            mask_base64 = response_data['mask']
            mask_image_data = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_image_data)).convert("L")
            # 원본 이미지 로드
            original_image = Image.open(image_path).convert("RGBA")
            # 알파 채널을 마스크 이미지로 설정하여 원본 이미지에 적용
            mask_image_resized = mask_image.resize(original_image.size)
            result_image = Image.new("RGBA", original_image.size)
            for x in range(mask_image_resized.width):
                for y in range(mask_image_resized.height):
                    if mask_image_resized.getpixel((x, y)) > 0:
                        result_image.putpixel((x, y), original_image.getpixel((x, y)))
                    else:
                        result_image.putpixel((x, y), (0, 0, 0, 0))
            result_image.save('./text_result.png')
            print("Segmentation process finished.")
        else:
            print("Segmentation process failed.")
    else:
        print("error")
else:
    print("Engine is busy. Try again later.")