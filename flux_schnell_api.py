import requests
import json
import base64
from PIL import Image
import io
import os
import random
from datetime import datetime

# 서버 URL
url = "http://localhost:5000/api/t2i"

# 결과물 저장 디렉토리 설정
output_dir = "generated_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def generate_image(
        prompt,
        negative_prompt="",
        width=1024,
        height=1024,
        guidance_scale=3.0,
        steps=20,
        cfg_scale=1.0,
        seed=random.randint(0, 2 ** 32 - 1)
):
    # 요청 데이터
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed
    }

    try:
        # POST 요청 보내기
        print("Sending request to server...")
        response = requests.post(url, json=data)

        # 응답 처리
        if response.status_code == 200:
            response_data = response.json()
            if response_data['status'] == 'success':
                # base64 이미지 정보 출력
                print("\nBase64 Image Data:")
                print(response_data['image'])

                # base64 이미지 디코딩
                image_data = base64.b64decode(response_data['image'])
                image = Image.open(io.BytesIO(image_data))

                # 파일명 생성 (타임스탬프 포함)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if seed is not None:
                    filename = f"generated_{timestamp}_seed{seed}.png"
                else:
                    filename = f"generated_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)

                # 이미지 저장
                image.save(filepath)
                print(f"\nImage saved to: {filepath}")
                print("Generation parameters:", response_data.get('parameters', {}))

                return image, filepath, response_data['image']  # base64 문자열도 반환
            else:
                print("Error:", response_data.get('message', 'Unknown error'))
                return None, None, None
        else:
            print("Request failed with status code:", response.status_code)
            print("Response:", response.text)
            return None, None, None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None


if __name__ == "__main__":
    # 테스트 프롬프트
    prompt = """Create a detailed UHD image of a rustic wooden table inside a cozy cabin, 
    featuring a wine bottle focus on the label ((labeled "Uplus" on text)) and glass, 
    adorned with intricate royal decorations, including golden engravings and gemstones, 
    shinning diamonds. The background is spectacular, masterpiece, fantasy, digital art, 
    highly detailed, overall detail, atmospheric lighting, Awash in a haze of light leaks 
    reminiscent of film photography, awesome background, highly detailed styling, studio photo, 
    intricate details, highly detailed, cinematic"""

    # 기본 파라미터로 이미지 생성
    print("Generating image with prompt:", prompt[:100] + "...")
    image, filepath, base64_string = generate_image(prompt)

    if image and filepath:
        print(f"\nSuccess! Image saved to: {filepath}")
    else:
        print("Image generation failed.")

    """
    # 커스텀 파라미터로 이미지 생성 예시
    image, filepath, base64_string = generate_image(
        prompt=prompt,
        width=1024,
        height=1024,
        guidance_scale=4.0,
        steps=30,
        cfg_scale=1.5,
        seed=12345
    )
    """