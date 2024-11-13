import base64
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import json
import numpy as np
from custom_nodes.ComfyUI_segment_anything_2.nodes import DownloadAndLoadSAM2Model, Sam2Segmentation

app = Flask(__name__)

# SAM2 모델을 전역 변수로 초기화
sam2_loader = None
sam2_model = None


def initialize_model():
    global sam2_loader, sam2_model
    sam2_loader = DownloadAndLoadSAM2Model()
    sam2_model = sam2_loader.loadmodel(
        model="sam2_hiera_base_plus.safetensors",
        segmentor="single_image",
        device="cuda",
        precision="bf16"
    )[0]


def process_image(image):
    """이미지를 RGB로 변환하고 텐서로 만듭니다."""
    # RGBA 이미지를 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # numpy 배열로 변환
    image_np = np.array(image)

    # 텐서로 변환하고 정규화
    image_tensor = torch.from_numpy(image_np).float() / 255.0

    # [H, W, C] -> [B, H, W, C] 형태로 변환
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)

    print("Final tensor shape:", image_tensor.shape)
    return image_tensor


@app.route('/api/segmentation', methods=['POST'])
def segmentation():
    try:
        data = request.get_json()

        # JSON에서 base64 이미지 문자열 가져오기
        base64_image = data.get('image')
        if not base64_image:
            return jsonify({'error': 'No image provided'}), 400

        # Base64 이미지를 디코딩하여 PIL 이미지로 변환
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))

        # 이미지 처리
        image_tensor = process_image(image)

        # 좌표 데이터 처리
        coordinates_positive = data.get('coordinates_positive', [])
        coordinates_negative = data.get('coordinates_negative', [])

        print("Processing image shape:", image_tensor.shape)
        print("Coordinates positive:", coordinates_positive)
        print("Coordinates negative:", coordinates_negative)

        # Sam2Segmentation 인스턴스 생성 및 segmentation 수행
        segmenter = Sam2Segmentation()
        mask_output = segmenter.segment(
            image=image_tensor,
            sam2_model=sam2_model,
            keep_model_loaded=True,
            coordinates_positive=json.dumps(coordinates_positive),
            coordinates_negative=json.dumps(coordinates_negative),
            individual_objects=True
        )[0]

        if mask_output is not None and mask_output.numel() > 0:
            # 텐서를 numpy 배열로 변환
            mask_np = mask_output.cpu().numpy()

            # 차원이 3D 이상인 경우 첫 번째 마스크만 사용
            if len(mask_np.shape) > 2:
                mask_np = mask_np[0]

            # numpy 배열을 PIL 이미지로 변환
            mask_image = Image.fromarray((mask_np * 255).astype('uint8'))

            # PIL 이미지를 base64로 인코딩
            buffered = io.BytesIO()
            mask_image.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                'status': 'success',
                'mask': mask_base64
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No valid mask generated'
            }), 500

    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()  # 상세한 에러 정보 출력
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    initialize_model()  # 서버 시작 시 모델 초기화
    app.run(host='0.0.0.0', port=5000, debug=True)