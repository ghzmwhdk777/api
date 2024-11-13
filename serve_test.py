import base64
import time
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import torch
from custom_nodes.ComfyUI_segment_anything2.nodes import DownloadAndLoadSAM2Model, Sam2Segmentation, Florence2toCoordinates
from custom_nodes.ComfyUI_Florence2.nodes import Florence2ModelLoader, DownloadAndLoadFlorence2Model, Florence2Run

# Base64 이미지를 numpy 배열로 변환하는 함수 정의
def load_image_base64_to_numpy(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)
    image_np = image_np[None, ...]  # (H, W, C) -> (B, H, W, C) 변환, 배치 크기 B=1
    return image_np


# Flask 앱 생성
app = Flask(__name__)

# 작업 상태 플래그 변수
is_processing = False

# SAM2 모델 로드
sam2_loader = DownloadAndLoadSAM2Model()
sam2_model = sam2_loader.loadmodel(
    model="sam2_hiera_base_plus.safetensors",
    segmentor="single_image",
    device="cuda",
    precision="bf16"
)
F_model = DownloadAndLoadFlorence2Model().loadmodel(model='microsoft/Florence-2-base', precision="fp16",
                                                    attention="sdpa")


@app.route('/api/segmentation/point_start', methods=['POST'])
def segmentation_point():
    global is_processing
    if is_processing:
        return jsonify({'status': '1' if is_processing else '0'})
    else:
        is_processing = True  # 작업 시작
        try:
            time.sleep(10)
            # JSON 데이터 수신
            data = request.get_json()
            # Base64 이미지 문자열 가져오기 및 변환
            base64_image = data.get('image')
            if not base64_image:
                return jsonify({'error': 'No image provided'}), 400
            image_np = load_image_base64_to_numpy(base64_image)
            image_tensor = torch.from_numpy(image_np).float()

            coordinates_positive = str(data['coordinates_positive'])
            coordinates_negative = str(data['coordinates_negative'])
            # Sam2Segmentation 인스턴스 생성 및 segmentation 수행
            segmenter = Sam2Segmentation()

            # segment 메서드 호출 - coordinates 값은 numpy 배열로 전달
            mask_output = segmenter.segment(
                image=image_tensor,
                sam2_model=sam2_model[0],
                keep_model_loaded=True,
                coordinates_positive=coordinates_positive,
                coordinates_negative=coordinates_negative,
            )

            # 결과 처리 및 반환
            mask_image = mask_output[0].cpu().numpy()
            mask_image = np.squeeze(mask_image)
            mask_image = Image.fromarray((mask_image * 255).astype('uint8'))
            buffered = io.BytesIO()
            mask_image.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({'status': 'success', 'mask': mask_base64})

        except Exception as e:
            print("Error during segmentation:", str(e))
            return jsonify({'status': 'error', 'message': str(e)}), 500
        finally:
            is_processing = False  # 작업 완료

@app.route('/api/segmentation/text_start', methods=['POST'])
def segmentation_text():
    global is_processing
    if is_processing:
        return jsonify({'status': '1' if is_processing else '0'})
    else:
        is_processing = True  # 작업 시작
        try:
            time.sleep(10)
            # JSON 데이터 수신
            data = request.get_json()
            # Base64 이미지 문자열 가져오기 및 변환
            base64_image = data.get('image')
            if not base64_image:
                return jsonify({'error': 'No image provided'}), 400
            image_np = load_image_base64_to_numpy(base64_image)
            image_tensor = torch.from_numpy(image_np).float()
            # F_model = DownloadAndLoadFlorence2Model().loadmodel(model='microsoft/Florence-2-base', precision = "fp16", attention = "sdpa")
            target = data['target'] # bottle
            _, __, __, F_out = Florence2Run().encode(image=image_tensor, florence2_model=F_model[0],text_input=target,task="caption_to_phrase_grounding", fill_mask = True)
            # print(Fout)
            Fco_out,_  = Florence2toCoordinates().segment(data = F_out, index="0")
            # Sam2Segmentation 인스턴스 생성 및 segmentation 수행
            segmenter = Sam2Segmentation()
            # segment 메서드 호출 - coordinates 값은 numpy 배열로 전달
            mask_output = segmenter.segment(
                image=image_tensor,
                sam2_model=sam2_model[0],
                keep_model_loaded=True,
                coordinates_positive=Fco_out,
            )
            # 결과 처리 및 반환
            mask_image = mask_output[0].cpu().numpy()
            mask_image = np.squeeze(mask_image)
            mask_image = Image.fromarray((mask_image * 255).astype('uint8'))
            buffered = io.BytesIO()
            mask_image.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({'status': 'success', 'mask': mask_base64})

        except Exception as e:
            print("Error during segmentation:", str(e))
            return jsonify({'status': 'error', 'message': str(e)}), 500
        finally:
            is_processing = False  # 작업 완료



# 작업 상태 확인 엔드포인트
@app.route('/api/segmentation/state', methods=['GET'])
def get_state():
    return jsonify({'status': '1' if is_processing else '0'})


# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
