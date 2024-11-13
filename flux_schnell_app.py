import base64
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import json
import numpy as np
import random
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
from nodes import CLIPTextEncode, VAEDecode, CheckpointLoaderSimple, LoraLoader
from comfy.samplers import KSampler

app = Flask(__name__)

# 샘플러와 스케줄러 옵션 정의
SAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                 "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
                 "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

# 전역 변수로 모델들을 초기화
model = None
clip = None
vae = None


def conditioning_set_guidance(conditioning, guidance):
    """FluxGuidance의 append 메서드 구현"""
    return node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})


def initialize_models():
    global model, clip, vae

    try:
        with torch.no_grad():
            # CheckpointLoaderSimple 사용
            checkpoint_loader = CheckpointLoaderSimple()
            model, clip, vae = checkpoint_loader.load_checkpoint(ckpt_name="flux1-schnell-fp8.safetensors")

            # LoraLoader 사용
            lora_loader = LoraLoader()
            model, clip = lora_loader.load_lora(
                model=model,
                clip=clip,
                lora_name="FluxDFaeTasticDetails.safetensors",
                strength_model=1.0,
                strength_clip=1.0
            )

            return model, clip, vae
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def create_empty_latent(width=1024, height=1024, batch_size=1):
    """SD3용 빈 레이턴트 이미지 생성"""
    device = comfy.model_management.intermediate_device()
    # Flux/SD3 모델용 16채널 latent
    latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=device)
    return {"samples": latent}


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
    """KSampler 실행"""
    from nodes import common_ksampler

    with torch.no_grad():
        result = common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=True
        )[0]  # common_ksampler는 튜플을 반환하므로 첫 번째 요소 사용

        # result는 이미 {"samples": tensor} 형태의 딕셔너리입니다
        return result["samples"]


@app.route('/api/t2i', methods=['POST'])
def text2image():
    try:
        with torch.no_grad():
            data = request.get_json()
            prompt = data.get('prompt', '')
            negative_prompt = data.get('negative_prompt', '')
            width = data.get('width', 1024)
            height = data.get('height', 1024)
            guidance_value = data.get('guidance_scale', 3.0)
            steps = data.get('steps', 20)
            cfg = data.get('cfg_scale', 8.0)
            seed = data.get('seed', random.randint(0, 2 ** 32 - 1))
            sampler_name = data.get('sampler_name', 'euler')
            scheduler = data.get('scheduler', 'normal')
            denoise = data.get('denoise', 1.0)

            if sampler_name not in SAMPLER_NAMES:
                return jsonify({'error': f'Invalid sampler_name. Must be one of: {SAMPLER_NAMES}'}), 400
            if scheduler not in SCHEDULER_NAMES:
                return jsonify({'error': f'Invalid scheduler. Must be one of: {SCHEDULER_NAMES}'}), 400

            print(f"Using seed: {seed}")

            # CLIP Text Encode
            clip_encode = CLIPTextEncode()
            positive_cond = clip_encode.encode(clip, prompt)[0]
            negative_cond = clip_encode.encode(clip, negative_prompt)[0]

            # FluxGuidance 적용
            positive_cond = conditioning_set_guidance(positive_cond, guidance_value)

            # SD3용 빈 레이턴트 생성
            latent = create_empty_latent(width=width, height=height)

            # Sampling 실행
            samples = sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_cond,
                negative=negative_cond,
                latent=latent,
                denoise=denoise
            )

            # VAE Decode (samples를 다시 딕셔너리로 감싸줍니다)
            vae_decoder = VAEDecode()
            decoded = vae_decoder.decode(vae, {"samples": samples})[0]

            # 텐서 형태 변환
            decoded = decoded.cpu().numpy()
            # (1, 1, height, width, 3) -> (height, width, 3)
            if len(decoded.shape) == 5:
                decoded = decoded[0, 0]
            elif len(decoded.shape) == 4:
                decoded = decoded[0]

            # uint8로 변환하고 범위를 0-255로 조정
            decoded = np.clip(decoded * 255, 0, 255).astype(np.uint8)

            # 이미지로 변환
            image = Image.fromarray(decoded)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return jsonify({
                'status': 'success',
                'image': img_str,
                'parameters': {
                    'width': width,
                    'height': height,
                    'guidance_scale': guidance_value,
                    'steps': steps,
                    'cfg_scale': cfg,
                    'seed': seed,
                    'sampler_name': sampler_name,
                    'scheduler': scheduler,
                    'denoise': denoise
                }
            })

    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("Initializing models...")
    model, clip, vae = initialize_models()
    print("Models loaded successfully!")
    app.run(host='0.0.0.0', port=5000, debug=True)