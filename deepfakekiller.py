from __future__ import annotations

import sys
import urllib.request
import os
try:
    import cv2
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import yt_dlp
    import numpy as np
    import pillow_heif
    import collections
    import mediapipe as mp
    from PIL import Image
except ImportError as e:
    print(f"필수 라이브러리가 설치되지 않았습니다: {e}")
    print("터미널에서 다음 명령어를 실행하세요: pip install opencv-python torch torchvision yt-dlp numpy pillow-heif mediapipe")
    sys.exit(1)

# 가중치 파일 위치: 기본값은 이 스크립트 옆의 weights/ 폴더.
# DEEPFAKEKILLER_WEIGHTS_DIR 환경변수로 다른 위치를 지정할 수 있음.
WEIGHTS_DIR = os.environ.get(
    "DEEPFAKEKILLER_WEIGHTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
)

# ---------------------------------------------
# 1. 모델 로드 및 전처리 설정
# ---------------------------------------------
# 맥북(M1/M2/M3) 전용 가속(mps) 및 일반 가속(cuda) 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"분석 장치: {device}")

# 얼굴 검출기 로드 (사람 및 동물의 얼굴 분석용)
# mediapipe 0.10.35+에서 legacy mp.solutions.face_detection이 제거되어
# Tasks API(FaceDetector)로 마이그레이션함. 모델(.tflite)은 최초 실행 시
# WEIGHTS_DIR에 자동 다운로드되고, 이후 실행부터는 캐시된 파일을 재사용한다.
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
FACE_MODEL_PATH = os.path.join(WEIGHTS_DIR, "blaze_face_short_range.tflite")


def _ensure_face_model() -> str | None:
    if os.path.exists(FACE_MODEL_PATH):
        return FACE_MODEL_PATH
    try:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        print("⬇️  얼굴 검출 모델(blaze_face_short_range.tflite) 다운로드 중...")
        req = urllib.request.Request(FACE_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(FACE_MODEL_PATH, "wb") as f:
            f.write(resp.read())
        return FACE_MODEL_PATH
    except Exception as e:
        print(f"⚠️ 얼굴 검출 모델 다운로드 실패: {e}")
        return None


try:
    _face_model_path = _ensure_face_model()
    if _face_model_path is None:
        raise RuntimeError("face detector model unavailable")
    _face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_face_model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.7,
    )
    face_detector = mp.tasks.vision.FaceDetector.create_from_options(_face_options)
    print("✅ MediaPipe 얼굴 검출기 로드 완료 (Tasks API)")
except Exception as e:
    print(f"⚠️ MediaPipe 로드 실패 (얼굴 정밀 분석 비활성화): {e}")
    face_detector = None

# 사람 및 동물 카테고리 정의
HUMAN_ANIMAL_LABELS = ['person', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# ---------------------------------------------
# 1.5. 시퀀스 분석용 모델 (Video-Transformer 스타일)
# ---------------------------------------------
class TemporalAnalyzer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, nhead=8, num_layers=2):
        super(TemporalAnalyzer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out = self.transformer(x)
        # 시퀀스의 마지막 시점 결과를 사용하여 판단
        return self.classifier(out[:, -1, :])

# COCO 데이터셋 클래스 이름 (사람 및 다양한 물체)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

try:
    import torchvision.models as models
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

    # 1. 객체 탐지 모델 로드 (사람 및 물체 탐지용)
    det_weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    detection_model = ssdlite320_mobilenet_v3_large(weights=det_weights).to(device)
    detection_model.eval()
    det_preprocess = det_weights.transforms()

    # 2. 딥페이크 앙상블 모델 구성
    # 모델 A: EfficientNet-V2-S (국소적 질감 분석 강점)
    model_cnn = models.efficientnet_v2_s(weights='DEFAULT')
    num_ftrs_cnn = model_cnn.classifier[1].in_features
    model_cnn.classifier[1] = nn.Linear(num_ftrs_cnn, 1)

    # 모델 B: ViT-B-16 (전역적 구조 분석 강점)
    model_trans = models.vit_b_16(weights='DEFAULT')
    num_ftrs_trans = model_trans.heads.head.in_features
    model_trans.heads.head = nn.Linear(num_ftrs_trans, 1)    

    # 3. 시퀀스 탐지 모델 초기화 (Liveness Detection용)
    model_temporal = TemporalAnalyzer(input_dim=num_ftrs_trans).to(device)

    cnn_weights = os.path.join(WEIGHTS_DIR, "weights_cnn.pth")
    vit_weights = os.path.join(WEIGHTS_DIR, "weights_vit.pth")
    temporal_weights = os.path.join(WEIGHTS_DIR, "weights_temporal.pth")

    if os.path.exists(cnn_weights):
        model_cnn.load_state_dict(torch.load(cnn_weights, map_location=device))
        print(f"✅ CNN 전용 가중치 로드 완료")
    if os.path.exists(vit_weights):
        model_trans.load_state_dict(torch.load(vit_weights, map_location=device))
        print(f"✅ ViT 전용 가중치 로드 완료")
    if os.path.exists(temporal_weights):
        model_temporal.load_state_dict(torch.load(temporal_weights, map_location=device))
        print(f"✅ Temporal 전용 가중치 로드 완료")

    if not os.path.exists(cnn_weights) and not os.path.exists(vit_weights):
        print("⚠️ 전용 가중치가 없습니다. 일반 사물 인식 기반으로 작동하여 생성형 AI 탐지율이 낮을 수 있습니다.")

    model_cnn = model_cnn.to(device).eval()
    model_trans = model_trans.to(device).eval()
    model_temporal = model_temporal.eval()

except Exception as e:
    print(f"⚠️ 모델 로드 중 오류 발생: {e}")
    model_cnn = None

# 앙상블 가중치 / 판정 임계값 — 객체별 판정, 전체 이미지 fallback, 영상 요약
# 집계가 전부 이 값 하나를 공유한다 (기존에는 세 군데가 서로 다른 값을 썼음).
ENSEMBLE_WEIGHTS = {"cnn": 0.45, "vit": 0.45, "ela": 0.10}
FAKE_THRESHOLD = 0.45


def compute_ai_score(score_cnn: float, score_trans: float, score_ela: float) -> float:
    return (
        score_cnn * ENSEMBLE_WEIGHTS["cnn"]
        + score_trans * ENSEMBLE_WEIGHTS["vit"]
        + score_ela * ENSEMBLE_WEIGHTS["ela"]
    )


def display_percent(status: str, ai_score: float) -> float:
    """판정 라벨(REAL/FAKE)에 대한 확신도를 퍼센트로 반환한다.
    ai_score는 항상 'FAKE일 확률'이라, REAL로 판정됐을 때 그대로 보여주면
    (예: REAL(44.1%)) "44.1% 확률로 진짜"처럼 오해하기 쉽다. 그래서 라벨에 맞게
    REAL이면 (1-ai_score), FAKE면 ai_score를 보여줘서 숫자와 라벨이 같은 방향을
    가리키게 한다."""
    return (ai_score if status == "FAKE" else (1 - ai_score)) * 100


# 이미지 전처리 설정
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_ela_score(image_bgr):
    """ 포렌식 기법: Error Level Analysis (ELA) 점수 계산 """
    try:
        quality = 90
        # 1. 이미지를 특정 품질로 압축하여 임시 저장 및 로드
        _, encoded_img = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        resaved_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # 2. 원본과 재압축본의 절대 차이 계산
        ela_diff = cv2.absdiff(image_bgr, resaved_img)
        # 3. 오차의 평균값을 점수로 환산 (조작된 부분은 오차가 큼)
        score = np.mean(ela_diff) / 255.0
        return min(score * 10, 1.0) # 스케일링
    except:
        return 0.5

def run_inference(frame):
    """ 이미지 내의 모든 사람과 물체를 찾아 각각 분석한 후 결과를 그린 프레임과 데이터를 반환 """
    if model_cnn is None: 
        return frame, [], None

    results_data = []
    feature_vector = None
    try:
        # 1. 객체 탐지 실행
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img_rgb)
        input_tensor = det_preprocess(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            detections = detection_model(input_tensor)[0]

        # 2. 감지된 각 객체에 대해 루프
        for i in range(len(detections['boxes'])):
            det_score = detections['scores'][i].item()
            if det_score < 0.5: continue # 확신도 50% 미만 무시

            box = detections['boxes'][i].cpu().numpy().astype(int)
            label_id = detections['labels'][i].item()
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]

            # 객체 영역 크롭
            x1, y1, x2, y2 = box
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0: continue

            # 3. 사람/동물일 경우 얼굴 검출 시도 (정밀 분석)
            analysis_target = crop
            # face_detector가 정상적으로 로드된 경우에만 실행
            if face_detector is not None and label_name in HUMAN_ANIMAL_LABELS:
                # MediaPipe는 RGB 이미지를 사용하므로 변환
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                results = face_detector.detect(mp_crop)

                if results.detections:
                    # Tasks API는 절대 픽셀 좌표(origin_x/y, width/height)를 반환
                    bbox = results.detections[0].bounding_box
                    ih, iw, _ = crop.shape
                    fx, fy = bbox.origin_x, bbox.origin_y
                    fw, fh = bbox.width, bbox.height

                    # 여백(Margin) 계산
                    mw, mh = int(fw * 0.2), int(fh * 0.2)
                    
                    # 분석 대상을 얼굴 영역으로 업데이트
                    face_crop = crop[max(0, fy-mh):min(ih, fy+fh+mh), 
                                     max(0, fx-mw):min(iw, fx+fw+mw)]
                    
                    if face_crop.size > 0:
                        analysis_target = face_crop
                    
                    # 얼굴 위치에 사각형 표시 (Cyan)
                    cv2.rectangle(frame, (x1+fx, y1+fy), (x1+fx+fw, y1+fy+fh), (255, 255, 0), 1)

            # 4. 분석 수행 (얼굴 혹은 물체 전체)
            target_rgb = cv2.cvtColor(analysis_target, cv2.COLOR_BGR2RGB)
            ai_input = preprocess(target_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                # 1. CNN 분석 (질감/노이즈)
                score_cnn = torch.sigmoid(model_cnn(ai_input)).item()
                # 2. ViT 분석 (구조적 일관성)
                score_trans = torch.sigmoid(model_trans(ai_input)).item()
                # 3. ELA 분석 (디지털 압축 흔적 포렌식)
                score_ela = calculate_ela_score(analysis_target)
                
                # [기업용 핵심] 시퀀스 분석을 위한 실제 특징 벡터 추출
                with torch.no_grad():
                    # ViT 내부 구조를 따라 특징 추출 (CLS 토큰 임베딩)
                    x = model_trans._process_input(ai_input)
                    n = x.shape[0]
                    batch_class_token = model_trans.class_token.expand(n, -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    x = model_trans.encoder(x)
                    # 마지막 레이어 통과 후 CLS 토큰(index 0) 추출
                    feature_vector = x[:, 0].cpu().squeeze() 

                # 종합 분석 (생성형 AI 탐지 최적화: ViT와 CNN 비중 강화)
                # 최신 생성 이미지는 ELA보다 ViT의 전역적 불일치 탐지가 더 정확함
                ai_score = compute_ai_score(score_cnn, score_trans, score_ela)

            # 포렌식 종합 판단 로직
            if ai_score > FAKE_THRESHOLD:
                status = "FAKE"
                color = (0, 0, 255) # Red
            else:
                status = "REAL"
                color = (0, 255, 0) # Green

            results_data.append((label_name, ai_score, status))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label_name}: {status}({display_percent(status, ai_score):.0f}%)"
            cv2.putText(frame, text, (x1, max(y1-10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. 아무 객체도 발견되지 않았을 경우 전체 이미지 분석 (Fallback)
        if not results_data:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ai_input = preprocess(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                score_cnn = torch.sigmoid(model_cnn(ai_input)).item()
                score_trans = torch.sigmoid(model_trans(ai_input)).item()
                score_ela = calculate_ela_score(frame) # 전체 이미지에 대한 ELA 점수 계산
                # 객체 탐지 시와 동일한 앙상블 가중치 적용
                ai_score = compute_ai_score(score_cnn, score_trans, score_ela)

                # Fallback 시에도 특징 추출
                x = model_trans._process_input(ai_input)
                batch_class_token = model_trans.class_token.expand(x.shape[0], -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = model_trans.encoder(x)
                feature_vector = x[:, 0].cpu().squeeze()

            if ai_score > FAKE_THRESHOLD:
                status = "FAKE"
                color = (0, 0, 255)
            else:
                status = "REAL"
                color = (0, 255, 0)

            results_data.append(("Whole Image", ai_score, status))
            cv2.putText(frame, f"Whole Image: {status}({display_percent(status, ai_score):.0f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame, results_data, feature_vector
    except Exception as e:
        print(f"추론 중 오류 발생: {e}")
        return frame, [], None
# ---------------------------------------------

def detect_from_image_url(url, display_results=True):
    print(f"\n[IMAGE] URL 분석시작: {url}")
    try:
        # 1. URL로부터 가상 메모리로 데이터 다운로드
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            # 2. 바이트 배열을 넘파이(NumPy) 배열로 변환
            image_array = np.asarray(bytearray(resp.read()), dtype = np.uint8)
        
        # 3. OpenCV 이미지 객체(BGR)로 디코딩
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("image url not available, please try again")
            return

        # 실제 추론 실행
        frame, detections, _ = run_inference(frame)
        
        if display_results:
            print(f"분석 완료: {len(detections)}개의 객체 발견")
            for name, score, status in detections:
                print(f" - [{name}]: {status} ({display_percent(status, score):.1f}%)")
    
            # 테스트용: 가져온 사진 띄워보기 (아무 키나 누르면 닫힘)
            cv2.imshow("Target Image", frame)
            cv2.waitKey(5000) # 5000ms(5초) 동안 대기 후 자동으로 닫힘 (0은 무한 대기)
            cv2.destroyAllWindows()
        else:
            print(f"분석 완료: {len(detections)}개의 객체 발견")
            
        # 모든 탐지 결과 리스트 반환
        return detections

    except Exception as e:
        print(f"error occurred:{e}")
        return []

def detect_from_local_image(path, display_results=True):
    """ 컴퓨터에 저장된 로컬 이미지 파일을 분석하는 함수 """
    print(f"\n[LOCAL IMAGE] 분석시작: {path}")
    
    # HEIC 포맷 지원을 위한 처리
    if path.lower().endswith(('.heic', '.heif')):
        try:
            heif_file = pillow_heif.read_heif(path)
            image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"HEIC 파일을 읽는 중 오류 발생: {e}")
            return
    else:
        frame = cv2.imread(path)

    if frame is None:
        print("로컬 이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
        return
    
    frame, detections, _ = run_inference(frame)
    
    if display_results:
        print(f"분석 완료: {len(detections)}개의 객체 발견")
        cv2.imshow("Local Image Detection", frame)
        cv2.waitKey(10000) # 5000ms(5초) 동안 대기 후 자동으로 닫힘
        cv2.destroyAllWindows()
        
    return detections

def detect_from_video(video_source):
    """ 로컬 영상 파일 경로 또는 스트리밍 주소를 받아서 프레임별로 탐지하는 함수 """
    print(f"\n[VIDEO] 분석시작: {video_source}")
    
    is_youtube = "youtube.com" in video_source or "youtu.be" in video_source
    actual_source = video_source

    # 유튜브 링크(쇼츠 포함)인 경우 스트리밍 주소 추출
    if is_youtube:
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': 'temp_youtube_video.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                # 403 Forbidden 에러 방지를 위한 추가 옵션
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'logtostderr': False,
                'no_color': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'referer': 'https://www.youtube.com/' # 유튜브에서 요청이 온 것처럼 보이게 합니다.
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("유튜브 영상을 다운로드 중입니다... (스트리밍 불안정 방지)")
                info = ydl.extract_info(video_source, download=True)
                actual_source = ydl.prepare_filename(info)
        except Exception as e:
            print(f"유튜브 처리 중 오류 발생: {e}")
            return
    else:
        actual_source = video_source

    cap = None
    frame_count = 0
    results = [] # 각 프레임의 분석 결과를 저장할 리스트
    try:
        cap = cv2.VideoCapture(actual_source)
        if not cap.isOpened():
            print("video not available, please try again")
            return [] # 영상 로드 실패 시 빈 리스트 반환 (finally에서 임시 파일 정리됨)

        # 시퀀스 분석을 위한 큐 (16프레임 단위)
        sequence_length = 16
        feature_buffer = collections.deque(maxlen=sequence_length)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("영상 끝 또는 프레임 읽기 실패.")
                break
            frame_count += 1

            # 1. 개별 프레임 분석 및 특징 추출
            frame, detections, feature_vec = run_inference(frame)

            # [기업용 핵심] 특징 벡터를 큐에 쌓음
            if feature_vec is not None:
                feature_buffer.append(feature_vec)

            # 2. 시퀀스(Temporal Analyzer) 분석 수행
            temporal_fake_prob = 0.0
            if len(feature_buffer) == sequence_length:
                # 버퍼에 쌓인 16프레임의 특징을 Transformer에 입력
                seq_input = torch.stack(list(feature_buffer)).unsqueeze(0).to(device)
                with torch.no_grad():
                    # TemporalAnalyzer를 통해 프레임간 부자연스러움 계산
                    temporal_logits = model_temporal(seq_input)
                    temporal_fake_prob = torch.sigmoid(temporal_logits).item()

            # 시퀀스 분석 결과 표시
            if temporal_fake_prob > 0.5:
                cv2.putText(frame, f"TEMPORAL ANOMALY: FAKE ({temporal_fake_prob*100:.1f}%)", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            for _, score, status in detections:
                results.append((score, status))

            # 영상 화면 표시 (테스트용)
            if len(feature_buffer) == sequence_length:
                cv2.putText(frame, f"Temporal Consistency: ANALYZING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

            cv2.imshow('Deepfake Detection Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("사용자에 의해 분석이 중단되었습니다.")
                break

            # 실제 탐지 시에는 여기서 모델을 호출합니다.
            # 테스트를 위해 100프레임마다 로그 출력
            if frame_count % 100 == 0:
                print(f"분석 중... ({frame_count} 프레임)")

        print(f" 총 {frame_count} 프레임 분석 완료!")
    finally:
        # cap 오픈 실패, 루프 중 예외 등 어떤 경로로 빠져나가든 항상 정리한다
        # (기존에는 cap.isOpened() 실패 시 여기 도달하지 못해 임시 파일이 안 지워졌음)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if is_youtube and os.path.exists(actual_source):
            os.remove(actual_source)
            print("임시 파일 삭제 완료.")

    return results # 모든 프레임의 분석 결과 리스트 반환

def start_detection(link):
    """ 링크의 성격에 따라 이미지 혹은 비디오 탐지 함수를 호출 """
    # .svg는 제외: OpenCV(cv2.imread/imdecode)가 벡터 포맷을 디코딩하지 못해
    # 여기 넣어봤자 항상 실패한다.
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

    results = None # 분석 결과를 저장할 변수
    # 1. 먼저 입력값이 내 컴퓨터에 실존하는 파일(로컬 파일)인지 확인합니다.
    if os.path.exists(link) and os.path.isfile(link):
        if any(link.lower().endswith(ext) for ext in image_extensions):
            results = detect_from_local_image(link, display_results=True)
        else:
            # 영상 확장자이거나 별도 확장자가 없더라도 파일이면 영상으로 시도
            results = detect_from_video(link)
    # 2. 로컬 파일이 아닐 때만 웹 링크(URL) 분석 수행
    elif any(link.lower().split('?')[0].endswith(ext) for ext in image_extensions) or "image" in link.lower():
        results = detect_from_image_url(link, display_results=True)
    # 유튜브 혹은 영상 확장자인 경우
    elif "youtube.com" in link or "youtu.be" in link or any(link.lower().endswith(ext) for ext in video_extensions):
        results = detect_from_video(link)
    else:
        print("알 수 없는 링크 형식입니다. 이미지 URL이나 유튜브 링크를 넣어주세요.")
        
    if results is not None:
        print("\n--- 분석 결과 요약 ---")
        if isinstance(results, list) and results:
            # 이미지 결과(3개 요소)와 영상 결과(2개 요소)를 구분하여 출력
            is_image = len(results[0]) == 3
            if is_image:
                for name, score, status in results:
                    print(f"[{name}]: {status} (확률: {display_percent(status, score):.1f}%)")
            else:
                fake_cnt = sum(1 for s, _ in results if s > FAKE_THRESHOLD)
                avg_prob = sum(s for s, _ in results) / len(results)
                print(f"총 {len(results)} 프레임 중 {fake_cnt}개 의심 (평균 확률: {avg_prob*100:.1f}%)")
        else:
            print("분석된 객체가 없거나 분석에 실패했습니다.")

if __name__ == "__main__":
    user_input = input("분석할 이미지/영상 링크를 입력하세요: ").strip()
    # 드래그 앤 드롭 시 경로 앞뒤에 생길 수 있는 따옴표 제거
    if (user_input.startswith("'") and user_input.endswith("'")) or \
       (user_input.startswith('"') and user_input.endswith('"')):
        user_input = user_input[1:-1]
    start_detection(user_input)
