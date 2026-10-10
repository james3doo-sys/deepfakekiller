# deepfakekiller

이미지/영상(로컬 파일, 이미지 URL, 유튜브 링크)에서 사람·동물을 탐지하고,
각 영역에 대해 CNN(EfficientNet-V2-S) + ViT(ViT-B-16) + ELA 포렌식 스코어를
앙상블하여 딥페이크 여부를 판정하는 스크립트. 영상의 경우 16프레임 단위로
Temporal Transformer를 돌려 프레임 간 부자연스러움도 함께 본다.

## 현재 상태: 2차 fine-tuning 완료 (일반화 격차, 숫자로 확인됨)

`model_cnn`(EfficientNet-V2-S), `model_trans`(ViT-B-16)는 ArtiFact(Kaggle
`awsaf49/artifact-dataset`) + Defactify(`Rajarshi-Roy-research/Defactify_Image_Dataset`,
SD3/SD2.1/SDXL/DALL-E 3/MidJourney v6 포함)를 병합한 데이터로 기존 1차
체크포인트 위에서 이어서(warm-start) fine-tuning했다 (2026-08-24, Colab).
가중치는 `weights/weights_cnn.pth`, `weights/weights_vit.pth`로 저장돼 있고
실행 시 자동으로 로드된다. 이전 1차 가중치는 `weights_cnn_10_a.pth` 등으로
보존돼 있다.

### 벤치마크 결과 (2026-08-24 학습분)

**In-distribution** (학습에 쓴 ArtiFact+Defactify의 val split, n=4000):

| 조합 | AUC | EER |
|---|---|---|
| CNN only | 0.9513 | 0.1205 |
| ViT only | 0.9469 | 0.1325 |
| CNN + ViT | 0.9606 | 0.1065 |
| CNN + ViT + ELA (기본값) | 0.9604 | 0.1055 |

**OOD 일반화** (`ComplexDataLab/OpenFake` core/test — Gemini nano-banana/
nano-banana-pro 포함, **학습에 전혀 안 쓴** held-out, n=1000):

| 조합 | AUC | EER |
|---|---|---|
| CNN only | 0.6314 | 0.4150 |
| ViT only | 0.5947 | 0.4390 |
| CNN + ViT | 0.6337 | 0.4081 |
| CNN + ViT + ELA (기본값) | 0.6274 | 0.4120 |

**해석**: in-distribution AUC(~0.96)와 OOD AUC(~0.63, 랜덤=0.5에 가까움) 사이에
큰 격차가 있다. 즉 이번 재학습은 "학습 데이터와 비슷한 최신 생성기"에는 잘
통했지만, **Gemini처럼 아예 학습에 안 보여준 생성기에는 여전히 거의 못
통한다** — "다른 최신 생성기로 학습하면 Gemini한테도 어느 정도 전이될 것"
이라는 가설이 이 수치로는 기각된 셈이다. 아래 "알려진 한계"에 이어지는
설명 참고.

**알려진 한계 (일반화 격차, 실측 확인됨)**: 위 벤치마크에서 보듯, 학습 데이터와
분포가 다른 생성기(특히 Gemini)에는 여전히 거의 랜덤 수준으로만 판별한다.
개선하려면 Gemini(nano-banana/nano-banana-pro) 샘플 자체를 학습 데이터에도
포함시켜야 할 것으로 보인다 — 예: `CONFIG["extra_hf_dataset"]`에 OpenFake의
`train` 스플릿(현재는 `core/test`만 OOD 검증용으로 씀)을 추가하는 방향.

`model_temporal`(영상 프레임 간 일관성 분석)은 **아직 학습되지 않았다** —
영상 분석 결과의 Temporal Anomaly 판정은 여전히 신뢰할 수 없다. 가중치
파일이 없는 항목은 실행 시 "⚠️ 전용 가중치가 없습니다" 경고가 뜨는 게
정상이며, 이 경고가 뜨는 동안은 해당 부분 결과를 신뢰하면 안 된다.

## 설치

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python torch torchvision yt-dlp numpy pillow-heif mediapipe
```

## 실행

```bash
python deepfakekiller.py
```

이미지 URL, 유튜브 링크, 로컬 이미지/영상 경로 중 하나를 입력하면 분석 결과를
보여준다.

## 가중치

가중치 경로는 하드코딩되어 있지 않고 `DEEPFAKEKILLER_WEIGHTS_DIR` 환경변수로
지정하며, 지정하지 않으면 스크립트 옆 `weights/` 폴더를 기본값으로 쓴다.

```bash
python scripts/download_weights.py
```

현재는 이 스크립트로 다운로드하지 않고, `scripts/train_colab.ipynb`로 Colab에서
직접 fine-tuning한 뒤 `weights_cnn.pth`/`weights_vit.pth`를 다운로드해 수동으로
`weights/`에 넣는 방식을 쓰고 있다. `download_weights.py`는 나중에 학습
결과물을 Hugging Face Hub나 GitHub Releases 같은 곳에 올려 공유하고 싶을 때
쓰는 경로로 남겨둔 것이며, 지금은 `WEIGHT_URLS`가 비어 있어 모든 파일을
건너뛴다는 메시지만 출력한다.

## 학습 노트북 (`scripts/train_colab.ipynb`)

CNN/ViT head fine-tuning은 로컬이 아니라 Colab에서 돌린다 (데이터셋/학습
과정이 로컬 디스크를 전혀 쓰지 않게 하기 위함). 데이터는 Kaggle
`awsaf49/artifact-dataset`의 `metadata.csv`(`image_path`, `target` 컬럼)를
읽어 라벨링하며, `target == 0`은 real(0), 그 외 값은 전부 fake(1)로 강제
이진화한다 (해당 데이터셋의 `target`이 문서와 달리 생성기별 ID가 섞여 있어
그대로 쓰면 `roc_auc_score`가 다중 클래스로 오인해 에러가 난다). 학습 루프는
매 epoch 최고 val AUC 갱신 시 가중치를 디스크에도 바로 저장해, 런타임 재시작
후에도(완전히 새 런타임이 아니라면) 다운로드 셀만 다시 돌리면 복구된다.

## 로드맵

- [x] 공개 벤치마크 연결 (ArtiFact) — 영상 벤치마크(FaceForensics++/Celeb-DF)는
      아직 미착수
- [x] CNN head, ViT head 1차 fine-tuning — 다만 최신 생성기(Gemini 등)에 대한
      일반화는 안 됨 (위 "알려진 한계" 참고)
- [x] 벤치마크/ablation 계산 코드 작성 (`scripts/train_colab.ipynb` 6번 섹션) —
      CNN only / ViT only / CNN+ViT / +ELA 조합별 AUC·EER을 val split에서
      계산해 마크다운 표로 출력. **아직 실행은 안 함** — 실제 숫자는 노트북을
      한 번 돌려야 나옴.
- [x] 기존 체크포인트에서 이어서 학습(warm-start) 지원 — `CONFIG["resume_from"]`
      에 기존 `weights_cnn.pth`/`weights_vit.pth` 경로를 넣으면 처음부터 다시
      학습하지 않고 이어서 fine-tuning
- [x] 최신 생성기(SD3/SDXL/DALL-E 3/MidJourney v6) 데이터셋 소스 추가 —
      `CONFIG["extra_hf_dataset"]`에 `Rajarshi-Roy-research/Defactify_Image_Dataset`
      기본 지정, ArtiFact와 병합해서 학습하도록 데이터 로딩부 확장
- [x] Cross-dataset(OOD) 일반화 벤치마크 코드 추가 (`train_colab.ipynb` 7번 섹션) —
      `ComplexDataLab/OpenFake`의 `core/test` 스플릿(학습에 전혀 안 씀, Gemini의
      nano-banana/nano-banana-pro 포함)으로 채점. 무료, API 키/유료 구독 불필요
      (CC-BY-SA-4.0, 다만 proprietary 생성기 쪽 fake 샘플은 상업적 재배포 불가 —
      평가 용도로 로컬에서 쓰는 지금 용도는 문제없음). **이것도 아직 실행은 안 함.**
- [x] 재학습 + in-distribution 벤치마크 + OOD 벤치마크 실제 실행 (2026-08-24) —
      결과는 위 "현재 상태" 참고. In-distribution AUC 0.96, OOD(Gemini 포함) AUC
      0.63로 일반화 격차가 실측으로 확인됨.
- [x] `extra_hf_datasets`를 리스트로 확장 + OpenFake `train` 스플릿(Gemini 포함,
      7번 셀이 쓰는 `core/test`와는 안 겹침) 추가 — Gemini 샘플 자체를 학습에도
      넣도록 설정. **아직 실행은 안 함.**
- [ ] **위 설정으로 다시 재학습 → OOD 벤치마크 재검증** — Gemini AUC가 0.63에서
      실제로 올라가는지 확인해야 함
- [ ] Temporal Transformer fine-tuning (FaceForensics++/Celeb-DF)

## Known Issues

해결됨 (이전에는 여기 있었음):
- ~~MediaPipe 얼굴 검출 API 불일치~~ — Tasks API(`FaceDetector`)로 마이그레이션 완료.
  모델(`blaze_face_short_range.tflite`)은 최초 실행 시 `weights/`에 자동 다운로드됨.
- ~~임계값/앙상블 가중치 불일치~~ — 객체별 판정/전체 이미지 fallback/영상 요약
  집계가 전부 `ENSEMBLE_WEIGHTS = {cnn: 0.45, vit: 0.45, ela: 0.10}`,
  `FAKE_THRESHOLD = 0.45` 하나를 공유하도록 통일함 (`compute_ai_score()`).
  다만 이 숫자 자체가 벤치마크로 검증된 최적값은 아니고 기존 휴리스틱을
  통일한 것 — 아래 로드맵의 벤치마크 작업이 끝나면 재조정 필요.
- ~~유튜브 임시 파일 미정리~~ — `detect_from_video`를 `try/finally`로 감싸서
  `cv2.VideoCapture` 오픈 실패/루프 중 예외와 무관하게 항상 정리되도록 수정.
- ~~`.svg` 지원 확장자 오기재~~ — 목록에서 제거 (OpenCV가 애초에 디코딩 못 함).

아직 남아있음:
- **최신 생성기(특히 Gemini) 일반화 격차** (가장 중요, 실측 확인됨) — Defactify를
  추가해서 재학습한 뒤에도 OOD 벤치마크(OpenFake core/test, Gemini nano-banana
  포함)에서 AUC 0.63으로 랜덤 수준에 가깝다 (in-distribution AUC 0.96과 대비).
  즉 "다른 최신 생성기로 학습하면 Gemini에도 어느 정도 전이된다"는 가설은
  기각됐고, Gemini 샘플 자체를 학습 데이터에 넣어야 해소될 것으로 보인다.
- `model_temporal`은 여전히 미학습 상태.
