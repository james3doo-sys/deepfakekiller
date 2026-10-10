"""
학습된 가중치를 weights/ 폴더로 내려받는 스크립트.

지금은 CNN/ViT/Temporal 헤드가 fine-tuning되지 않았기 때문에,
아래 WEIGHT_URLS는 실제 학습이 끝나고 결과물을 어딘가(예: Hugging Face
Hub, GitHub Releases)에 올린 뒤에 채워 넣어야 한다. URL이 비어 있으면
해당 가중치는 건너뛰고, deepfakekiller.py는 ImageNet 사전학습 backbone +
미학습 head로 동작한다는 경고를 출력한다.

사용법:
    python scripts/download_weights.py
    DEEPFAKEKILLER_WEIGHTS_DIR=/other/path python scripts/download_weights.py
"""
import os
import urllib.request

WEIGHTS_DIR = os.environ.get(
    "DEEPFAKEKILLER_WEIGHTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"),
)

# TODO: fine-tuning 완료 후 실제 다운로드 URL로 채우기 (현재는 비어 있음 = 미학습 상태)
WEIGHT_URLS = {
    "weights_cnn.pth": os.environ.get("DEEPFAKEKILLER_CNN_URL", ""),
    "weights_vit.pth": os.environ.get("DEEPFAKEKILLER_VIT_URL", ""),
    "weights_temporal.pth": os.environ.get("DEEPFAKEKILLER_TEMPORAL_URL", ""),
}


def download(filename, url):
    if not url:
        print(f"⚠️  {filename}: 다운로드 URL이 설정되지 않았습니다. 건너뜁니다.")
        return

    dest = os.path.join(WEIGHTS_DIR, filename)
    print(f"⬇️  {filename} 다운로드 중... ({url})")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"✅ {filename} 저장 완료: {dest}")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"가중치 저장 위치: {WEIGHTS_DIR}\n")

    any_url_set = any(WEIGHT_URLS.values())
    for filename, url in WEIGHT_URLS.items():
        download(filename, url)

    if not any_url_set:
        print(
            "\n현재 학습된 가중치가 없습니다. deepfakekiller.py는 ImageNet "
            "사전학습 backbone + 미학습(random) 분류 head로 동작하며, "
            "REAL/FAKE 판정은 신뢰할 수 없습니다."
        )


if __name__ == "__main__":
    main()
