"""input/의 모든 WAV를 잡음 제거해 output/에 같은 파일명으로 저장합니다.

학습 없이 weights/의 가중치만 불러와 추론합니다.
사용법 (저장소 최상위 폴더에서):
    python inference.py
옵션:
    python inference.py --input 입력폴더 --output 출력폴더 --no-tta --device cpu
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model import MODEL_CLASSES, SAMPLE_RATE

# 모든 기본 경로는 이 스크립트가 있는 폴더 기준입니다. 절대 경로를 고칠 필요가 없습니다.
ROOT = Path(__file__).resolve().parent
TTA_SHIFTS = (0, 32, 64, 96)   # 입력을 hop(128)보다 작게 밀어 처리한 뒤 평균 (시간 이동 TTA)


def parse_args():
    parser = argparse.ArgumentParser(description="딥러닝 음성 잡음 제거 추론 (앙상블)")
    parser.add_argument("--input", type=Path, default=ROOT / "input", help="noisy WAV 폴더")
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="복원 WAV 저장 폴더")
    parser.add_argument("--config", type=Path, default=ROOT / "weights" / "ensemble.json",
                        help="앙상블 구성 파일")
    parser.add_argument("--device", default=None, help="cuda 또는 cpu (기본: 자동 선택)")
    parser.add_argument("--no-tta", action="store_true", help="시간 이동 TTA를 끄고 빠르게 실행")
    return parser.parse_args()


def load_ensemble(config_path, device):
    if not config_path.is_file():
        raise FileNotFoundError(f"앙상블 구성 파일이 없습니다: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    members = []
    for item in config["members"]:
        weight_path = config_path.parent / item["file"]
        if not weight_path.is_file():
            raise FileNotFoundError(f"가중치 파일이 없습니다: {weight_path}")
        model = MODEL_CLASSES[item["model"]]().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        members.append((item["name"], model, float(item["weight"])))
    total = sum(weight for _, _, weight in members)
    return [(name, model, weight / total) for name, model, weight in members]


@torch.inference_mode()
def enhance(members, noisy, device, shifts):
    length = noisy.shape[-1]
    # 아주 짧은 입력도 STFT가 가능하도록 0으로 채운 뒤 마지막에 원래 길이로 자릅니다.
    min_length = max(model.n_fft for _, model, _ in members)
    wave = torch.from_numpy(np.pad(noisy, (0, max(0, min_length - length)))).unsqueeze(0).to(device)
    padded_length = wave.shape[-1]
    result = torch.zeros_like(wave)
    for _, model, weight in members:
        member_sum = torch.zeros_like(wave)
        for shift in shifts:
            shifted = torch.nn.functional.pad(wave, (shift, 0))
            member_sum += model.enhance(shifted)[:, shift:shift + padded_length]
        result += weight * member_sum / len(shifts)
    return result[0, :length].cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    shifts = (0,) if args.no_tta else TTA_SHIFTS

    if not args.input.is_dir():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {args.input}")
    wav_paths = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not wav_paths:
        raise FileNotFoundError(f"입력 폴더에 WAV 파일이 없습니다: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    members = load_ensemble(args.config, device)
    summary = ", ".join(f"{name} {weight:.2f}" for name, _, weight in members)
    print(f"장치 {device} | TTA {'끔' if args.no_tta else f'{len(shifts)}회'} | 입력 {len(wav_paths)}개")
    print(f"앙상블: {summary}")

    for index, in_path in enumerate(wav_paths, 1):
        info = sf.info(in_path)
        if info.samplerate != SAMPLE_RATE or info.channels != 1:
            raise ValueError(f"16 kHz 모노 WAV가 필요합니다: {in_path.name} "
                             f"({info.samplerate} Hz, {info.channels}채널)")
        noisy, _ = sf.read(in_path, dtype="float32")
        enhanced = enhance(members, noisy, device, shifts)

        # 입력과 같은 표본 수, 같은 저장 형식(예: PCM_16)으로 저장합니다.
        assert enhanced.shape == noisy.shape
        if info.subtype.startswith("PCM"):
            enhanced = np.clip(enhanced, -1.0, 1.0)
        sf.write(args.output / in_path.name, enhanced, SAMPLE_RATE, subtype=info.subtype)
        print(f"[{index}/{len(wav_paths)}] {in_path.name}")

    extra = ({p.name for p in args.output.iterdir() if p.suffix.lower() == ".wav"}
             - {p.name for p in wav_paths})
    if extra:
        print(f"주의: output/에 입력에 없는 WAV {len(extra)}개가 있습니다. 제출 전에 지우세요: "
              f"{sorted(extra)[:5]}")
    print(f"완료: {len(wav_paths)}개 파일을 {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()
