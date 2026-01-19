import argparse
from pathlib import Path

import torch
import torchaudio
from torch.nn import functional as F

CLIP_MIN = -1.0
CLIP_MAX = 1.0


def safe_clip(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.clamp(min=CLIP_MIN, max=CLIP_MAX)


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform, sample_rate


def pad_to_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.shape[-1] == length:
        return waveform
    if waveform.shape[-1] > length:
        return waveform[..., :length]
    pad_amount = length - waveform.shape[-1]
    return F.pad(waveform, (0, pad_amount))


def sum_stems(stems: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(stem.shape[-1] for stem in stems)
    padded = [pad_to_length(stem, max_length) for stem in stems]
    stacked = torch.stack(padded, dim=0)
    return stacked.sum(dim=0)


def check_sample_rates(sample_rates: list[int], folder: Path) -> int:
    if len(set(sample_rates)) != 1:
        raise ValueError(f"Sample rate mismatch in {folder}: {sample_rates}")
    return sample_rates[0]


def process_track(folder: Path) -> bool:
    vocals_path = folder / "vocals.wav"
    melody_path = folder / "melody.wav"
    instruments_path = folder / "instruments.wav"
    bass_path = folder / "bass.wav"
    drums_path = folder / "drums.wav"
    full_path = folder / "full.wav"

    if not bass_path.exists() or not drums_path.exists() or not instruments_path.exists():
        return False

    is_vocals = vocals_path.exists()
    is_instrumental = melody_path.exists()

    if not (is_vocals or is_instrumental):
        return False

    if full_path.exists():
        full_path.unlink()

    stems = []
    sample_rates = []

    bass, sr = load_audio(bass_path)
    stems.append(bass)
    sample_rates.append(sr)

    drums, sr = load_audio(drums_path)
    stems.append(drums)
    sample_rates.append(sr)

    if is_vocals:
        if instruments_path.exists():
            other_path = folder / "other.wav"
            if other_path.exists():
                other, sr = load_audio(other_path)
            else:
                instruments, sr = load_audio(instruments_path)
                instruments_path.rename(other_path)
                other = instruments
            stems.append(other)
            sample_rates.append(sr)

        vocals, sr = load_audio(vocals_path)
        stems.append(vocals)
        sample_rates.append(sr)

    if is_instrumental:
        instruments, sr = load_audio(instruments_path)
        sample_rates.append(sr)

        melody, sr = load_audio(melody_path)
        sample_rates.append(sr)

        other = sum_stems([instruments, melody])
        other = safe_clip(other)
        other_path = folder / "other.wav"
        torchaudio.save(str(other_path), other, sr)
        stems.append(other)

        if not vocals_path.exists():
            vocals = torch.zeros_like(other)
            torchaudio.save(str(vocals_path), vocals, sr)
            stems.append(vocals)
            sample_rates.append(sr)

    sample_rate = check_sample_rates(sample_rates, folder)
    mixture = sum_stems(stems)
    mixture = safe_clip(mixture)
    torchaudio.save(str(folder / "mixture.wav"), mixture, sample_rate)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stems for Mamba-S-Net.")
    parser.add_argument("root", type=Path, help="Root directory containing UUID folders.")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    processed = 0
    skipped = 0
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if process_track(entry):
            processed += 1
        else:
            skipped += 1

    print(f"Processed: {processed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
