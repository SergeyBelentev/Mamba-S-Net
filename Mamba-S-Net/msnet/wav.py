from collections import OrderedDict
import hashlib
import math
import json
import os
from pathlib import Path
import tqdm

import julius
import torch as th
from torch.nn import functional as F
from torchcodec.decoders import AudioDecoder

from msnet.utils import convert_audio_channels
from accelerate import Accelerator

accelerator = Accelerator()

MIXTURE = "mixture"
EXT = ".wav"


def _decode_num_samples_and_sr(file: Path) -> tuple[int, int]:
    """
    TorchCodec не даёт "num_frames" как torchaudio.info, но даёт:
      - metadata.sample_rate
      - metadata.duration_seconds_from_header (может быть None/неточной)
    Для WAV обычно достаточно header, но для точности делаем fallback на decode.
    """
    dec = AudioDecoder(str(file))
    md = dec.metadata  # AudioStreamMetadata :contentReference[oaicite:2]{index=2}

    sr = md.sample_rate
    dur = md.duration_seconds_from_header

    if sr is not None and dur is not None:
        # duration в секундах -> сэмплы
        n = int(round(dur * sr))
        if n > 0:
            return n, sr

    # fallback: декодируем всё и берём точную длину
    samples = dec.get_all_samples()  # AudioSamples :contentReference[oaicite:3]{index=3}
    data = samples.data
    return int(data.shape[-1]), int(samples.sample_rate)


def _load_range(file: Path, start_sample: int, num_samples: int, sr: int) -> th.Tensor:
    """
    TorchCodec API для аудио — по времени (секунды), не по sample index. :contentReference[oaicite:4]{index=4}
    Поэтому конвертим sample->seconds. На выходе: (C, T).
    """
    dec = AudioDecoder(str(file))
    if num_samples < 0:
        out = dec.get_all_samples().data
    else:
        start_s = start_sample / sr
        stop_s = (start_sample + num_samples) / sr
        out = dec.get_samples_played_in_range(start_s, stop_s).data
    return out.to(th.float32)


def _track_metadata(track, sources, normalize=True, ext=EXT):
    track_length = None
    track_samplerate = None
    mean = 0.0
    std = 1.0

    for source in sources + [MIXTURE]:
        file = Path(track) / f"{source}{ext}"
        try:
            length, sr = _decode_num_samples_and_sr(file)
        except RuntimeError:
            print(file)
            raise

        if track_length is None:
            track_length = length
            track_samplerate = sr
        else:
            if track_length != length:
                raise ValueError(
                    f"Invalid length for file {file}: expecting {track_length} but got {length}."
                )
            if sr != track_samplerate:
                raise ValueError(
                    f"Invalid sample rate for file {file}: expecting {track_samplerate} but got {sr}."
                )

        if source == MIXTURE and normalize:
            try:
                wav = _load_range(file, start_sample=0, num_samples=-1, sr=track_samplerate)
            except RuntimeError:
                print(file)
                raise
            wav = wav.mean(0)  # mono
            mean = wav.mean().item()
            std = wav.std().item()

    return {"length": track_length, "mean": mean, "std": std, "samplerate": track_samplerate}


def build_metadata(path, sources, normalize=True, ext=EXT):
    meta = {}
    path = Path(path)
    pendings = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith('.') or folders or root == path:
                continue
            name = str(root.relative_to(path))
            pendings.append((name, pool.submit(_track_metadata, root, sources, normalize, ext)))
        for name, pending in tqdm.tqdm(pendings, ncols=120):
            meta[name] = pending.result()
    return meta


class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT):

        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue

            meta = self.metadata[name]
            offset = 0
            num_frames = -1
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))

            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav = _load_range(file, start_sample=offset, num_samples=num_frames, sr=meta['samplerate'])
                wav = convert_audio_channels(wav, self.channels)
                wavs.append(wav)

            example = th.stack(wavs)  # (S, C, T)
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)

            if self.normalize:
                example = (example - meta['mean']) / meta['std']

            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))

            return example


def get_wav_datasets(args):
    sig = hashlib.sha1(str(args.wav).encode()).hexdigest()[:8]
    metadata_file = Path(args.metadata) / ('wav_' + sig + ".json")
    train_path = Path(args.wav) / "train"
    valid_path = Path(args.wav) / "test"

    if not metadata_file.is_file() and accelerator.is_main_process:
        metadata_file.parent.mkdir(exist_ok=True, parents=True)
        train = build_metadata(train_path, args.sources)
        valid = build_metadata(valid_path, args.sources)
        json.dump([train, valid], open(metadata_file, "w"))

    accelerator.wait_for_everyone()

    train, valid = json.load(open(metadata_file))
    kw_cv = {}

    train_set = Wavset(
        train_path, train, args.sources,
        segment=args.segment, shift=args.shift,
        samplerate=args.samplerate, channels=args.channels,
        normalize=args.normalize
    )
    valid_set = Wavset(
        valid_path, valid, [MIXTURE] + list(args.sources),
        samplerate=args.samplerate, channels=args.channels,
        normalize=args.normalize, **kw_cv
    )
    return train_set, valid_set
