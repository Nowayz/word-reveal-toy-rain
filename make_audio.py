from pathlib import Path
import json
import shutil
import subprocess
import sys

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


ASSETS = Path("assets")
AUDIO_DIR = ASSETS / "audio"
MANIFEST = ASSETS / "manifest.json"
QWEN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
PREFERRED_SPEAKERS = ("vivian", "sohee", "serena", "ono_anna")
INSTRUCTION = "Speak clearly and slowly, sounding each word out carefully, as if teaching it to a child."
SENTENCE_TEMPLATE = "It's a {item}"


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_female_speaker(tts: Qwen3TTSModel) -> str:
    supported = tts.model.get_supported_speakers() or []
    if not supported:
        raise RuntimeError("Qwen-TTS did not return a supported speaker list.")

    by_lower = {s.lower(): s for s in supported}
    for preferred in PREFERRED_SPEAKERS:
        if preferred.lower() in by_lower:
            return by_lower[preferred.lower()]

    return supported[0]


def parse_args():
    force = False
    speaker = None
    targets = set()
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--force":
            force = True
            i += 1
        elif args[i] == "--speaker":
            if i + 1 >= len(args):
                raise RuntimeError("--speaker requires a value.")
            speaker = args[i + 1]
            i += 2
        else:
            targets.add(args[i])
            i += 1
    return force, speaker, targets


def encode_opus(wav_path, opus_path):
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-c:a",
        "libopus",
        "-b:a",
        "48k",
        "-vbr",
        "on",
        str(opus_path),
    ]
    subprocess.run(command, check=True)


def build_sentence(word: str) -> str:
    return SENTENCE_TEMPLATE.format(item=word)


def main():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to encode Opus files.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    force, forced_speaker, targets = parse_args()
    device = device_name()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"device_map": device, "dtype": dtype}

    print(f"Loading Qwen-TTS ({QWEN_MODEL}) on {device}...")
    tts = Qwen3TTSModel.from_pretrained(QWEN_MODEL, **model_kwargs)
    if forced_speaker:
        supported = tts.model.get_supported_speakers() or []
        by_lower = {s.lower(): s for s in supported}
        if forced_speaker.lower() in by_lower:
            speaker = by_lower[forced_speaker.lower()]
        else:
            raise RuntimeError(
                f"Speaker '{forced_speaker}' is not available. Supported: {sorted(supported)}"
            )
    else:
        speaker = pick_female_speaker(tts)
    print(f"Using speaker: {speaker}")

    for i, item in enumerate(manifest, start=1):
        slug = Path(item["image"]).stem
        wav_path = AUDIO_DIR / f"{slug}.wav"
        opus_path = AUDIO_DIR / f"{slug}.opus"
        selected = not targets or slug in targets or item["word"] in targets

        if selected and (force or not opus_path.exists()):
            text = build_sentence(item["word"])
            print(f"[{i}/{len(manifest)}] {text}")
            wavs, sample_rate = tts.generate_custom_voice(
                text=text,
                speaker=speaker,
                language="English",
                instruct=INSTRUCTION,
                temperature=0.65,
            )
            sf.write(str(wav_path), wavs[0], sample_rate)
            encode_opus(wav_path, opus_path)
            wav_path.unlink(missing_ok=True)

        item["audio"] = str(opus_path).replace("\\", "/")

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated/linked {len(manifest)} Opus files in {AUDIO_DIR}.")


if __name__ == "__main__":
    main()
