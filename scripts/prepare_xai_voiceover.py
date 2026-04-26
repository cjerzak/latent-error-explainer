#!/usr/bin/env python3
"""Create xAI TTS narration and mux it into the Manim renders."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN = ROOT / "latent_predictor_visualization_slide_script.md"
DEFAULT_MEDIA_DIR = ROOT / "media/videos/latent_predictor_scaling_manim"
TTS_URL = "https://api.x.ai/v1/tts"
TRANSITION_SECONDS = 2.0
CONTROL_TOKEN_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9_-]*\]")


@dataclass(frozen=True)
class RenderSpec:
    quality_dir: str
    fps: int
    width: int
    height: int


@dataclass(frozen=True)
class VideoSegment:
    name: str
    clip_start: int
    clip_end: int
    audio_index: int | None
    target_seconds: float | None = None


RENDERS = [
    RenderSpec("720p30", fps=30, width=1280, height=720),
    RenderSpec("480p15", fps=15, width=854, height=480),
]

VIDEO_SEGMENTS = [
    VideoSegment("slide_01", clip_start=1, clip_end=6, audio_index=0),
    VideoSegment("slide_02", clip_start=7, clip_end=14, audio_index=1),
    VideoSegment(
        "transition_02_to_03",
        clip_start=15,
        clip_end=16,
        audio_index=None,
        target_seconds=TRANSITION_SECONDS,
    ),
    VideoSegment("slide_03", clip_start=17, clip_end=23, audio_index=2),
    VideoSegment("slide_04", clip_start=24, clip_end=31, audio_index=3),
    VideoSegment("slide_05", clip_start=32, clip_end=35, audio_index=4),
]

CONCISE_SECTIONS = [
    (
        "Many political science quantities are latent. We do not observe X directly. "
        "We estimate it as tilde X, equal to X plus error. Then identification puts "
        "both the true trait and estimate on "
        "mean zero, unit variance scales."
    ),
    (
        "The top row is X with variance one. Add error and tilde X spreads out; its "
        "standard deviation grows with sigma U squared. But the regression score is "
        "X hat, standardized again. Noise widens the estimate; standardization "
        "compresses it back."
    ),
    (
        "Rescaling changes attenuation. Classical error multiplies the slope by one "
        "over one plus sigma U squared. Identified latent scores remove the variance "
        "inflation, leaving the square-root factor."
    ),
    (
        "Split indicators estimate the correction. Build two scores from disjoint "
        "indicators. If their errors are independent, their correlation estimates "
        "reliability, rho. The unadjusted slope converges to the square root of rho "
        "times the true slope, so divide by the square root of rho."
    ),
    (
        "The lesson is simple: latent predictors are noisy variables on an identified "
        "scale. Estimate rho from split scores, then correct the regression slope by "
        "dividing the unadjusted estimate by the square root of rho."
    ),
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    assignment = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = assignment.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        if name in os.environ:
            continue
        value = raw_value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if value:
            os.environ[name] = value


def load_api_key() -> str:
    for path in [ROOT / ".env", Path.home() / ".Renviron", Path.home() / ".zshrc"]:
        load_env_file(path)

    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        raise SystemExit("Missing XAI_API_KEY. Add it to .env, ~/.Renviron, or ~/.zshrc.")
    return key


def run(cmd: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def voiceover_sections(markdown_path: Path, mode: str, include_closing: bool) -> list[str]:
    if mode == "concise":
        return [normalize_for_tts(section) for section in CONCISE_SECTIONS]

    return parse_markdown_voiceover_sections(markdown_path, include_closing)


def parse_markdown_voiceover_sections(markdown_path: Path, include_closing: bool) -> list[str]:
    text = markdown_path.read_text()
    sections = re.findall(
        r"\*\*Voiceover:\*\*\n(.+?)(?=\n\n\*\*Transition:|\n\n## Optional)",
        text,
        flags=re.S,
    )
    if len(sections) != 5:
        raise SystemExit(f"Expected 5 voiceover sections, found {len(sections)} in {markdown_path}.")

    cleaned = [normalize_for_tts(section) for section in sections]
    if include_closing:
        closing = re.search(r'## Optional short closing line\s*\n\s*"(.+?)"', text, flags=re.S)
        if closing:
            cleaned[-1] = f"{cleaned[-1]} {normalize_for_tts(closing.group(1))}"
    return cleaned


def normalize_for_tts(text: str) -> str:
    text = " ".join(text.split())
    replacements = {
        "tilde X": "tilde X",
        "hat X": "X hat",
        "hat X_1": "X hat one",
        "hat X_2": "X hat two",
        "sigma_U squared": "sigma U squared",
        "sigma_U^2": "sigma U squared",
        "sqrt one plus": "the square root of one plus",
        "sqrt rho": "the square root of rho",
        "beta_X": "beta X",
        "beta_star": "beta star",
        "mean-zero": "mean zero",
        "unit-variance": "unit variance",
        "split-specific": "split specific",
        "split-based": "split based",
        "latent-scale": "latent scale",
        "errors-in-variables": "errors in variables",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    text = re.sub(r"\bsqrt\s*\(\s*rho\s*\)", "the square root of rho", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def synthesize_tts(
    text: str,
    output_path: Path,
    *,
    api_key: str,
    voice_id: str,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"Using cached TTS: {output_path}")
        return

    control_tokens = sorted(set(CONTROL_TOKEN_RE.findall(text)))
    if control_tokens:
        tokens = ", ".join(control_tokens)
        raise ValueError(f"Refusing to send control token(s) to TTS: {tokens}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": text,
        "voice_id": voice_id,
        "language": "en",
        "output_format": {"codec": "mp3", "sample_rate": 44100, "bit_rate": 192000},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg, application/octet-stream",
    }

    for attempt in range(1, 6):
        response = requests.post(TTS_URL, headers=headers, json=payload, timeout=120)
        content_type = response.headers.get("content-type", "")
        if response.ok and not content_type.startswith("application/json"):
            output_path.write_bytes(response.content)
            print(f"Wrote TTS: {output_path}")
            return

        retryable = response.status_code in {429, 500, 502, 503, 504}
        if retryable and attempt < 5:
            wait_seconds = min(30, 2**attempt)
            print(f"TTS request returned {response.status_code}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
            continue

        detail = response.text[:500].replace(api_key, "<redacted>")
        raise RuntimeError(f"xAI TTS request failed with {response.status_code}: {detail}")


def local_concat_list(source_list: Path, output_list: Path) -> list[Path]:
    base = source_list.parent
    clips: list[Path] = []
    for line in source_list.read_text().splitlines():
        if not line.startswith("file "):
            continue
        original = line.split(" ", 1)[1].strip().strip("'\"")
        clips.append(base / Path(original).name)

    if not clips:
        raise SystemExit(f"No clips found in {source_list}")

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("".join(f"file '{clip}'\n" for clip in clips))
    return clips


def write_concat_list(paths: Iterable[Path], output_list: Path) -> None:
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("".join(f"file '{path}'\n" for path in paths))


def concat_video(clips: list[Path], output_path: Path, scratch_dir: Path) -> None:
    concat_file = scratch_dir / f"{output_path.stem}_concat.txt"
    write_concat_list(clips, concat_file)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
    )


def concat_audio(audio_paths: list[Path], output_path: Path, scratch_dir: Path) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    inputs = [["-i", str(path)] for path in audio_paths]
    input_args = [arg for pair in inputs for arg in pair]
    filter_inputs = "".join(f"[{index}:a]" for index in range(len(audio_paths)))
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            *input_args,
            "-filter_complex",
            f"{filter_inputs}concat=n={len(audio_paths)}:v=0:a=1[a]",
            "-map",
            "[a]",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ]
    )


def create_silence_audio(output_path: Path, seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            f"{seconds:.3f}",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def segment_audio_sequence(audio_paths: list[Path], silence_audio_path: Path) -> list[Path]:
    sequence: list[Path] = []
    for segment in VIDEO_SEGMENTS:
        if segment.audio_index is None:
            sequence.append(silence_audio_path)
        else:
            sequence.append(audio_paths[segment.audio_index])
    return sequence


def voiceover_dir(media_dir: Path, voice_id: str, script_mode: str) -> Path:
    mode_dir = "fullscript" if script_mode == "full" else script_mode
    return media_dir / f"xai_voiceover_{voice_id}_{mode_dir}"


def split_video_segments(clips: list[Path]) -> list[tuple[VideoSegment, list[Path]]]:
    expected_clip_count = max(segment.clip_end for segment in VIDEO_SEGMENTS)
    if expected_clip_count != len(clips):
        raise SystemExit(
            f"Expected {expected_clip_count} clips but found {len(clips)}; segment map needs updating."
        )
    return [
        (segment, clips[segment.clip_start - 1 : segment.clip_end])
        for segment in VIDEO_SEGMENTS
    ]


def retime_video_to_duration(
    video_path: Path,
    output_path: Path,
    *,
    target_duration: float,
    fps: int,
    width: int,
    height: int,
) -> None:
    video_duration = probe_duration(video_path)
    ratio = target_duration / video_duration
    scale_filter = (
        f"setpts={ratio:.9f}*PTS,"
        f"scale={width}:{height}:flags=lanczos,"
        f"fps={fps},format=yuv420p"
    )
    print(
        f"Retiming {video_path.name}: video={video_duration:.2f}s target={target_duration:.2f}s ratio={ratio:.3f}"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def retime_video_to_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    audio_duration = probe_duration(audio_path)
    retime_video_to_duration(
        video_path,
        output_path,
        target_duration=audio_duration,
        fps=fps,
        width=width,
        height=height,
    )


def mux_upload_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    )


def process_render(
    render: RenderSpec,
    media_dir: Path,
    voice_dir: Path,
    audio_paths: list[Path],
    full_audio_path: Path,
    voice_id: str,
    output_suffix: str,
) -> Path:
    partial_dir = (
        media_dir
        / render.quality_dir
        / "partial_movie_files"
        / "LatentPredictorScalingDynamics"
    )
    source_list = partial_dir / "partial_movie_file_list.txt"
    scratch_dir = voice_dir / "scratch" / render.quality_dir
    scratch_dir.mkdir(parents=True, exist_ok=True)
    clips = local_concat_list(source_list, scratch_dir / "local_partial_movie_file_list.txt")

    base_video = media_dir / render.quality_dir / "LatentPredictorScalingDynamics_reconstructed.mp4"
    concat_video(clips, base_video, scratch_dir)

    section_videos: list[Path] = []
    for index, (segment, section_clips) in enumerate(split_video_segments(clips), start=1):
        section_base = scratch_dir / f"segment_{index:02d}_{segment.name}_base.mp4"
        section_timed = scratch_dir / f"segment_{index:02d}_{segment.name}_timed.mp4"
        concat_video(section_clips, section_base, scratch_dir)
        if segment.audio_index is None:
            if segment.target_seconds is None:
                raise SystemExit(f"Segment {segment.name} needs target_seconds.")
            retime_video_to_duration(
                section_base,
                section_timed,
                target_duration=segment.target_seconds,
                fps=render.fps,
                width=render.width,
                height=render.height,
            )
        else:
            retime_video_to_audio(
                section_base,
                audio_paths[segment.audio_index],
                section_timed,
                fps=render.fps,
                width=render.width,
                height=render.height,
            )
        section_videos.append(section_timed)

    timed_silent = media_dir / render.quality_dir / "LatentPredictorScalingDynamics_timed_silent.mp4"
    concat_video(section_videos, timed_silent, scratch_dir)

    upload_path = (
        media_dir
        / render.quality_dir
        / f"LatentPredictorScalingDynamics_YouTube_xai_{voice_id}{output_suffix}.mp4"
    )
    mux_upload_video(timed_silent, full_audio_path, upload_path)
    return upload_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--voice-id", default="ara")
    parser.add_argument("--script-mode", choices=["concise", "full"], default="concise")
    parser.add_argument("--force-tts", action="store_true")
    parser.add_argument(
        "--force-tts-section",
        type=int,
        action="append",
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Regenerate one 1-based voiceover section; may be passed more than once.",
    )
    parser.add_argument("--include-closing", dest="include_closing", action="store_true", default=None)
    parser.add_argument("--no-include-closing", dest="include_closing", action="store_false")
    args = parser.parse_args()

    api_key = load_api_key()
    include_closing = args.include_closing
    if include_closing is None:
        include_closing = args.script_mode == "full"
    sections = voiceover_sections(args.markdown, args.script_mode, include_closing=include_closing)
    voice_dir = voiceover_dir(args.media_dir, args.voice_id, args.script_mode)
    voice_dir.mkdir(parents=True, exist_ok=True)

    (voice_dir / "spoken_script.json").write_text(
        json.dumps(
            [
                {"section": index, "mode": args.script_mode, "text": text}
                for index, text in enumerate(sections, start=1)
            ],
            indent=2,
        )
    )

    audio_paths: list[Path] = []
    force_tts_sections = set(args.force_tts_section or [])
    for index, text in enumerate(sections, start=1):
        audio_path = voice_dir / f"slide_{index:02d}_{args.voice_id}.mp3"
        synthesize_tts(
            text,
            audio_path,
            api_key=api_key,
            voice_id=args.voice_id,
            force=args.force_tts or index in force_tts_sections,
        )
        audio_paths.append(audio_path)

    silence_audio_path = voice_dir / f"silence_{TRANSITION_SECONDS:.1f}s.wav"
    create_silence_audio(silence_audio_path, TRANSITION_SECONDS)
    full_audio_sequence = segment_audio_sequence(audio_paths, silence_audio_path)

    full_audio_path = voice_dir / f"LatentPredictorScalingDynamics_voiceover_{args.voice_id}.mp3"
    concat_audio(full_audio_sequence, full_audio_path, voice_dir / "scratch")

    output_suffix = "" if args.script_mode == "concise" else "_fullscript"
    upload_paths = [
        process_render(
            render,
            args.media_dir,
            voice_dir,
            audio_paths,
            full_audio_path,
            args.voice_id,
            output_suffix,
        )
        for render in RENDERS
    ]

    print("\nCreated upload files:")
    for path in upload_paths:
        print(f"  {path} ({probe_duration(path):.2f}s)")
    print(f"Voiceover audio: {full_audio_path} ({probe_duration(full_audio_path):.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
