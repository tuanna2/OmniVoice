#!/usr/bin/env python3
"""Stable HTTP service for OmniVoice inference.

The Gradio demo is intentionally kept as a human-facing interface.  This
module exposes a small machine-facing contract so callers do not depend on
Gradio's generated function indexes or queue protocol.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field


LOGGER = logging.getLogger(__name__)

AUTO_VALUES = frozenset({"", "auto", "automatic", "default", "none", "null"})
SUPPORTED_GENDERS = frozenset({"male", "female"})
SUPPORTED_AGES = frozenset(
    {"child", "teenager", "young adult", "middle-aged", "elderly"}
)
SUPPORTED_PITCHES = frozenset(
    {
        "very low pitch",
        "low pitch",
        "moderate pitch",
        "high pitch",
        "very high pitch",
    }
)

_LANGUAGE_ALIASES = {
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "french": "fr",
    "fr-fr": "fr",
    "japanese": "ja",
    "ja-jp": "ja",
    "korean": "ko",
    "kor": "ko",
    "ko-kr": "ko",
    "vietnamese": "vi",
    "vie": "vi",
    "vi-vn": "vi",
}


class SpeechRequest(BaseModel):
    """Request body for ``POST /v1/audio/speech``."""

    text: str = Field(min_length=1)
    language: str | None = None
    voice_key: str | None = Field(default=None, max_length=120)
    ref_audio_base64: str | None = Field(default=None, max_length=30_000_000)
    ref_text: str | None = Field(default=None, max_length=10_000)
    gender: str | None = "auto"
    age: str | None = "auto"
    pitch: str | None = "auto"
    num_step: int = Field(default=32, ge=4, le=64)
    guidance_scale: float = Field(default=2.0, ge=0.0, le=4.0)
    denoise: bool = True
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    duration: float | None = Field(default=None, gt=0.0)
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    normalize_text: bool = False


def normalize_language(value: str | None) -> str | None:
    """Normalize common language names/locales while preserving other IDs."""

    normalized = str(value or "").strip()
    if not normalized or normalized.lower() in AUTO_VALUES:
        return None

    lookup = normalized.lower().replace("_", "-")
    if lookup in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[lookup]
    return lookup


def _normalize_attribute(
    value: str | None,
    *,
    label: str,
    supported: frozenset[str],
) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", " ")
    if normalized in AUTO_VALUES:
        return None

    if label == "age":
        normalized = normalized.replace("young-adult", "young adult")
    if label == "pitch" and normalized in {
        "very low",
        "low",
        "moderate",
        "high",
        "very high",
    }:
        normalized = f"{normalized} pitch"

    if normalized not in supported:
        allowed = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported {label} '{value}'. Allowed values: auto, {allowed}")
    return normalized


def build_voice_instruct(
    *,
    gender: str | None,
    age: str | None,
    pitch: str | None,
) -> str | None:
    """Build OmniVoice's comma-separated voice-design instruction.

    ``auto`` values are deliberately omitted so the model chooses that
    attribute instead of receiving an unsupported literal ``auto`` token.
    """

    attributes = [
        _normalize_attribute(
            gender,
            label="gender",
            supported=SUPPORTED_GENDERS,
        ),
        _normalize_attribute(age, label="age", supported=SUPPORTED_AGES),
        _normalize_attribute(
            pitch,
            label="pitch",
            supported=SUPPORTED_PITCHES,
        ),
    ]
    selected = [item for item in attributes if item]
    return ", ".join(selected) if selected else None


def decode_clone_reference(encoded_audio: str) -> tuple[np.ndarray, int, str]:
    """Decode and fingerprint a source-controlled clone reference."""

    try:
        audio_bytes = base64.b64decode(encoded_audio, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("ref_audio_base64 must contain valid base64 audio") from error
    if len(audio_bytes) <= 44:
        raise ValueError("ref_audio_base64 contains empty audio")
    try:
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"Unable to decode clone reference audio: {error}") from error
    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError("Clone reference audio is empty")
    # soundfile returns (samples, channels); OmniVoice expects (channels, samples).
    channels_first = np.asarray(waveform.T, dtype=np.float32)
    return channels_first, int(sample_rate), hashlib.sha256(audio_bytes).hexdigest()


def _default_model_loader(model_name: str, device: str | None) -> Any:
    # Heavy imports stay inside the loader so unit tests can inject a fake model
    # without downloading or initializing a checkpoint.
    import torch

    from omnivoice import OmniVoice
    from omnivoice.utils.common import get_best_device

    resolved_device = device or get_best_device()
    dtype = torch.float32 if str(resolved_device).lower() == "cpu" else torch.float16
    LOGGER.info("Loading OmniVoice model=%s device=%s", model_name, resolved_device)
    return OmniVoice.from_pretrained(
        model_name,
        device_map=resolved_device,
        dtype=dtype,
        load_asr=False,
    )


def create_app(
    *,
    model: Any | None = None,
    model_loader: Callable[[], Any] | None = None,
    model_name: str = "k2-fsa/OmniVoice",
) -> FastAPI:
    """Create the service app, optionally injecting a model for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.model is None and not app.state.model_load_started:
            if model_loader is None:
                raise RuntimeError("OmniVoice model loader is not configured")

            app.state.model_load_started = True
            app.state.model_loading = True

            def load_model_in_background() -> None:
                try:
                    loaded_model = model_loader()
                    if loaded_model is None:
                        raise RuntimeError("OmniVoice model loader returned no model")
                    app.state.model = loaded_model
                except Exception as error:  # noqa: BLE001
                    app.state.model_load_error = f"{type(error).__name__}: {error}"
                    LOGGER.exception("OmniVoice model loading failed")
                finally:
                    app.state.model_loading = False

            load_thread = threading.Thread(
                target=load_model_in_background,
                name="omnivoice-model-loader",
                daemon=True,
            )
            app.state.model_load_thread = load_thread
            load_thread.start()
        yield

    app = FastAPI(title="OmniVoice HTTP Service", version="1.0.0", lifespan=lifespan)
    app.state.model = model
    app.state.model_name = model_name
    app.state.model_load_started = model is not None
    app.state.model_loading = False
    app.state.model_load_error = None
    app.state.model_load_thread = None
    app.state.inference_lock = threading.Lock()
    app.state.voice_clone_prompt_cache = {}

    @app.get("/health")
    def health() -> Any:
        loaded_model = app.state.model
        load_error = app.state.model_load_error
        payload = {
            "status": "ok" if loaded_model is not None else ("error" if load_error else "loading"),
            "modelLoaded": loaded_model is not None,
            "model": app.state.model_name,
        }
        if load_error:
            payload["error"] = load_error
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.post("/v1/audio/speech", response_class=Response)
    def synthesize_speech(request: SpeechRequest) -> Response:
        normalized_text = request.text.strip()
        if not normalized_text:
            raise HTTPException(status_code=422, detail="text must not be blank")

        try:
            instruct = build_voice_instruct(
                gender=request.gender,
                age=request.age,
                pitch=request.pitch,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        loaded_model = app.state.model
        if loaded_model is None:
            if app.state.model_load_error:
                raise HTTPException(
                    status_code=503,
                    detail=f"OmniVoice model failed to load: {app.state.model_load_error}",
                )
            raise HTTPException(status_code=503, detail="OmniVoice model is not loaded")

        clone_fields = [request.voice_key, request.ref_audio_base64, request.ref_text]
        has_clone_reference = any(value is not None and str(value).strip() for value in clone_fields)
        clone_audio: tuple[np.ndarray, int] | None = None
        clone_cache_key: tuple[str, str, str, bool] | None = None
        if has_clone_reference:
            voice_key = str(request.voice_key or "").strip()
            ref_text = str(request.ref_text or "").strip()
            encoded_audio = str(request.ref_audio_base64 or "").strip()
            if not voice_key or not ref_text or not encoded_audio:
                raise HTTPException(
                    status_code=422,
                    detail="voice_key, ref_audio_base64, and ref_text are all required for voice cloning",
                )
            try:
                waveform, sample_rate, audio_digest = decode_clone_reference(encoded_audio)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            clone_audio = (waveform, sample_rate)
            clone_cache_key = (
                voice_key.casefold(),
                audio_digest,
                hashlib.sha256(ref_text.encode("utf-8")).hexdigest(),
                bool(request.preprocess_prompt),
            )

        generation_options: dict[str, Any] = {
            "text": normalized_text,
            "language": normalize_language(request.language),
            "instruct": instruct,
            "num_step": request.num_step,
            "guidance_scale": request.guidance_scale,
            "denoise": request.denoise,
            "preprocess_prompt": request.preprocess_prompt,
            "postprocess_output": request.postprocess_output,
            "normalize_text": request.normalize_text,
        }
        if request.speed is not None:
            generation_options["speed"] = request.speed
        if request.duration is not None:
            generation_options["duration"] = request.duration

        try:
            with app.state.inference_lock:
                if clone_cache_key is not None and clone_audio is not None:
                    clone_prompt = app.state.voice_clone_prompt_cache.get(clone_cache_key)
                    if clone_prompt is None:
                        clone_prompt = loaded_model.create_voice_clone_prompt(
                            ref_audio=clone_audio,
                            ref_text=str(request.ref_text).strip(),
                            preprocess_prompt=request.preprocess_prompt,
                        )
                        app.state.voice_clone_prompt_cache[clone_cache_key] = clone_prompt
                    generation_options["voice_clone_prompt"] = clone_prompt
                generated = loaded_model.generate(**generation_options)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("OmniVoice generation failed")
            raise HTTPException(
                status_code=500,
                detail=f"OmniVoice generation failed: {type(error).__name__}: {error}",
            ) from error

        if not isinstance(generated, (list, tuple)) or not generated:
            raise HTTPException(status_code=500, detail="OmniVoice returned no audio")

        audio = np.asarray(generated[0], dtype=np.float32).squeeze()
        if audio.ndim != 1 or audio.size == 0:
            raise HTTPException(status_code=500, detail="OmniVoice returned invalid audio")

        sample_rate = int(getattr(loaded_model, "sampling_rate", 24000) or 24000)
        output = io.BytesIO()
        sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = output.getvalue()
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"',
                "X-OmniVoice-Sample-Rate": str(sample_rate),
                "X-OmniVoice-Voice-Mode": "clone" if has_clone_reference else "design",
            },
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve OmniVoice over HTTP")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    import uvicorn

    app = create_app(
        model_loader=lambda: _default_model_loader(args.model, args.device),
        model_name=args.model,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
