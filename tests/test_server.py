import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from omnivoice.cli.server import (
    build_parser,
    build_voice_instruct,
    create_app,
    normalize_language,
)


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [np.zeros(2400, dtype=np.float32)]


def test_build_voice_instruct_omits_auto_attributes():
    assert (
        build_voice_instruct(gender="Female", age="Young Adult", pitch="auto")
        == "female, young adult"
    )
    assert build_voice_instruct(gender="auto", age="auto", pitch="auto") is None


def test_normalize_supported_language_aliases():
    assert normalize_language("Korean") == "ko"
    assert normalize_language("ko_KR") == "ko"
    assert normalize_language("vi-VN") == "vi"
    assert normalize_language("Auto") is None


def test_health_and_speech_contract_use_injected_model():
    model = FakeModel()
    app = create_app(model=model, model_name="fake-model")

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "Xin chao",
                "language": "vi-VN",
                "gender": "Male",
                "age": "Young Adult",
                "pitch": "Auto",
                "num_step": 16,
            },
        )

    assert health.status_code == 200
    assert health.json()["modelLoaded"] is True
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert model.calls == [
        {
            "text": "Xin chao",
            "language": "vi",
            "instruct": "male, young adult",
            "num_step": 16,
            "guidance_scale": 2.0,
            "denoise": True,
            "preprocess_prompt": True,
            "postprocess_output": True,
            "normalize_text": False,
        }
    ]


def test_model_loader_runs_once_for_service_lifetime():
    model = FakeModel()
    load_count = 0

    def load_model():
        nonlocal load_count
        load_count += 1
        return model

    app = create_app(model_loader=load_model)
    with TestClient(app) as client:
        for _ in range(100):
            health = client.get("/health")
            if health.json()["modelLoaded"]:
                break
            time.sleep(0.01)
        assert health.status_code == 200
        assert health.json()["modelLoaded"] is True
        assert client.post("/v1/audio/speech", json={"text": "hello"}).status_code == 200

    assert load_count == 1


def test_health_stays_available_while_model_loads():
    model = FakeModel()
    loader_started = threading.Event()
    release_loader = threading.Event()

    def load_model():
        loader_started.set()
        release_loader.wait(timeout=2)
        return model

    app = create_app(model_loader=load_model)
    with TestClient(app) as client:
        assert loader_started.wait(timeout=1)
        loading = client.get("/health")
        speech = client.post("/v1/audio/speech", json={"text": "hello"})
        assert loading.status_code == 200
        assert loading.json()["status"] == "loading"
        assert loading.json()["modelLoaded"] is False
        assert speech.status_code == 503

        release_loader.set()
        for _ in range(100):
            ready = client.get("/health")
            if ready.json()["modelLoaded"]:
                break
            time.sleep(0.01)
        assert ready.status_code == 200
        assert ready.json()["status"] == "ok"


def test_invalid_voice_attribute_returns_422_without_inference():
    model = FakeModel()
    app = create_app(model=model)

    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"text": "hello", "pitch": "extreme"},
        )

    assert response.status_code == 422
    assert model.calls == []


def test_server_cli_accepts_host_and_port():
    args = build_parser().parse_args(["--host", "127.0.0.1", "--port", "8123"])
    assert args.host == "127.0.0.1"
    assert args.port == 8123
