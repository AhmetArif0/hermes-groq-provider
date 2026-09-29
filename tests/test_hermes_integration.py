"""End-to-end checks against a real Hermes checkout (skipped when Hermes is not importable).

Each check runs in a fresh interpreter whose HERMES_HOME holds an installed copy of the plugin
(``$HERMES_HOME/plugins/groq-provider``), so Hermes discovers it the way it does after
``hermes plugins install groq-provider``. Nothing talks to Groq: the model list comes from a local
server replaying Groq's recorded ``/models`` response, and chat turns go to Hermes' own scripted
loopback LLM (``tests/fakes/fake_llm_provider.py``), which records every request body.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

hermes_cli = pytest.importorskip("hermes_cli")

from conftest import CATALOG, PLUGIN_DIR  # noqa: E402

HERMES_ROOT = Path(hermes_cli.__file__).resolve().parents[1]
API_KEY = "gsk-e2e-not-a-real-key"


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "hermes-home"
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "groq-provider", ignore=shutil.ignore_patterns("__pycache__"))
    return home


def _env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI", "ANTHROPIC", "GROQ", "OPENROUTER", "HERMES_"))}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(HERMES_ROOT), NO_COLOR="1")
    return env


def _python(home: Path, code: str):
    """Run *code* in a fresh interpreter and return the JSON it prints last."""
    proc = subprocess.run([sys.executable, "-c", code], env=_env(home), cwd=str(home.parent),
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_hermes_discovers_the_installed_plugin(home):
    (home / ".env").write_text(f"GROQ_API_KEY={API_KEY}\n", encoding="utf-8", newline="\n")
    found = _python(home, """
import json
from providers import get_provider_profile
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.config import OPTIONAL_ENV_VARS
from hermes_cli.models import list_available_providers
profile = get_provider_profile("groq")
row = PROVIDER_REGISTRY.get("groq")
picker = [p for p in list_available_providers() if p["id"] == "groq"]
print(json.dumps({"cls": type(profile).__name__, "base_url": profile.base_url, "env": list(profile.env_vars),
                  "row_url": row.inference_base_url, "row_env": list(row.api_key_env_vars),
                  "setup_lists_key": "GROQ_API_KEY" in OPTIONAL_ENV_VARS, "picker": picker}))
""")
    assert found == {"cls": "GroqProfile", "base_url": "https://api.groq.com/openai/v1", "env": ["GROQ_API_KEY"],
                     "row_url": "https://api.groq.com/openai/v1", "row_env": ["GROQ_API_KEY"],
                     "setup_lists_key": True,
                     "picker": [{"id": "groq", "label": "Groq", "aliases": [], "authenticated": True}]}


class _CatalogServer:
    """Serves Groq's recorded ``/models`` response and records the Authorization header."""

    def __init__(self):
        seen = self.seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):  # noqa: N802
                seen.append((self.path, self.headers.get("Authorization", "")))
                body = json.dumps(CATALOG).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/openai/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_the_model_list_leaves_out_models_hermes_cannot_run(home):
    server = _CatalogServer()
    try:
        models = _python(home, f"""
import json
from providers import get_provider_profile
print(json.dumps(get_provider_profile("groq").fetch_models(api_key="{API_KEY}", base_url="{server.base_url}")))
""")
    finally:
        server.close()
    assert models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b", "qwen/qwen3.8-27b"]
    assert server.seen == [("/openai/v1/models", f"Bearer {API_KEY}")]


def _fake_llm_module():
    path = HERMES_ROOT / "tests" / "fakes" / "fake_llm_provider.py"
    if not path.is_file():
        pytest.skip("this Hermes checkout has no tests/fakes/fake_llm_provider.py")
    spec = importlib.util.spec_from_file_location("hermes_fake_llm_provider", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model, effort, sent", [
    ("openai/gpt-oss-120b", None, None),
    ("openai/gpt-oss-120b", "none", "low"),
    ("openai/gpt-oss-120b", "xhigh", "high"),
    ("qwen/qwen3.8-27b", "none", "none"),
    ("qwen/qwen3.8-27b", "minimal", "low"),
    ("llama-3.3-70b-versatile", "high", None),
])
def test_a_real_turn_sends_the_reasoning_value_the_model_accepts(home, model, effort, sent):
    fake = _fake_llm_module()
    with fake.FakeLLMServer([fake.Text("pong")], api_key=API_KEY) as llm:
        (home / "config.yaml").write_text(
            "model:\n  provider: groq\n"
            f"  base_url: {llm.base_url}\n  default: {model}\n  context_length: 131072\n"
            "agent:\n  api_max_retries: 1\n" + (f"  reasoning_effort: {effort}\n" if effort else ""),
            encoding="utf-8", newline="\n")
        (home / ".env").write_text(f"GROQ_API_KEY={API_KEY}\n", encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "chat", "-q", "Reply with: pong", "--oneshot", "-Q", "-t", "todo"],
            env=_env(home), cwd=str(home.parent), capture_output=True, text=True, timeout=240, check=False)
        assert proc.returncode == 0, (proc.stdout[-1000:], proc.stderr[-2000:])
        assert "pong" in proc.stdout
        main = llm.main_requests()
    assert main, "the turn made no chat request"
    body = main[0]
    assert body["model"] == model
    assert body.get("reasoning_effort") == sent
    assert "reasoning" not in body  # the generic nested fallback never goes to Groq
