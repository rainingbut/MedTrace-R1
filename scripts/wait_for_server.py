"""Wait for the pinned vLLM OpenAI-compatible server to become ready."""

from __future__ import annotations

import argparse
import json
import time
from urllib import request as urllib_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def get_json(url: str) -> dict[str, object]:
    with urllib_request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    server_root = base_url[:-3] if base_url.endswith("/v1") else base_url
    deadline = time.monotonic() + args.timeout
    last_error = "server has not responded"

    while time.monotonic() < deadline:
        try:
            models = get_json(f"{base_url}/models")
            model_ids = {str(model["id"]) for model in models.get("data", [])}
            if args.model not in model_ids:
                raise RuntimeError(
                    f"expected model {args.model!r}; server exposes {sorted(model_ids)}"
                )
            version = get_json(f"{server_root}/version")
            print(json.dumps({"models": models, "version": version}, indent=2))
            return
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"Waiting for server: {last_error}", flush=True)
            time.sleep(args.interval)

    raise TimeoutError(f"server was not ready within {args.timeout}s: {last_error}")


if __name__ == "__main__":
    main()
