"""Tiny OpenAI-compatible Responses server backed by the fake provider."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from gpt_dog_eval.inference import FakeInferenceProvider, InferenceRequest


class _FakeResponsesServer(ThreadingHTTPServer):
    provider: FakeInferenceProvider
    api_key: str
    provider_lock: threading.Lock

    @property
    def host_port(self) -> tuple[str, int]:
        """Return the IPv4 address selected by the server socket."""
        return cast(tuple[str, int], self.server_address)


class FakeResponsesHandler(BaseHTTPRequestHandler):
    """Serve the subset of ``POST /responses`` used by this project."""

    server: _FakeResponsesServer

    def do_POST(self) -> None:
        if self.path not in {"/responses", "/v1/responses"}:
            self._json_error(HTTPStatus.NOT_FOUND, "not found")
            return
        if self.headers.get("Authorization") != f"Bearer {self.server.api_key}":
            self._json_error(HTTPStatus.UNAUTHORIZED, "invalid API key")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4 * 1024 * 1024:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            with self.server.provider_lock:
                result = self.server.provider.complete(
                    InferenceRequest(
                        model=str(payload.get("model", "fake-go1")),
                        instructions=str(payload.get("instructions", "")),
                        input_items=tuple(payload.get("input") or ()),
                        tools=tuple(payload.get("tools") or ()),
                        reasoning_effort=(payload.get("reasoning") or {}).get("effort"),
                    )
                )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "id": f"fake_response_{self.server.provider.calls}",
                "object": "response",
                "status": "completed",
                "model": payload.get("model", "fake-go1"),
                "output": list(result.output_items),
                "usage": dict(result.usage),
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _json_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": {"message": message}})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def make_fake_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    api_key: str = "fake-local-key",
    chunk_steps: int = 10,
) -> _FakeResponsesServer:
    """Construct a fake server without starting its serving loop."""
    server = _FakeResponsesServer((host, port), FakeResponsesHandler)
    server.api_key = api_key
    server.provider = FakeInferenceProvider(chunk_steps=chunk_steps)
    server.provider_lock = threading.Lock()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--api-key", default="fake-local-key")
    parser.add_argument("--chunk-steps", type=int, default=10)
    args = parser.parse_args()
    server = make_fake_server(
        args.host,
        args.port,
        api_key=args.api_key,
        chunk_steps=args.chunk_steps,
    )
    host, port = server.host_port
    print(f"fake Responses endpoint: http://{host}:{port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
