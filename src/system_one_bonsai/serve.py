"""Local HTTP server that keeps the model loaded and answers requests in the System One API format.

    python -m system_one_bonsai.serve [--backend mlx|torch] [--host 127.0.0.1] [--port 8765]

    GET  /health        -> {"ok": true, "backend": ...}
    POST /v1/systemone  {"model"?: str, "state": object, "questions": {id: question}}
                        -> {"model": str, "answers": {id: answer}, "usage": {"input_tokens", "output_tokens"}}

The request and response shapes follow what System One API clients send and accept (for example
`models.classify()` in the pi coding agent). They were derived from a client implementation, not from a
published specification; compatibility is not guaranteed. No authentication: the server ignores any
`Authorization` header and binds to localhost by default. Requests are served one at a time.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .packing import Branch, PackedRequest, pack_request

MODEL_ID = "bonsai-4b-system-one"


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def to_request(body: dict) -> tuple[PackedRequest, list[tuple[str, str, list[str]]]]:
    """Translate a System One request into a packed request.

    Returns the request and, per question, (id, type, labels) for building the answers.
    The state object is rendered as indented JSON. Choice options read `label: description`.
    Score levels are the criteria strings, lowest first. A noul question's true/false criteria
    are appended to its instructions; its options stay [No, Yes].
    """
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    state = body.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("state must be a non-empty JSON object")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    branches, meta = [], []
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"question {qid} must be an object")
        kind, criteria = question.get("type"), question.get("criteria")
        instructions = _text(question.get("instructions"), f"{qid}.instructions")
        if kind == "choice":
            if not isinstance(criteria, dict):
                raise ValueError(f"{qid}.criteria must map labels to descriptions")
            labels = [str(label) for label in criteria]
            options = [f"{label}: {desc}" if desc else label for label, desc in zip(labels, criteria.values())]
            branches.append(Branch("choice", instructions, options))
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError(f"{qid}.criteria must list the levels, lowest first")
            labels = [_text(level, f"{qid}.criteria[]") for level in criteria]
            branches.append(Branch("score", instructions, labels))
        elif kind in ("noul", "bool"):
            if isinstance(criteria, dict) and (criteria.get("true") or criteria.get("false")):
                instructions = f"{instructions} (Yes: {criteria.get('true', '')}; No: {criteria.get('false', '')})"
            labels = ["No", "Yes"]
            branches.append(Branch("noul", instructions))
        else:
            raise ValueError(f"{qid}.type must be choice, score, or noul")
        meta.append((qid, branches[-1].kind, labels))
    state_text = json.dumps(state, indent=2, ensure_ascii=False)
    return PackedRequest(state_text, branches), meta


def to_answers(meta: list[tuple[str, str, list[str]]], probabilities: list[list[float]]) -> dict:
    """Build System One answers. `confidence` is the highest option probability (this server's choice)."""
    answers = {}
    for (qid, kind, labels), p in zip(meta, probabilities):
        if kind == "choice":
            best = max(range(len(p)), key=p.__getitem__)
            answers[qid] = {"type": "choice", "choice": labels[best],
                            "probabilities": dict(zip(labels, p)), "confidence": p[best]}
        elif kind == "score":
            answers[qid] = {"type": "score", "score": sum(i * q for i, q in enumerate(p)), "confidence": max(p)}
        else:
            answers[qid] = {"type": "noul", "noul": p[1]}
    return answers


def make_handler(backend: str, model, tokenizer, predict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "backend": backend})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in ("/v1/systemone", "/systemone"):
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
                request, meta = to_request(body)
                input_tokens = len(pack_request(request, tokenizer)[0])
            except (ValueError, TypeError) as error:
                self._send(400, {"error": str(error)})
                return
            probabilities = predict(model, tokenizer, request)
            self._send(200, {"model": body.get("model") or MODEL_ID, "answers": to_answers(meta, probabilities),
                             "usage": {"input_tokens": input_tokens, "output_tokens": 0}})

        def log_message(self, format: str, *args) -> None:  # one short line per request, no payloads
            print(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}", flush=True)

    return Handler


def main() -> None:
    from .runner import load_backend

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["torch", "mlx"], default="mlx")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    model, tokenizer, predict = load_backend(args.backend)
    server = HTTPServer((args.host, args.port), make_handler(args.backend, model, tokenizer, predict))
    print(f"serving {MODEL_ID} ({args.backend}) on http://{args.host}:{args.port}/v1/systemone", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
