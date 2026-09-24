"""Thin client for TypeSafe's System One endpoint (pooled keep-alive)."""

import json
import os
import time

import requests

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


class JevError(RuntimeError):
    pass


class Jev:
    def __init__(self, api_key=None, timeout=20.0):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("Set TYPESAFE_API_KEY (see .env.example)")
        self.timeout = timeout
        self.calls = 0
        self.total_latency = 0.0
        self.input_tokens = 0

        # One pooled, keep-alive connection. Opening a fresh TLS connection per call costs
        # ~335ms of handshake - more than Jev takes to answer. Reusing it roughly halves
        # end-to-end latency.
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}",
                                     "Content-Type": "application/json",
                                     "Connection": "keep-alive"})

    def ask(self, state, questions, retries=4):
        """Evaluate `state` against a map of typed questions. Returns (answers, latency)."""
        body = json.dumps({"state": state, "model": MODEL, "questions": questions})
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                r = self.session.post(ENDPOINT, data=body, timeout=self.timeout)
                if r.status_code in (429, 529) and attempt < retries - 1:
                    time.sleep(2**attempt * 0.5)
                    continue
                if r.status_code != 200:
                    raise JevError(f"HTTP {r.status_code}: {r.text[:400]}")
                payload = r.json()
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt * 0.5)

        latency = time.perf_counter() - t0
        self.calls += 1
        self.total_latency += latency
        self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
        return payload["answers"], latency
