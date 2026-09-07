import asyncio
import json
import os
import threading
import unittest
from types import SimpleNamespace

os.environ.setdefault("DEEPGRAM_API_KEY", "test-api-key")

from fastapi.testclient import TestClient
from pydantic import BaseModel

import app
from deepgram.core.api_error import ApiError


class TurnInfo(BaseModel):
    type: str = "TurnInfo"
    event: str = "StartOfTurn"
    turn_index: int = 1
    transcript: str = "hello"


class FakeConnection:
    def __init__(self):
        self.media = []
        self.configure = None
        self.close_stream = None
        self.close_stream_sent = threading.Event()

    async def __aiter__(self):
        yield b"audio-response"
        yield TurnInfo()
        while True:
            await asyncio.sleep(3600)

    async def send_media(self, message):
        self.media.append(message)

    async def send_configure(self, message):
        self.configure = message

    async def send_close_stream(self, message):
        self.close_stream = message
        self.close_stream_sent.set()


class FakeConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeListenV2:
    def __init__(self):
        self.connection = FakeConnection()
        self.connect_kwargs = None

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs
        return FakeConnectionContext(self.connection)


class RejectedConnectionContext:
    async def __aenter__(self):
        raise ApiError(
            status_code=400,
            headers={"Authorization": "Token test-api-key"},
            body="Invalid request",
        )

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class RejectedListenV2:
    def connect(self, **kwargs):
        return RejectedConnectionContext()


class FluxBridgeTests(unittest.TestCase):
    def setUp(self):
        self.original_deepgram = app.deepgram

    def tearDown(self):
        app.deepgram = self.original_deepgram

    def _token(self, client):
        response = client.get("/api/session")
        self.assertEqual(response.status_code, 200)
        return response.json()["token"]

    def test_bridge_forwards_media_controls_and_events(self):
        listen_v2 = FakeListenV2()
        app.deepgram = SimpleNamespace(listen=SimpleNamespace(v2=listen_v2))

        with TestClient(app.app) as client:
            token = self._token(client)
            with client.websocket_connect(
                "/api/flux?eot_threshold=0.6&keyterm=Deepgram&keyterm=Flux",
                subprotocols=[f"access_token.{token}"],
            ) as websocket:
                self.assertEqual(websocket.receive_bytes(), b"audio-response")
                self.assertEqual(
                    json.loads(websocket.receive_text()),
                    {
                        "type": "TurnInfo",
                        "event": "StartOfTurn",
                        "turn_index": 1,
                        "transcript": "hello",
                    },
                )
                websocket.send_bytes(b"browser-audio")
                configure = {
                    "type": "Configure",
                    "thresholds": {"eot_threshold": 0.4},
                }
                websocket.send_text(json.dumps(configure))
                websocket.send_text(json.dumps({"type": "CloseStream"}))
                self.assertTrue(listen_v2.connection.close_stream_sent.wait(timeout=1))

        self.assertEqual(listen_v2.connection.media, [b"browser-audio"])
        self.assertEqual(listen_v2.connection.configure, configure)
        self.assertEqual(listen_v2.connection.close_stream.type, "CloseStream")
        self.assertEqual(listen_v2.connect_kwargs["eot_threshold"], "0.6")
        self.assertEqual(listen_v2.connect_kwargs["keyterm"], ["Deepgram", "Flux"])

    def test_api_errors_are_sanitized_before_browser_delivery(self):
        app.deepgram = SimpleNamespace(listen=SimpleNamespace(v2=RejectedListenV2()))

        with TestClient(app.app) as client:
            token = self._token(client)
            with client.websocket_connect(
                "/api/flux", subprotocols=[f"access_token.{token}"]
            ) as websocket:
                error = json.loads(websocket.receive_text())

        self.assertEqual(error["code"], "CONNECTION_FAILED")
        self.assertEqual(error["description"], "Deepgram rejected the connection (HTTP 400)")
        self.assertNotIn("test-api-key", error["description"])
        self.assertNotIn("Authorization", error["description"])


if __name__ == "__main__":
    unittest.main()
