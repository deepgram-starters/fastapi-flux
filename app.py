"""
FastAPI Flux Starter - Backend Server

Bridges a browser WebSocket to Deepgram's Flux real-time transcription
(v2 listen) using the official `deepgram-sdk` async `AsyncDeepgramClient` and
its `listen.v2` API.

Key Features:
- WebSocket endpoint: /api/flux
- JWT session auth for API protection
- SDK-backed streaming bridge to Deepgram Flux API
"""

import os
import json
import secrets
import time
import asyncio

import jwt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import toml

from deepgram import AsyncDeepgramClient
from deepgram.environment import DeepgramClientEnvironment
from deepgram.listen.v2.types import ListenV2CloseStream

load_dotenv(override=False)

CONFIG = {
    "port": int(os.environ.get("PORT", 8081)),
    "host": os.environ.get("HOST", "0.0.0.0"),
}

def load_api_key():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY required")
    return api_key

API_KEY = load_api_key()

# One async SDK client, reused across connections; the browser never sees the API key.
# DEEPGRAM_BASE_URL (e.g. wss://api.staging.deepgram.com) overrides the default
# production endpoint. listen.v2 uses environment.production for the /v2/listen ws.
def _build_client():
    base_url = os.environ.get("DEEPGRAM_BASE_URL")
    if base_url:
        https = base_url.replace("wss://", "https://").replace("ws://", "http://")
        env = DeepgramClientEnvironment(
            base=https, production=base_url, agent=base_url, agent_rest=https
        )
        print(f"Using custom Deepgram base URL: {base_url}")
        return AsyncDeepgramClient(api_key=API_KEY, environment=env)
    return AsyncDeepgramClient(api_key=API_KEY)


deepgram = _build_client()

# ============================================================================
# SESSION AUTH - JWT tokens for API protection
# ============================================================================

SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
JWT_EXPIRY = 3600  # 1 hour


# Read frontend/dist/index.html for serving
_index_html_template = None
try:
    with open(os.path.join(os.path.dirname(__file__), "frontend", "dist", "index.html")) as f:
        _index_html_template = f.read()
except FileNotFoundError:
    pass  # No built frontend (dev mode)


def require_session(authorization: str = Header(None)):
    """FastAPI dependency for JWT session validation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "MISSING_TOKEN",
                    "message": "Authorization header with Bearer token is required",
                }
            }
        )
    token = authorization[7:]
    try:
        jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Session expired, please refresh the page",
                }
            }
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Invalid session token",
                }
            }
        )


app = FastAPI(title="Deepgram Flux API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SESSION ROUTES - Auth endpoints (unprotected)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve index.html."""
    if not _index_html_template:
        raise HTTPException(status_code=404, detail="Frontend not built. Run make build first.")
    return HTMLResponse(content=_index_html_template)


@app.get("/api/session")
async def get_session():
    """Issues a JWT session token."""
    token = jwt.encode(
        {"iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRY},
        SESSION_SECRET,
        algorithm="HS256",
    )
    return JSONResponse(content={"token": token})


# ============================================================================
# WEBSOCKET ROUTE
# ============================================================================

@app.websocket("/api/flux")
async def flux(websocket: WebSocket):
    """Raw WebSocket proxy endpoint for Flux streaming STT"""
    # Validate JWT from subprotocol
    protocols = websocket.headers.get("sec-websocket-protocol", "")
    protocol_list = [p.strip() for p in protocols.split(",")]
    valid_proto = None
    for proto in protocol_list:
        if proto.startswith("access_token."):
            token = proto[len("access_token."):]
            try:
                jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
                valid_proto = proto
            except Exception:
                pass
            break

    if not valid_proto:
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept(subprotocol=valid_proto)
    print("Client connected to /api/flux")

    # Build Flux connection parameters from query string. Only forward params that
    # are actually present so the SDK/API defaults apply otherwise.
    model = "flux-general-en"
    connect_kwargs = {
        "model": model,
        "encoding": websocket.query_params.get("encoding", "linear16"),
        "sample_rate": websocket.query_params.get("sample_rate", "16000"),
    }
    for name in ("eot_threshold", "eager_eot_threshold", "eot_timeout_ms"):
        value = websocket.query_params.get(name)
        if value is not None:
            connect_kwargs[name] = value
    keyterms = websocket.query_params.getlist("keyterm")
    if keyterms:
        connect_kwargs["keyterm"] = keyterms

    print(
        f"Connecting to Deepgram Flux: model={model}, "
        f"encoding={connect_kwargs['encoding']}, sample_rate={connect_kwargs['sample_rate']}"
    )

    try:
        async with deepgram.listen.v2.connect(**connect_kwargs) as connection:
            print("Connected to Deepgram Flux API")

            # Task to forward messages from Deepgram to the client, preserving the
            # raw Flux message contract (binary passthrough, JSON events as-is).
            async def forward_from_deepgram():
                try:
                    async for message in connection:
                        if isinstance(message, (bytes, bytearray)):
                            await websocket.send_bytes(bytes(message))
                        elif hasattr(message, "model_dump_json"):
                            await websocket.send_text(message.model_dump_json())
                        elif isinstance(message, str):
                            await websocket.send_text(message)
                        else:
                            await websocket.send_text(json.dumps(message))
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"Error forwarding from Deepgram: {e}")
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "Error",
                            "description": str(e),
                            "code": "PROVIDER_ERROR"
                        }))
                    except Exception:
                        pass

            forward_task = asyncio.create_task(forward_from_deepgram())

            # Forward messages from client to Deepgram via the SDK.
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break

                    if message.get("bytes") is not None:
                        await connection.send_media(message["bytes"])
                    elif message.get("text") is not None:
                        try:
                            data = json.loads(message["text"])
                        except (ValueError, TypeError):
                            print("Ignoring non-JSON message from client")
                            continue
                        msg_type = data.get("type")
                        if msg_type == "CloseStream":
                            await connection.send_close_stream(
                                ListenV2CloseStream(type="CloseStream")
                            )
                        elif msg_type == "Configure":
                            await connection.send_configure(data)
                        else:
                            print(f"Ignoring unknown client message type: {msg_type}")

            except WebSocketDisconnect:
                print("Client disconnected")
            except Exception as e:
                print(f"Error forwarding to Deepgram: {e}")
            finally:
                forward_task.cancel()
                try:
                    await forward_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "Error",
                "description": str(e),
                "code": "CONNECTION_FAILED"
            }))
        except Exception:
            pass

    finally:
        print("Connection cleanup complete")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/api/metadata")
async def get_metadata():
    try:
        with open('deepgram.toml', 'r') as f:
            config = toml.load(f)
        return JSONResponse(content=config.get('meta', {}))
    except:
        return JSONResponse(status_code=500, content={"error": "Metadata read failed"})

if __name__ == "__main__":
    import uvicorn
    print(f"\n{'=' * 70}")
    print(f"FastAPI Flux Server: http://localhost:{CONFIG['port']}")
    print(f"")
    print(f"   GET  /api/session")
    print(f"   WS   /api/flux (auth required)")
    print(f"   GET  /api/metadata")
    print(f"   GET  /health")
    print(f"{'=' * 70}\n")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
