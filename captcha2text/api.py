"""The HTTP API described in docs/API.md.

Run:  uvicorn captcha2text.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from . import admin as admin_panel
from . import web
from .config import Settings
from .errors import APIError, error_response, new_request_id
from .images import ImageError, decode_image
from .keys import ApiKey, KeyStore
from .mathexpr import solve_expression
from .model import ModelNotFound, ModelRegistry
from .ratelimit import RateLimiter

log = logging.getLogger("captcha2text")

_IMAGE_ERROR_STATUS = {"invalid_image": 400, "image_too_large": 413}


def wants_plain_text(request: Request) -> bool:
    types = [part.split(";")[0].strip().lower() for part in request.headers.get("accept", "").split(",")]
    return "text/plain" in types and "application/json" not in types


def authenticate(keys: KeyStore, header: str | None) -> ApiKey:
    if not header:
        raise APIError(
            401, "missing_api_key", "Add the header Authorization: Bearer <API_KEY>.",
            {"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = header.partition(" ")
    token = token.strip()
    key = keys.find(token) if scheme.lower() == "bearer" and token else None
    if key is None or not key.active:
        raise APIError(
            401, "invalid_api_key", "The API key is wrong or has been revoked.",
            {"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
    return key


def decode_base64_image(value: str) -> bytes:
    if value.startswith("data:"):
        head, sep, value = value.partition(",")
        if not sep or ";base64" not in head:
            raise APIError(400, "invalid_image", "The data URL must be base64, like data:image/png;base64,....")
    value = "".join(value.split()).replace("-", "+").replace("_", "/")
    value += "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise APIError(400, "invalid_image", "The image is not valid base64.") from None


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):g} MB"


async def read_body(request: Request, limit: int, max_image_bytes: int) -> bytes:
    too_large = APIError(
        413, "image_too_large", f"The request is too large. Images can be up to {_mb(max_image_bytes)}."
    )
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise too_large
    return bytes(body)


CAPTCHA_TYPES = ("text", "math")


async def image_from_multipart(request: Request, body: bytes) -> tuple[bytes, Any]:
    sent = False

    async def replay() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    try:
        form = await Request(request.scope, replay).form(max_files=1, max_fields=20)
    except (MultiPartException, ValueError, KeyError):
        raise APIError(400, "invalid_body", "The form data could not be read. Send the file in a field named image.") from None
    try:
        item = form.get("image")
        if item is None:
            raise APIError(400, "missing_image", "The request has no image field. Send the file in a field named image.")
        if not isinstance(item, UploadFile):
            raise APIError(400, "invalid_image", "Send image as a file. To send base64, use a JSON body instead.")
        return await item.read(), form.get("type")
    finally:
        await form.close()


def image_from_json(body: bytes) -> tuple[bytes, Any]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise APIError(400, "invalid_body", "The request body is not valid JSON.") from None
    if not isinstance(payload, dict) or payload.get("image") in (None, ""):
        raise APIError(400, "missing_image", 'The JSON has no image field. Send {"image": "<base64>"}.')
    if not isinstance(payload["image"], str):
        raise APIError(400, "invalid_image", "image must be a base64 string.")
    return decode_base64_image(payload["image"]), payload.get("type")


def captcha_type(value: Any, request: Request) -> str:
    """type from the body, else from the query string (?type=math), else "text"."""
    if value is None or value == "":
        value = request.query_params.get("type") or "text"
    if not isinstance(value, str) or value.strip().lower() not in CAPTCHA_TYPES:
        raise APIError(400, "invalid_type", 'type must be "text" or "math".')
    return value.strip().lower()


async def read_image(request: Request, settings: Settings) -> tuple[bytes, str]:
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type not in ("multipart/form-data", "application/json"):
        raise APIError(
            415, "unsupported_media_type", "Send the image as multipart/form-data or application/json."
        )
    # Base64 makes the body about 4/3 of the image size; leave room for form or JSON framing.
    body = await read_body(request, settings.max_image_bytes * 4 // 3 + 64 * 1024, settings.max_image_bytes)
    if media_type == "application/json":
        data, kind = image_from_json(body)
    else:
        data, kind = await image_from_multipart(request, body)
    kind = captcha_type(kind, request)
    if len(data) > settings.max_image_bytes:
        raise APIError(
            413, "image_too_large",
            f"The image is over {_mb(settings.max_image_bytes)}. Send the captcha image itself.",
        )
    return data, kind


def create_app(
    settings: Settings | None = None,
    solver: Any = None,
    keys: KeyStore | None = None,
    models: ModelRegistry | None = None,
) -> FastAPI:
    """`solver` or `models` replace the model folders, for tests."""
    settings = settings or Settings.from_env()

    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.keys = keys or KeyStore(settings.db_path)
        app.state.admins = admin_panel.AdminStore(app.state.keys.db_path)
        if models is not None:
            app.state.models = models
        else:
            preloaded = {settings.default_model: solver} if solver is not None else None
            app.state.models = ModelRegistry(
                settings.models_dir, settings.default_model, settings.ort_threads, loaded=preloaded
            )
        app.state.limiter = RateLimiter()
        # Load the default model now, so a broken one stops the server instead of failing requests.
        default = app.state.models.default
        log.info("default model %s ready (%s)", default.name, settings.default_model)
        yield

    app = FastAPI(
        title="Captcha2Text API", version="1", lifespan=lifespan,
        docs_url=None, redoc_url=None, openapi_url=None,
    )

    app.include_router(web.build_router())
    app.include_router(admin_panel.build_router(settings))

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError) -> JSONResponse:
        return error_response(exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        message = {404: "There is no endpoint at this path.", 405: "This endpoint does not accept that method."}
        return error_response(
            APIError(exc.status_code, code, message.get(exc.status_code, str(exc.detail))), new_request_id()
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return error_response(APIError(500, "internal_error", "Something failed on our side."), new_request_id())

    @app.get("/v1/health")
    async def health(request: Request) -> Response:
        registry = getattr(request.app.state, "models", None)
        if registry is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        return JSONResponse({"status": "ok", "model": registry.default.name})

    @app.post("/v1/solve")
    async def solve(request: Request) -> Response:
        state = request.app.state
        request_id = new_request_id()
        started = time.perf_counter()
        key: ApiKey | None = None
        model_name: str | None = None
        status, error_code, model_ms = 500, None, None
        try:
            key = await run_in_threadpool(authenticate, state.keys, request.headers.get("authorization"))

            wait = state.limiter.check(key.id, key.rpm)
            if wait is not None:
                raise APIError(
                    429, "rate_limited", "Too many requests for this API key. Wait and try again.",
                    {"Retry-After": str(max(1, math.ceil(wait)))},
                )

            data, kind = await read_image(request, settings)

            # Math captchas use the math model; everything else the key's model or the default.
            folder = settings.math_model if kind == "math" else key.model
            try:
                solver = await run_in_threadpool(state.models.get, folder)
            except Exception as e:
                if isinstance(e, ModelNotFound):
                    log.error("key %s needs model %r, which has no folder with a model.json", key.id, folder)
                else:
                    log.exception("could not load model %r for key %s", folder, key.id)
                what = "Math captchas aren't" if kind == "math" else "The model for this API key isn't"
                raise APIError(503, "service_unavailable", f"{what} available right now. Contact support.") from None
            model_name = solver.name

            try:
                rgb = await run_in_threadpool(decode_image, data, settings.max_image_pixels)
            except ImageError as e:
                raise APIError(_IMAGE_ERROR_STATUS[e.code], e.code, e.message) from None

            try:
                result = await run_in_threadpool(solver.solve, rgb)
            except Exception:
                log.exception("model failed on %s", request_id)
                raise APIError(500, "internal_error", "Something failed on our side.") from None

            model_ms = result.processing_ms
            if not result.text:
                raise APIError(422, "no_text_found", "No text was found in the image. Load a new captcha and try again.")

            answer = None
            if kind == "math":
                value = solve_expression(result.text)
                if value is None:
                    raise APIError(
                        422, "invalid_expression",
                        f"The image reads {result.text!r}, which isn't a sum we can work out. "
                        "Load a new captcha and try again.",
                    )
                answer = str(value)

            status = 200
            headers = {"X-Request-Id": request_id}
            if wants_plain_text(request):
                return PlainTextResponse(answer if answer is not None else result.text, headers=headers)
            body = {
                "success": True,
                "request_id": request_id,
                "text": result.text,
                "confidence": result.confidence,
                "processing_ms": result.processing_ms,
            }
            if answer is not None:
                body["answer"] = answer
            return JSONResponse(body, headers=headers)
        except APIError as err:
            status, error_code = err.status, err.code
            return error_response(err, request_id)
        finally:
            total_ms = round((time.perf_counter() - started) * 1000)
            log.info(
                "solve %s key=%s model=%s status=%s code=%s model_ms=%s total_ms=%s",
                request_id, key.id if key else "-", model_name or "-", status, error_code or "-", model_ms, total_ms,
            )
            # Rejected keys and rate-limited calls are not logged, so a flood can't fill the table.
            if key is not None and status != 429:
                try:
                    await run_in_threadpool(
                        state.keys.log_usage, key.id, request_id, status, error_code, model_ms, model_name
                    )
                except Exception:
                    log.exception("could not write usage for %s", request_id)

    return app


app = create_app()
