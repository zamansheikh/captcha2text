"""The error shape every JSON endpoint uses."""

from __future__ import annotations

import secrets

from fastapi.responses import JSONResponse

_ID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


def new_request_id() -> str:
    return "req_" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(12))


def error_response(err: APIError, request_id: str | None = None) -> JSONResponse:
    request_id = request_id or new_request_id()
    return JSONResponse(
        {"success": False, "request_id": request_id, "error": {"code": err.code, "message": err.message}},
        status_code=err.status,
        headers={**err.headers, "X-Request-Id": request_id},
    )
