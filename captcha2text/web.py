"""Web pages served next to the API: the landing page, the API docs and the admin panel.

Each page is one HTML file in web/. Inline scripts are allowed by their SHA-256 hash in the
Content-Security-Policy, so nothing else can run on these pages.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response

WEB_DIR = Path(__file__).parent / "web"
_INLINE_SCRIPT = re.compile(r"<script>(.*?)</script>", re.S)

# docs.html is also published as a standalone page without a document wrapper; add one here.
_WRAP_START = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    '<link rel="icon" href="/favicon.svg" type="image/svg+xml">'
    "<style>:root{padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}"
    "body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>"
    "</head><body>"
)

FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#2b4dd3"/>'
    '<path d="M6 19c4-7 7 3 10-3s6 5 10-2" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/>'
    "</svg>"
)


class Page:
    def __init__(self, filename: str):
        html = (WEB_DIR / filename).read_text()
        if not html.lstrip().lower().startswith("<!doctype"):
            html = _WRAP_START + html + "</body></html>"
        hashes = [
            "'sha256-" + base64.b64encode(hashlib.sha256(s.encode()).digest()).decode() + "'"
            for s in _INLINE_SCRIPT.findall(html)
        ]
        self.body = html.encode()
        self.csp = "; ".join([
            "default-src 'none'",
            "script-src 'self' " + " ".join(hashes),
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src https://fonts.gstatic.com",
            "img-src 'self' data:",
            "connect-src 'self'",
            "base-uri 'none'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ])

    def response(self, private: bool = False) -> Response:
        headers = {
            "Content-Security-Policy": self.csp,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-store" if private else "no-cache",
        }
        if private:
            headers["X-Robots-Tag"] = "noindex, nofollow"
        return Response(self.body, media_type="text/html; charset=utf-8", headers=headers)


def build_router() -> APIRouter:
    router = APIRouter()
    landing, docs, admin = Page("index.html"), Page("docs.html"), Page("admin.html")

    @router.get("/", include_in_schema=False)
    async def landing_page() -> Response:
        return landing.response()

    @router.get("/docs", include_in_schema=False)
    async def docs_page() -> Response:
        return docs.response()

    @router.get("/admin", include_in_schema=False)
    async def admin_page() -> Response:
        return admin.response(private=True)

    @router.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> Response:
        return Response(FAVICON, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})

    @router.get("/robots.txt", include_in_schema=False)
    async def robots() -> Response:
        return PlainTextResponse("User-agent: *\nAllow: /\nDisallow: /admin\n")

    return router
