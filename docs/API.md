# Captcha2Text API

Send a captcha image, get back the text in it.

| | |
|---|---|
| Base URL | `https://captcha.silifton.com/v1` |
| Auth | `Authorization: Bearer <API_KEY>` |
| Main call | `POST /v1/solve` with the image, the answer comes back in the same response |
| Formats | Request: `multipart/form-data` or JSON (base64). Response: JSON, or plain text on request |

The API is synchronous: one request, one answer. There is no job id to poll.

---

## Quick start

```bash
curl -X POST https://captcha.silifton.com/v1/solve \
  -H "Authorization: Bearer $CAPTCHA2TEXT_KEY" \
  -F "image=@captcha.png"
```

```json
{
  "success": true,
  "request_id": "req_01JD8X4M2Q7Z",
  "text": "kerry",
  "confidence": 0.98,
  "processing_ms": 21
}
```

Type `text` into the captcha field. That's the whole flow.

---

## Authentication

Every request except `GET /v1/health` needs your API key in the `Authorization` header:

```
Authorization: Bearer c2t_live_xxxxxxxxxxxxxxxxxxxxxxxx
```

- Keys start with `c2t_live_`.
- Never put the key in a query string. URLs end up in logs.
- Call the API from your server only. A key inside a web page or mobile app can be copied by anyone who opens it.
- If a key leaks, contact us and we will revoke it and issue a new one.

---

## What it can read

| | |
|---|---|
| Captcha type | Text captchas: one word drawn on an image. Math captchas: a sum such as `4 + 3 - 5`, see [Math captchas](#math-captchas) |
| Characters | Text: letters `a`–`z`; other characters, such as digits, can be set up for your account. Math: digits, `+`, `-`, `×` |
| Answer case | Always lowercase |
| Image formats | PNG, JPEG |
| Max file size | 2 MB |
| Image size | Any. Send the captcha image itself, not a screenshot of the page |

**Not supported:** reCAPTCHA, hCaptcha, Cloudflare Turnstile, "select all images with…" puzzles, sliders, audio captchas, math with division, brackets or words ("four plus three"), and mixed upper/lower case.

**Accuracy** depends on how your captcha looks. Before you go live, send us 300 or more of your captchas with their answers. We measure accuracy on them, and if it isn't high enough we train a model for your captcha style and set your API key to use it. Use `confidence` to skip answers that are likely to be wrong.

---

## Solve a captcha

```
POST /v1/solve
```

Send the image in one of two ways.

### Option A: file upload (`multipart/form-data`)

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | yes | The captcha image, PNG or JPEG |
| `type` | text | no | `text` (default) or `math`. See [Math captchas](#math-captchas) |

### Option B: JSON with base64 (`application/json`)

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | string | yes | The image bytes, base64-encoded. A `data:image/png;base64,` prefix is allowed |
| `type` | string | no | `text` (default) or `math` |

```json
{ "image": "iVBORw0KGgoAAAANSUhEUgAAAMgAAABQ...", "type": "text" }
```

Use option B when the image is already in memory, for example after your code downloaded it. You can also put the type in the URL instead: `POST /v1/solve?type=math`.

### Headers

| Header | Required | Value |
|---|---|---|
| `Authorization` | yes | `Bearer <API_KEY>` |
| `Content-Type` | yes | `multipart/form-data` or `application/json` |
| `Accept` | no | `application/json` (default) or `text/plain` |

### Response: `200 OK`

```json
{
  "success": true,
  "request_id": "req_01JD8X4M2Q7Z",
  "text": "kerry",
  "confidence": 0.98,
  "processing_ms": 21
}
```

| Field | Type | Description |
|---|---|---|
| `success` | boolean | `true` |
| `request_id` | string | Unique id for this request. Include it when you contact support |
| `text` | string | The text read from the image, lowercase `a`–`z`. For math captchas, the sum as written, e.g. `4+3-5` |
| `answer` | string | Math captchas only: the result of the sum, e.g. `"2"`. Type this into the captcha field |
| `confidence` | number | From 0 to 1. Lower means the answer is more likely wrong. Pick your own cut-off after testing on your captchas |
| `processing_ms` | integer | Milliseconds the model spent on the image |

### Plain-text response

Send `Accept: text/plain` and the body is only the answer (for math captchas, the result of the sum):

```bash
curl -X POST https://captcha.silifton.com/v1/solve \
  -H "Authorization: Bearer $CAPTCHA2TEXT_KEY" \
  -H "Accept: text/plain" \
  -F "image=@captcha.png"
```

```
kerry
```

Errors are still returned as JSON, whatever `Accept` says.

---

## Math captchas

For captchas that show a sum, such as `4 + 3 - 5`, send `type=math`. The API reads the sum, works it out, and returns the result in `answer`:

```bash
curl -X POST https://captcha.silifton.com/v1/solve \
  -H "Authorization: Bearer $CAPTCHA2TEXT_KEY" \
  -F "image=@captcha.png" \
  -F "type=math"
```

```json
{
  "success": true,
  "request_id": "req_01JD8XA1C4RT",
  "text": "4+3-5",
  "answer": "2",
  "confidence": 0.99,
  "processing_ms": 8
}
```

- Supported: whole numbers with `+`, `-` and `×` (also drawn as `x` or `*`). Multiplication is done first, then left to right, so `2+3×4` is `14`. Answers can be negative.
- With `Accept: text/plain` the body is just the answer: `2`.
- If the image doesn't read as a complete sum, you get `422 invalid_expression` with what was read. Load a new captcha and try again.
- Type `answer` into the captcha field, not `text`.

---

## Errors

Every error has the same shape:

```json
{
  "success": false,
  "request_id": "req_01JD8X9V3B1K",
  "error": {
    "code": "invalid_image",
    "message": "The file is not a PNG or JPEG image."
  }
}
```

Branch on `error.code` in your code. `message` is for people and may change. Every response also carries the request id in an `X-Request-Id` header.

| HTTP | `code` | What happened | What to do |
|---|---|---|---|
| 400 | `missing_image` | The request has no `image` field | Send the image in a field named `image` |
| 400 | `invalid_image` | The file is not a PNG or JPEG, or the base64 is broken | Send the original image bytes |
| 400 | `invalid_body` | The body is not valid JSON or form data | Check the request format in the examples below |
| 400 | `invalid_type` | `type` is not `text` or `math` | Send `type=text`, `type=math`, or leave it out |
| 401 | `missing_api_key` | No `Authorization` header | Add `Authorization: Bearer <API_KEY>` |
| 401 | `invalid_api_key` | The key is wrong or revoked | Check the key, or ask us for a new one |
| 413 | `image_too_large` | The image is over 2 MB | Send the captcha image itself, not a screenshot of the page |
| 415 | `unsupported_media_type` | `Content-Type` is not `multipart/form-data` or `application/json` | Use one of the two request formats |
| 422 | `no_text_found` | The model found no text in the image | Load a new captcha and try again |
| 422 | `invalid_expression` | `type=math`, but what was read isn't a complete sum | Load a new captcha and try again |
| 429 | `rate_limited` | Too many requests for your key | Wait the number of seconds in the `Retry-After` header |
| 500 | `internal_error` | Something failed on our side | Retry once. If it keeps failing, send us the `request_id` |
| 503 | `service_unavailable` | The service is starting up or under maintenance | Retry with backoff |

### When to retry

- Retry `429`, `500` and `503`. Wait 1 s, then 2 s, then 4 s, and stop after 3 tries. For `429`, wait for `Retry-After` instead.
- Don't retry other `4xx` errors. The same request will fail the same way.
- Set a client timeout of about 15 seconds.

---

## Rate limits

Each API key has a requests-per-minute limit set by your plan. When you go over it you get `429 rate_limited` with a `Retry-After` header that says how many seconds to wait.

---

## When the answer is wrong

The same image always gives the same answer, so sending it again won't help. If the website rejects the answer, load a new captcha and solve that one.

You can use `confidence` to skip likely-wrong answers: if it is below your cut-off, load a new captcha before submitting.

---

## Code examples

All examples read the key from the `CAPTCHA2TEXT_KEY` environment variable.

### Python

```python
import os
import requests

API_URL = "https://captcha.silifton.com/v1/solve"
API_KEY = os.environ["CAPTCHA2TEXT_KEY"]


def solve_captcha(path: str) -> str:
    with open(path, "rb") as f:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"image": f},
            timeout=15,
        )
    data = resp.json()
    if not data.get("success"):
        err = data["error"]
        raise RuntimeError(f"{resp.status_code} {err['code']}: {err['message']}")
    return data["text"]


print(solve_captcha("captcha.png"))
```

If the image is already in memory, send it as base64:

```python
import base64

resp = requests.post(
    API_URL,
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"image": base64.b64encode(image_bytes).decode()},
    timeout=15,
)
```

### Node.js (18 or newer)

Save as `solve.mjs`:

```js
import { readFile } from "node:fs/promises";

const API_URL = "https://captcha.silifton.com/v1/solve";
const API_KEY = process.env.CAPTCHA2TEXT_KEY;

async function solveCaptcha(path) {
  const bytes = await readFile(path);
  const form = new FormData();
  form.append("image", new Blob([bytes], { type: "image/png" }), "captcha.png");

  const res = await fetch(API_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${API_KEY}` },
    body: form,
    signal: AbortSignal.timeout(15000),
  });
  const data = await res.json();
  if (!data.success) {
    throw new Error(`${res.status} ${data.error.code}: ${data.error.message}`);
  }
  return data.text;
}

console.log(await solveCaptcha("captcha.png"));
```

### PHP

```php
<?php
$apiUrl = 'https://captcha.silifton.com/v1/solve';
$apiKey = getenv('CAPTCHA2TEXT_KEY');

$ch = curl_init($apiUrl);
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => ["Authorization: Bearer $apiKey"],
    CURLOPT_POSTFIELDS => ['image' => new CURLFile('captcha.png', 'image/png')],
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT => 15,
]);
$body = curl_exec($ch);
$status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

$data = json_decode($body, true);
if (empty($data['success'])) {
    throw new RuntimeException("$status {$data['error']['code']}: {$data['error']['message']}");
}
echo $data['text'];
```

### cURL with base64 JSON

```bash
curl -X POST https://captcha.silifton.com/v1/solve \
  -H "Authorization: Bearer $CAPTCHA2TEXT_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 < captcha.png | tr -d '\n')\"}"
```

---

## Health check

```
GET /v1/health
```

No API key needed. Use it for uptime monitoring.

```json
{ "status": "ok", "model": "words-v1" }
```

Returns `200` when the service is ready and `503` while it is starting up.

---

## Versioning

Everything under `/v1` stays compatible. We may add new fields to responses, so your code should ignore fields it doesn't know. Changes that would break existing code will ship under `/v2`, and `/v1` will keep working.

---

## Acceptable use

Use the API only on captchas you own or are allowed to solve: testing your own website, accessibility tools, or automating your own accounts on services that permit it. We suspend keys that are used to break another service's terms.

---

## Support

Email **hello@silifton.com** with your `request_id` and, if you can, the captcha image that failed.
