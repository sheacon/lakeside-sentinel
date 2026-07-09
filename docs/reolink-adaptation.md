# Adapting Lakeside Sentinel for Reolink cameras

## Context

The system is currently wired to Google Nest via a proprietary internal API (`nest-camera-frontend.googleapis.com`) using a Google master-token OAuth flow. This document covers what would be required to instead pull events and clips from a Reolink camera (RTSP + HTTP API).

## Feasibility: yes, with a small abstraction layer

The camera layer is isolated to `src/lakeside_sentinel/camera/`. Everything downstream (frame extraction, ROI crop, YOLO detection, Claude verification, reporting, email) operates on an MP4 file on disk and is fully camera-agnostic. The only Nest-specific surfaces are:

- `camera/nest_api.py` — DASH manifest XML + MP4 clip download
- `camera/auth.py` — Google master-token → Nest-scoped access token
- `camera/models.py:CameraEvent` — already generic (start_time + duration)
- `config.py` — `GOOGLE_MASTER_TOKEN`, `GOOGLE_USERNAME`, `NEST_DEVICE_ID`
- `main.py:Monitor.__init__` — directly instantiates `NestAuth` and `NestCameraAPI`

`CameraEvent` is already protocol-agnostic (start + duration), so no model changes are needed.

## Approach

### 1. Introduce a small camera protocol

Add `src/lakeside_sentinel/camera/protocol.py` with a `Protocol` class defining:

```python
class CameraClient(Protocol):
    def get_events(self, start_time: datetime, end_time: datetime) -> list[CameraEvent]: ...
    def download_clip(self, event: CameraEvent) -> bytes: ...
```

No base class — just a structural typing hook so `Monitor` can accept either implementation.

### 2. Add a Reolink client

New file `src/lakeside_sentinel/camera/reolink_api.py` implementing the protocol:

- **Events**: Reolink's HTTP API exposes motion detection state and recorded file search. Use `Search` command (`cmd=Search`) against `http(s)://<ip>/cgi-bin/api.cgi` with a time range to list on-camera NVR recordings; map each recording's start+duration to `CameraEvent`. Authenticate with `cmd=Login` to obtain a token, cache it, refresh on 401.
- **Clip download**: Use `cmd=Download` with the filename from the search result. Reolink returns MP4 bytes directly — no DASH/MPD parsing needed, which is simpler than the Nest path.
- **Alternative for cameras without local storage**: fall back to RTSP capture (`rtsp://user:pass@ip:554/h264Preview_01_main`) with a short ffmpeg segment around the motion timestamp. Only build this if the target camera lacks an SD card / NVR.

### 3. Config

Extend `config.py` with Reolink fields and a `CAMERA_BACKEND` selector (`"nest"` | `"reolink"`, default `"nest"` to preserve current behavior):

```
CAMERA_BACKEND=reolink
REOLINK_HOST=192.168.x.x
REOLINK_USERNAME=...
REOLINK_PASSWORD=...
REOLINK_CHANNEL=0           # multi-channel NVRs
```

Keep existing Google/Nest vars optional so the two backends coexist.

### 4. Wire the factory in Monitor

In `main.py:Monitor.__init__` (~line 194), replace the hard-coded `NestAuth` + `NestCameraAPI` instantiation with a small factory that reads `CAMERA_BACKEND` and returns a `CameraClient`. Everything else in `Monitor` stays unchanged — it only calls `get_events()` and `download_clip()`.

### 5. Tests

Mirror `tests/test_nest_api.py` with `tests/test_reolink_api.py`, mocking `httpx` responses for `Login`, `Search`, and `Download`. No live camera needed. Add a factory test confirming `CAMERA_BACKEND` selection.

## Critical files

- **New**: `src/lakeside_sentinel/camera/protocol.py`, `src/lakeside_sentinel/camera/reolink_api.py`, `tests/test_reolink_api.py`
- **Modified**: `src/lakeside_sentinel/config.py` (add Reolink vars + backend selector), `src/lakeside_sentinel/main.py` (factory in `Monitor.__init__`), `README.md` (Reolink setup section), `.env.example` if present

## What does NOT change

- `camera/models.py:CameraEvent` — already generic
- `veh_detector.py`, `hsp_detector.py`, frame extraction, ROI crop
- Claude verification, report generation, Resend email
- Daylight window, cleanup, staging/fine-tuning flow
- Review web app (`review.sh`)

## Verification

1. Run existing suite unchanged: `uv run pytest tests/` — proves Nest path still works.
2. Run new Reolink tests: `uv run pytest tests/test_reolink_api.py`.
3. Point a real Reolink camera at a driveway, set `CAMERA_BACKEND=reolink` + credentials, run `uv run python -m lakeside_sentinel.main` over a window known to contain motion, confirm an email alert lands with the expected clip + verification.
4. `uv run ruff check . && uv run mypy .` clean.

## Open question

Does the target Reolink model have local storage (SD card or NVR)? That decides whether the adapter uses the Search/Download API (simple, preferred) or needs an RTSP-capture fallback (more code, ffmpeg dependency). Worth confirming before implementing.
