# Backend visual explainer

Open [index.html](index.html) in a modern browser. This is a self-contained, offline,
16:9 animated presentation of the **implemented** Industrial Safety AI backend on the
`feat/backend-term1-integration` branch. It does not call the API or require a running
database. Arrows and moving dots illustrate data flow; they are not a live telemetry feed.

## Scene sequence

1. **System map:** a structured event sender, FastAPI, validation, service, PostgreSQL.
2. **Ingest:** `POST /api/v1/events` and the Pydantic request contract.
3. **Save:** one atomic transaction; a normal event creates one row, a violation creates
   linked event, violation and alert rows; HTTP 201 confirms the result.
4. **Read:** clients query event, violation and alert list/detail APIs.
5. **Errors:** invalid payload (422), duplicate event ID (409), other integrity error (500).
6. **Health:** `GET /health` obtains a database session and runs `SELECT 1`. A successful
   check returns 200; a failed query returns a generic 503 and logs the failure.
7. **Boundaries:** local PostgreSQL and migrations, alongside features that are not
   implemented: authentication, frontend, AI runtime, queues, media storage and external
   notification delivery.

The repository's `src/industrial_safety` detection, tracking, safety and pipeline
packages currently contain only package placeholders. `backend/docker-compose.yml`
defines a local PostgreSQL service, not a deployed API or AI worker. An `alert` in this
backend is a database record readable through the API.
The FastAPI route functions serve as the controller layer; there is no separate
`controllers` package or authentication step in the current request path.

## Visual-to-code map

| Visual component | Actual source |
|---|---|
| Sample event sender and payloads | `backend/scripts/send_sample_event.py`, `backend/examples/*.json` |
| HTTP application and routes | `backend/app/main.py`, `backend/app/api/routes/{events,violations,alerts,health}.py` |
| Request and response validation | `backend/app/schemas/{event,violation,alert}.py` |
| Ingest, deduplication, rollback and read queries | `backend/app/services/{events,violations,alerts}.py` |
| Database session and configuration | `backend/app/db/session.py`, `backend/app/core/config.py` |
| Tables and links | `backend/app/models/entities.py`, `backend/alembic/versions/0001_initial.py` |
| Local database service | `backend/docker-compose.yml` |

## Present and record

1. Open `index.html` by double-clicking it (or use **Open File** in your browser).
2. Press **F** or click **Fullscreen**. It scales to fit a 16:9 display. Press **Esc** to
   leave fullscreen. The animation advances automatically every 11.5 seconds; press
   **Space** to pause, or use **← / →** and the scene buttons to present manually.
3. To export an MP4, capture the browser in OBS Studio at 1920×1080, 30 fps. Select
   browser/window capture, start recording before scene 1, and stop after scene 7.
   OBS may save MKV by default; use **File → Remux Recordings** to make an MP4.
4. To make a GIF from an MP4 with FFmpeg:

   ```bash
   ffmpeg -i backend-explainer.mp4 -vf "fps=12,scale=1280:-1:flags=lanczos,palettegen" palette.png
   ffmpeg -i backend-explainer.mp4 -i palette.png -lavfi "fps=12,scale=1280:-1:flags=lanczos[x];[x][1:v]paletteuse" backend-explainer.gif
   ```

The file uses only local HTML, CSS and JavaScript. No fonts, images, CDN assets, build
step or network connection are needed.
