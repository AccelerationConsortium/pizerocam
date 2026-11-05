# PiCamera unit — agent guide

A Raspberry Pi Zero 2 W with a Pi Camera and a 12-pixel NeoPixel light ring.
It takes full-resolution snapshots, serves a live MJPEG stream, and switches
the light on/off with a chosen color and brightness.

This service conforms to lab status spec **v1.2** (`ac-organic-lab/docs/STATUS_SPEC.md`).
The binding lab rules are in `ac-organic-lab/docs/AGENTIC_LAB_DESIGN.md`, Part I.
This documentation describes the transport; it does not authorize hardware
execution. Drive the unit only through the `lab-skills` SDK skills
(`pizerocam.snapshot`, `pizerocam.light.on`, `pizerocam.light.off`,
`pizerocam.light.set`), never by composing HTTP to this service.

## Primary operation and states

- **Primary operation:** acquiring a still photo. `activity` is `running` from
  the moment `/control/snapshot` is accepted until the JPEG is stored, otherwise
  `idle`. Streaming and the light are not the primary operation; they are
  reported in `components.camera.state` (`idle`, `streaming`, `capturing`,
  `failed`, `disabled`) and `components.light.state` (`on`, `off`, `failed`,
  `disabled`).
- `equipment_status`:
  - `ready`: camera and light initialized, no capture running.
  - `busy`: a snapshot is being captured (`activity: running`).
  - `degraded`: the light failed or is disabled (snapshots without flash still
    work), or the camera was disabled on purpose (light only).
  - `error`: the camera failed to initialize. The unit cannot take photos.
  - `dry_run`: started with both camera and light disabled.
- `metrics.cycles_total` counts snapshots since service start (`unit: count`).
- Hardware is initialized once at service start. There is no startup/shutdown
  action; restart the service to retry a failed part.

## Claims (hard enforcement)

`POST /control/claim {owner, session_id, ttl_s}` returns a token. `ttl_s` is
clamped to 5–300 s. Send `X-Claim-Token` on every other `/control/*` call, and
heartbeat (`POST /control/heartbeat`) more often than `heartbeat_interval_s`.

- Missing or stale token on a control action: **423** `claim_required`.
- Claim held by another session: **409** `claimed` with `claimed_by` and
  `retry_after_s` (plus a `Retry-After` header).
- Heartbeat with an unknown or expired token: **401** `claim_lost`. Treat the
  claim as lost.
- `POST /control/release` is idempotent (204).

Claims only serialize control. **Releasing or losing a claim changes no
hardware state**: the light stays as it is, and a capture already accepted
finishes.

## Control actions

| Skill | Route | Body | Result |
|---|---|---|---|
| `pizerocam.snapshot` | `POST /control/snapshot` | `{flash?: bool, settle_s?: 0–8}` | `{path, url, taken_at, lens, width, height, bytes}` |
| `pizerocam.light.on` | `POST /control/light/on` | `{color?: [r,g,b], brightness?: 0–1}` | `{on, color, brightness}` |
| `pizerocam.light.off` | `POST /control/light/off` | `{}` | `{on, color, brightness}` |
| `pizerocam.light.set` | `POST /control/light/set` | `{color?, brightness?}` (at least one) | `{on, color, brightness}` |

- `flash: true` switches the light on for the capture (if it was off), waits
  `settle_s` (default 5 s) for exposure and focus, captures, then switches it
  off again. If the light was already on, it is left on and `settle_s`
  defaults to 0.
- A snapshot pauses a running stream for a moment while the camera switches to
  full resolution.
- Light values are commanded values. There is no light sensor, so the response
  confirms a driver write, not measured light output.
- Snapshots are stored on the unit under `/media/snapshots/main/<name>` and the
  oldest are deleted once the cap (default 200) is reached. The unit is not a
  record store: copy any image that matters to BitacoraDB.

## Refusals

| Status | `code` | Meaning |
|---|---|---|
| 409 | `camera_unavailable` / `light_unavailable` | That part failed to initialize or is disabled. Not listed in `allowed_actions`. |
| 409 | `capture_in_progress` | Another snapshot is running (`snapshot` is omitted from `allowed_actions` while busy). |
| 409 | `claimed` | Claim held by another session. |
| 422 | `invalid_request` | Body out of range (color channels 0–255, brightness 0–1, settle_s 0–8). |
| 423 | `claim_required` | No live claim token. |
| 500 | `capture_failed` / `light_write_failed` | Execution failed; recorded in `last_error`. |

`last_error.code` taxonomy: `capture_failed`, `light_write_failed`. It clears
on the next successful control action. Refusals (409/422/423) never set it.

## Ambiguous outcomes

If a snapshot request times out on the caller's side, the capture may still
have completed. Check `details.last_snapshot` and `metrics.cycles_total` on
`/status` before taking another. If a light request times out, read
`details.light` and reconcile before acting again. A light write failure leaves
the light's physical state unknown.

## Reading

- `GET /status`: STATUS_SPEC envelope. `details.light`, `details.camera`
  (resolutions, stream viewers), `details.last_snapshot`, `details.claimed_by`.
- `GET /stream`: MJPEG (`multipart/x-mixed-replace`). On the dashboard, open it
  only through the authenticated camera broker after a viewer chooses Show stream.
- `GET /media` and `GET /media/snapshots/main/<name>`: stored snapshots.

## Authentication

None in the service. Tailnet ACLs and the dashboard edge gate access.
