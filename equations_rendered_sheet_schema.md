# Equations Rendered — Google Sheet Schema

Create one Google Sheets file with these tabs. Exact tab names matter — the
n8n workflow references them directly.

---

## Tab 1: `Queue`

Your ordered backlog — what to render next, pulled from the tagged master
list. One row per item.

| Column | Type | Notes |
|---|---|---|
| `id` | text | Short unique id, e.g. `q-0001`. Just increment. |
| `item_name` | text | The concept name, e.g. "Fibonacci spiral" |
| `simulation_module` | text | Which `simulations/*.py` module this maps to, once you decide (can be blank until then) |
| `config_notes` | text | Any special config overrides / notes for this render |
| `priority` | number | Lower = sooner. Ties broken by row order. |
| `status` | text | One of: `pending`, `in_progress`, `done` |
| `date_added` | date | When you added it to the queue |

The render-trigger workflow reads the lowest-`priority` row where
`status = pending`. When none exist, the queue is considered exhausted.

---

## Tab 2: `Settings`

A single-row control tab holding pipeline-wide flags. Only one data row ever
exists here — the workflow reads/updates it in place, matched via a fixed
`settings_key = "main"` column (added during implementation, since the
Google Sheets node needs a stable column to match on for updates).

| Column | Type | Notes |
|---|---|---|
| `queue_empty_notified` | boolean (`TRUE`/`FALSE`) | Has the "queue is empty" Telegram message already been sent since the queue last had entries? |
| `last_checked` | datetime | Timestamp of the last render-trigger run, for your own visibility |
| `settings_key` | text | Always `"main"` — fixed value used so n8n's Update Row operation can reliably find this one row |

**Logic this drives (implemented in `queue_check_and_notify.json`):**
- Queue has a pending row → proceed to render, and if `queue_empty_notified`
  is `TRUE`, flip it back to `FALSE` (so the *next* time the queue empties,
  you get notified again).
- Queue has no pending rows → if `queue_empty_notified` is `FALSE`, send the
  Telegram ping and set it to `TRUE`. If it's already `TRUE`, do nothing —
  no duplicate messages.

---

## Tab 3: `Tracking Log`

One row per rendered video, lifecycle state through publish. (Referenced in
Phase C's metadata/approval/upload steps — not needed for the queue-empty
logic itself, included here for completeness.)

| Column | Type | Notes |
|---|---|---|
| `video_id` | text | Your own id, e.g. `v-0001` |
| `queue_id` | text | Links back to the `Queue` row it came from |
| `simulation_name` | text | |
| `status` | text | `rendered` → `verified` → `visual_checked` → `metadata_ready` → `pending_approval` → `approved`/`rejected` → `uploaded`/`queued_for_tomorrow` |
| `youtube_video_id` | text | Filled in after upload |
| `youtube_title` | text | |
| `render_date` | date | |
| `upload_date` | date | |