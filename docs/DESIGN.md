# lichess-otb — Technical Design

Status: **draft for review** · Revision 2 (YOLO-based piece detection)

---

## 1. Goal

A static web app (no backend) that watches a physical chessboard through a webcam
and turns what happens on the board into a stream of legal chess moves.

v1 deliverable:

1. Open the page, grant camera access (or load a recorded video file).
2. Locate the board in the image (manual 4-corner calibration first, automatic
   later).
3. Figure out the board orientation: which corner is a1, and whether the image
   is mirrored.
4. Verify the starting position is set up.
5. Detect each move as it is played, validate it with the rules of chess, and
   show it on a 2D board plus a move list.
6. Export the game as PGN or open it in the lichess analysis board.

Everything runs in the browser and is hosted as static files (GitHub Pages).
Pieces are recognized by a **YOLO object detector** running in the browser with
ONNX Runtime Web. The model is trained by us, on footage from the target camera
setup (section 7).

### 1.1 Primary camera placement

The camera sits **at one player's end of the board, raised above that player**
(behind and above White or Black), looking down the files toward the opponent.
In the image, one back rank is near the camera and the other is far away.

This is the setup the app is built and tested for first. Existing tools such as
ChessCam assume a camera at the **side** of the board, looking across the files,
and fail in this setup. That is the main reason this project exists, and the
main reason to train our own model: a detector only works well on the
viewpoints it was trained on.

A camera at the side of the board stays supported (orientation handles all four
rotations), but it is secondary.

What this placement implies:

- Pieces lean and occlude along files. Near pieces partly cover the pieces
  behind them on the same file. The near player's back rank covers their own
  pawns, and the far side's pawns cover the far back rank.
- The far ranks are foreshortened, about 1.6× smaller than the near ranks in
  the samples.
- Hands enter from both ends of the image: the near player's from the bottom,
  the opponent's from the top.
- Either colour can be near the camera.

### 1.2 Non-goals for v1

- Live play on lichess (Board API). It is also server-less (OAuth PKCE plus
  streaming HTTP from the browser), so it is the natural next phase
  (section 13).
- Chess variants, clocks, multi-board, mobile-first UI.

### 1.3 License

The repository is **AGPL-3.0**. We train with Ultralytics YOLO, whose code and
trained weights are AGPL-3.0. The app is open source and served as static
files, so this costs nothing in practice.

---

## 2. What the sample footage tells us

The four photos and the 24 s clip (1264×862, about 47 fps) in
`test/fixtures/samples/` set the working conditions:

| Observation | Consequence for the design |
|---|---|
| Camera sits at one player's end of the board (rank 1 or rank 8 nearest), about 45–55° above it, and the whole board is in frame. Near ranks look about 1.6× taller than far ranks. | Strong perspective: we need a full homography, not a rotated rectangle. |
| Pieces are tall. A king's silhouette covers its own square **and about one square behind it** (away from the camera). | A piece belongs to the square under its **base**, not the square where most of its pixels are. We map each detection's base point to the board (section 5.4). |
| This sample's image is **mirrored**: rank 1 is near the camera, but the h-file is on the left and h1 is a *light* square. On a real board a1 is always dark. Other webcams, apps and OS settings deliver an unmirrored image. | **Both mirrored and unmirrored input are first-class.** Mirroring is part of orientation and is detected automatically (section 5.3), with a manual toggle as a fallback. Nothing assumes either case. |
| Either end of the board can face the camera: photo 1 has rank 1 near the camera, photos 2–4 have rank 8 near. | All four rotations must be supported, and training data must cover both colours near the camera. |
| Folding board: a hinge seam runs across the middle, and the two halves are slightly misaligned. One square on the 2nd rank has chipped veneer that shows white. | Board detection must tolerate one bad square and a small grid misfit. |
| Hands and forearms cover large parts of the board while moving. There is a bright white table, glare on skin, and auto-exposure drift. | We only analyse the board when it is **stable** (section 5.5). |
| Light pieces on light squares and dark pieces on dark squares have low contrast. | Hard for hand-written rules, routine for a trained detector, provided the training data contains it. |
| The user warns that boards, pieces, colours and sizes will vary. | Nothing is hard-coded to one set. Generalization comes from training-data variety (section 7.4). |

---

## 3. Architecture overview

```
 ┌──────────────── main thread (UI) ──────────────────┐       ┌──────────── vision worker ─────────────┐
 │                                                    │       │                                        │
 │  FrameSource ──(ImageBitmap, ~10 fps)───────────────────▶ │  StabilityGate  (cheap, every frame)    │
 │   · webcam (getUserMedia)                          │       │     │ stable + changed                 │
 │   · video file (testing, dataset mode)             │       │     ▼                                  │
 │                                                    │       │  BoardCrop  (homography → model input) │
 │  CalibrationView ── corners / orientation ───────────────▶ │     │                                  │
 │   (video + draggable corners + grid overlay)       │       │     ▼                                  │
 │                                                    │       │  PieceDetector (YOLO, ONNX Runtime Web)│
 │  GameView                                          │       │     │ boxes + class scores             │
 │   · 2D board (SVG), move list, PGN / lichess       │       │     ▼                                  │
 │   · status + manual correction                     │       │  SquareMapper (base point → square)    │
 │                                                    │ ◀──── │     │ P[64][13]  (per-square classes)  │
 │  GameTracker (chess.js)                            │ obs.  │                                        │
 │   · MoveInference: P[64][13] → legal move          │       │  BoardDetector (on demand)             │
 │   · history, undo                                  │       └────────────────────────────────────────┘
 │                                                    │
 │  DatasetMode: video + PGN → labelled YOLO dataset  │ ──▶ zip download ──▶ training/ (Python, user's GPU)
 │                                                    │                          │
 │  Settings / persistence (localStorage)             │      public/models/pieces-vN.onnx ◀── export
 └────────────────────────────────────────────────────┘
```

**Why split it this way.** The worker produces *observations*: for every square,
a probability for each of 13 classes (empty or one of 12 pieces). It knows
nothing about the rules of chess. The main thread owns the *game*, and
`MoveInference` turns observations into moves using chess.js. The two hard
problems stay apart and can be tested on their own:

- vision: "what is probably standing on each square?"
- inference: "given those probabilities, which legal move happened?"

The detector never has to be perfect. It only has to make the true move more
likely than the other 20–40 legal moves.

### 3.1 Tech stack

| Concern | Choice | Why |
|---|---|---|
| Build | **Vite + TypeScript**, static output in `dist/` | Fast dev loop, trivial static hosting. |
| Rules | **chess.js** (BSD-2) | Legal move generation, SAN, PGN, FEN. |
| Inference | **onnxruntime-web**: WebGPU, with a WebAssembly (SIMD + threads) fallback | Runs any exported ONNX model. WebGPU is fast where available, and WASM works everywhere. |
| Geometry and image ops | Hand-written TypeScript on typed arrays | Only a few primitives are needed (homography, warp, resize, motion). This avoids an 8–10 MB OpenCV.js download. |
| Concurrency | One **Web Worker** with `OffscreenCanvas` | Keeps the UI smooth while inference runs. |
| UI | Plain TS + small DOM/SVG components | Small surface. No framework needed. |
| Training | **Python + Ultralytics YOLO** in `training/`, run on the user's GPU machine | Best-documented training and ONNX export path. |
| Tests | **Vitest** (unit and fixture tests in Node), **Playwright** (e2e with Chromium's fake camera fed by a video file), Ultralytics validation for the model | |
| Hosting | GitHub Pages through a GitHub Actions workflow | Zero server. |

Browser support target: current Chrome, Edge, Firefox and Safari 16.4+.
WebGPU where available, otherwise WASM. `getUserMedia` requires HTTPS or
localhost, and GitHub Pages provides HTTPS.

---

## 4. Coordinates and geometry

**Board coordinates** `(u, v) ∈ [0, 8]²`, in units of one square.
`(0, 0)` is the first calibration corner, C0. Orientation (section 5.3) maps
board cells to chess squares.

`geometry/` provides:

- **`Homography`** (board → image, 3×3).
  - With 4 points: direct linear transform, an 8×8 linear solve.
  - With N > 4 points: normalized DLT least squares (Hartley normalization).
  - `invert()`, `apply(u, v)`.
- **`warp(image, H, out)`**: inverse mapping with bilinear sampling. Used for
  debug views and board detection.
- **`CameraPose.fromHomography(H, imageSize)`** recovers a full projective
  camera from a single view of a planar board. Assume square pixels and a
  principal point at the image centre, so
  `K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]`. Solve `f` from the two
  orthogonality constraints on `K⁻¹H` (columns r1 and r2 orthogonal and of
  equal length, by least squares). Then `r1, r2, t` come from `K⁻¹H`, and
  `r3 = r1 × r2`.

  This lets us **project 3D points above the board**, such as
  `(u, v, height)`. It is used to:
  - generate training boxes from piece-shape priors (section 7.2);
  - find the camera direction on the board, needed to correct base points
    (section 5.4);
  - draw piece-height guides in the calibration overlay.

  It is validated by unit tests on synthetic cameras, and on the samples by
  checking that projected verticals line up with real piece silhouettes.

---

## 5. Vision pipeline (worker)

### 5.1 Frame acquisition (`FrameSource`)

- `getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })`,
  with a camera picker (`enumerateDevices`) and a video-file source behind the
  same interface.
- Frames are scheduled with `requestVideoFrameCallback`, falling back to
  `requestAnimationFrame`, and throttled to about 10 fps for the stability gate.
  Each frame is sent as a transferable `ImageBitmap`, with at most one in flight.
  If the worker is busy, frames are dropped rather than queued.
- **Mirrored vs. unmirrored cameras.** The source may arrive mirrored or not,
  and we cannot tell which from the stream. The worker analyses pixels exactly
  as delivered. Mirroring is resolved as part of orientation (section 5.3). The
  preview can optionally be mirrored for comfort with display-only CSS, which
  never affects analysis.

### 5.2 Board localization

**Manual calibration (built first, always available).** The user drags 4 handles
onto the outer corners of the 8×8 playing area (not the frame). A live 9×9 grid
plus the piece-height guides are drawn through the homography and camera pose,
so misalignment is obvious. Calibration (corners, frame size, camera
`deviceId`) and orientation are persisted, because the camera usually stays put.

**Automatic detection (`BoardDetector`, later).** Classic X-corner lattice
fitting, run on demand, not every frame:

1. Downscale to about 640 px wide, convert to grayscale, Gaussian blur
   (σ ≈ 1.5).
2. Saddle response `S = Ixy² − Ixx·Iyy` from the Hessian. Inner corners of a
   checkerboard (X-corners) are strong maxima. Keep about 100–400 candidates.
3. Seed quads from 4 mutually adjacent candidates whose surrounding patches
   alternate dark/light.
4. Grow the lattice RANSAC-style: project integer points through the seed
   homography, count inliers (tolerance 0.15 square), refit with normalized DLT,
   repeat.
5. Choose the 8×8 window: neighbouring cells alternate in luminance inside it,
   the ring of cells outside it doesn't, and its border is a strong straight
   edge (playing area to frame).
6. Validate: at least 16 inliers, reprojection RMS below 0.08 square. Otherwise
   fall back to manual calibration with the best guess pre-filled.

Later option: every dataset-mode frame comes with known corners, so a small
keypoint model for board corners can be trained from the same data if the
classical detector turns out too fragile.

### 5.3 Orientation

The calibrated quad gives a board frame. To name squares we need one of the
**8 symmetries** of the square: 4 rotations × {normal, mirrored}. This is the
symmetry `σ`, and `square(σ, cellX, cellY) → "e4"`.

**Automatic, during setup:**

- With the detector, each σ predicts where every piece of the starting position
  should be, and we pick the σ with the highest board likelihood (section 6.1).
  The pieces break every symmetry: White versus Black fixes the rotation, and
  king versus queen (e1/d1) fixes the mirror.
- **Independent check: square parity.** a1 must be dark. A mirror flips the
  handedness of the square pattern, so parity alone separates mirrored from
  unmirrored for any rotation. If the two methods disagree, we ask the user.
  If parity stops matching a stored σ (for example, another app toggled the
  camera's mirroring), we re-resolve σ.

**Manual:** **Rotate 90°** and **Mirror** buttons, while watching the labelled
grid overlay.

Unit tests cover all 8 symmetries on synthetic boards, rendered both mirrored
and unmirrored.

### 5.4 Piece detection

**Input crop (`BoardCrop`).** We feed the model the board region, not the whole
frame, so pieces get more pixels:

- Take the axis-aligned bounding box of the board quad.
- Extend it upward by the projected height of a king on the far rank (from the
  camera pose), so far pieces aren't cut off, plus a 4% margin on each side.
- Letterbox to 640×640.

Dataset mode produces exactly the same crops, so training and inference see
the same distribution.

**Model (`PieceDetector`).** A YOLO detector with **12 classes**
(`wP wN wB wR wQ wK bP bN bB bR bQ bK`), exported to ONNX with a 640 input.
The output is boxes with the **full class-score vector** per box, not only the
argmax. Inference uses onnxruntime-web: the WebGPU execution provider, falling
back to WASM. Before use, the session is warmed up with one dummy inference.

**Post-processing.**

- Decode the boxes.
- Class-agnostic NMS (IoU 0.6), unless the exported model is NMS-free.
- Keep boxes down to a low confidence of 0.05. Move inference wants soft
  evidence, not hard decisions.

**Square mapping (`SquareMapper`).** For each box:

1. Take the **bottom-centre** of the box. That point is the front of the
   piece's base, the point nearest the camera.
2. Map it to board coordinates through H⁻¹.
3. Shift it **away from the camera** by the base radius (about 0.3 square,
   using the camera direction from `CameraPose`) to reach the base centre.

This makes lean and occlusion largely irrelevant: a king whose silhouette
covers two squares still has its base on one of them.

Each box `i` (base point `b_i`, class scores `q_i[c]`, objectness `o_i`)
spreads evidence to nearby squares with weights
`w_is ∝ exp(−|b_i − centre_s|² / 2σ²)`, where σ ≈ 0.25 square, normalized
over squares within one square of `b_i`. For each square:

```
P_s(empty) = Π_i (1 − w_is·o_i)
P_s(c)     = (1 − P_s(empty)) · Σ_i w_is·o_i·q_i[c] / Σ_i w_is·o_i      for each of the 12 pieces
```

All probabilities are clipped to `[ε, 1 − ε]` with ε = 0.01, so one confident
mistake can't veto the right move.

**Temporal fusion.** During one stable period the detector runs on up to 3
frames, and the per-square log-probabilities are averaged. This cuts down on
single-frame flukes.

### 5.5 Stability gate and change trigger

The detector costs tens to hundreds of milliseconds, so it runs only when
needed. A cheap gate runs on every frame:

- **Motion:** mean absolute difference between consecutive 64×64 grayscale
  warps of the board plus a one-square margin. Hands usually enter through the
  margin first.
- **State machine:** `MOVING → SETTLING → STABLE`. The board is STABLE once
  motion stays below `τ_motion` for 6 consecutive frames (about 0.6 s).
  `τ_motion` is 4× the measured idle noise floor.
- **Change trigger:** on reaching STABLE, compare the frame with the reference
  snapshot (the last frame for which the position was confirmed). If no square
  changed, skip the detector entirely. If something changed, run the detector
  and move inference. The per-square change values `d[64]` are also passed
  along as weak evidence (section 6.1).
- Luminance is normalized against the frame margin (wood) before comparing, to
  absorb auto-exposure drift.

A hand held perfectly still over the board can look "stable". The detector
then sees an implausible board, nothing explains it well, and the update is
rejected (section 6.2).

---

## 6. From observations to moves (main thread)

### 6.1 Board likelihood

For a hypothetical position `Q`, where `Q[s]` is one of 13 classes:

```
L(Q) = Σ_s log P_s(Q[s])
```

This is the log-probability that the detector would see what it saw if the
board really were `Q`. Squares the detector is unsure about, such as a rank
behind a wall of pawns, contribute almost nothing either way. That is exactly
the right behaviour under occlusion.

Weak change evidence adds a small term
`γ · Σ_{s ∈ changed(P→Q)} log(d_s + ε)`, with `γ` small. It nudges toward moves
whose squares actually changed in the image, and it is the only signal if the
detector is unavailable.

### 6.2 Move inference

Given the current position `P`, the candidates are:

- **H0:** nothing happened (`P` itself);
- **H1:** `P + m` for each legal move `m`, about 20–40 candidates. Castling, en
  passant and all promotion choices are separate candidates, because they
  produce different positions;
- **H2:** `P + m1 + m2`, about 1,000 candidates, considered only if no H1
  candidate is clearly best. It covers both sides moving between two stable
  frames, as in blitz.

**Decision:**

- Accept the best candidate if its log-likelihood beats the runner-up by at
  least `τ_margin`: about 3 nats (roughly 20:1) for H1 and 5 nats for H2.
- If H0 wins, nothing changed meaningfully. Refresh the reference snapshot.
- If no candidate wins clearly, the change is **unexplained**. Leave the game
  untouched, show "Board doesn't match — fix the board or enter the move", and
  highlight the squares where the detector disagrees with the best candidate.

When a move is accepted: apply it, append it to history, and refresh the
reference snapshot.

Why this is robust: the decision is relative. A detector that misreads a bishop
as a pawn on some unrelated square penalizes every candidate equally, so the
right move still wins. Only squares that differ between candidates matter.

### 6.3 Position recognition (snapshot FEN)

The same observations give a whole-board guess. Take the per-square argmax,
then apply constraints:

- exactly one king per colour (keep the most likely one);
- no pawns on the back ranks;
- at most 8 pawns per colour.

The user supplies the side to move and castling rights. This is used for the
setup check, for "start from the current position", and for resyncing after an
unexplained change.

### 6.4 Edge cases

| Case | Handling |
|---|---|
| Piece nudged but not moved | H0 wins, so nothing happens. |
| Touch-move then return | No net change. The change trigger doesn't fire, or H0 wins. |
| Captured piece placed next to the board | Outside the 8×8 area, so its base doesn't map to any square and it is ignored. |
| Promotion | All four promotion pieces are candidates, and the detector picks one. The queen is the default when scores are close. |
| Two moves between stable states | H2 (section 6.2). |
| Wrong inference | **Undo**, plus drag-to-move on the 2D board to enter the right move. The reference re-syncs on the next stable frame. |
| Game not starting from the initial position | "Start from FEN", or "Start from current position" (section 6.3). |
| Camera bumped | Every candidate is unlikely, and parity or grid alignment breaks. The app prompts to recalibrate. |

### 6.5 Setup flow

```
 [Camera] ─▶ [Locate board] ─▶ [Set up pieces] ─▶ [Playing] ─▶ [Finished]
               manual / auto     live snapshot FEN on the 2D board,
                                 orientation auto-resolved,
                                 green when it matches the start position
```

---

## 7. Model and training pipeline

### 7.1 Model

- **Ultralytics YOLO, nano or small size.** Candidates are YOLO11n/s and
  YOLO26n/s. YOLO26 is NMS-free, which simplifies browser post-processing.
  We pick by benchmark: accuracy on our validation games versus browser
  latency.
- Starts from COCO-pretrained weights and is fine-tuned on our data.
- 640×640 input, 12 classes. Exported to **ONNX (opset 17)**, then FP16 for
  WebGPU, and possibly INT8 for WASM if it is accurate enough. Expected size is
  about 3–10 MB.
- Shipped as `public/models/pieces-vN.onnx` with a **model card** JSON
  (`pieces-vN.json`): classes, input size, crop recipe version, training data
  summary and licences, metrics. The app pins an exact model version. A service
  worker caches the model, so the app works offline after the first visit.

### 7.2 Labelled data without drawing boxes: dataset mode

Hand-drawing boxes is the expensive part of YOLO training. We avoid it by
generating labels from a recorded game plus its PGN. This is done **inside the
app** ("Dataset" mode), so it reuses exactly the same geometry, stability gate
and crop code as live recognition:

1. Load a **video** of a game and its **PGN** (from lichess, or entered
   afterwards in an analysis board).
2. Calibrate the corners once for the video, and confirm the orientation.
3. The stability gate splits the video into stable segments. Segment *k* shows
   the position after *k* plies of the PGN, and the change trigger confirms a
   change happened between segments. If segments and plies don't line up, the
   tool flags the mismatch and lets the user nudge the alignment.
4. For each segment, export 1–3 frames, picked to be spread apart and sharp.
5. **Boxes from piece-shape priors.** Each piece type is modelled as a cylinder
   with a type-specific height and radius, in square units. Initial values for
   a standard Staunton set:

   | Piece | Height (squares) | Radius (squares) |
   |---|---|---|
   | Pawn | 0.9 | 0.30 |
   | Knight | 1.15 | 0.33 |
   | Bishop | 1.3 | 0.32 |
   | Rook | 1.0 | 0.33 |
   | Queen | 1.6 | 0.35 |
   | King | 1.8 | 0.35 |

   With the camera pose (section 4), project each cylinder and take the bounding
   box of the projection. A one-time **"fit piece sizes"** step per piece set
   lets the user scale heights with a slider while the projected boxes are drawn
   over the video.
6. Write crops (section 5.4 recipe) and YOLO-format labels, plus `meta.json`
   (source video, corners, σ, FEN per frame, recipe version). Download as a zip.
7. **Review grid:** thumbnails with boxes drawn, for a quick skim. Rejecting a
   frame drops it. There is no per-box editing in v1.

Boxes from priors are slightly loose. That's fine: what matters for our use is
the class and the base point, and both are exact because they come from the PGN
and the geometry.

The same trick applies to the **chesscog** synthetic dataset (MIT licence,
about 5,000 rendered images with FEN and corners), which can be converted into
extra boxes.

### 7.3 How much data

| Stage | Target | Source | Your effort |
|---|---|---|---|
| **v1: your board, your set, your camera** | about 500–800 frames, roughly 10–20k boxes | **8–10 recorded games** | About 10 games, plus a PGN for each and a skim of the review grid |
| v2: other sets and boards | about 2–5k frames | 3–5 more sets at about 200 frames each, chesscog synthetic, permissively licensed public sets | More recordings |

Rationale:

- A 40-move game has about 80 distinct positions, so 10 games give about 800
  genuinely different frames.
- Each frame carries 15–30 pieces. Even the rarest classes get enough instances:
  every frame has both kings, about 800 of each.
- We fine-tune a pretrained model on a narrow domain, which needs far less data
  than training from scratch.
- Consecutive video frames are near-duplicates. Variety matters much more than
  count, which is why we export only 1–3 frames per position.

**What to vary across the 10 games:**

- both camera ends: White near for 5 games, Black near for 5;
- 2–3 lighting conditions (daylight, evening lamp, mixed);
- small changes in camera height and angle between sessions;
- real middlegames with captures. At least one game should include castling on
  both sides, and ideally a promotion.

### 7.4 Splits, augmentation and metrics

- **Split by recording session, never by frame.** Frames from one game are near
  copies, so splitting by frame would leak them across sets. Use 7 games for
  training, 1 for validation and 2 for testing.
- **Augmentation:**
  - HSV and brightness shifts, blur, noise;
  - small rotation, scale and perspective changes;
  - horizontal flip, which is legitimate because cameras can be mirrored and a
    mirrored knight is still a knight;
  - no vertical flip.
  - Mosaic augmentation is on early and turned off for the last epochs.
- **Metrics:**
  - the model: Ultralytics mAP50-95 per class;
  - what we actually care about, measured by replaying the held-out test games
    through the app's own pipeline: **square accuracy** (per-square argmax
    against ground truth) and **move accuracy** (the PGN the app produces
    against the true PGN).

### 7.5 `training/` layout

The training code is run by the user on their GPU machine:

```
training/
  pyproject.toml           # ultralytics, onnx, onnxruntime, onnxconverter-common
  README.md                # setup + commands
  otb_train/
    merge.py               # combine dataset-mode zips (+ optional public sets) → YOLO dataset, session-based split
    train.py               # fine-tune with fixed seed + config
    evaluate.py            # mAP + square/move accuracy on held-out games
    export.py              # ONNX export, FP16 conversion, model card JSON
  configs/pieces-v1.yaml   # classes, image size, augmentation, epochs
```

Expected training time on one consumer GPU: well under an hour for nano or
small with about 1k images.

---

## 8. UI

A single page with three panels:

1. **Camera panel.** Live video with a canvas overlay: the grid, square labels,
   corner handles in calibration mode, and detected boxes plus base points in
   debug mode.
2. **Board panel.** 2D board in SVG, in the player's orientation, with Unicode
   glyphs. It shows the tracked position and highlights the last move. In setup
   mode it shows the live snapshot FEN. In correction mode it supports
   drag-to-move.
3. **Game panel.** Status line (`Waiting for stable board` / `Move detected: e4`
   / `Board doesn't match`), move list in SAN, buttons for Undo, Flip, New game,
   Copy PGN, Download PGN and **Open in lichess analysis**, plus a debug toggle.
   Debug shows the top 5 candidates with their log-likelihoods.

A separate **Dataset mode** page covers section 7.2.

**Lichess export** without a server: open
`https://lichess.org/analysis/pgn/<url-encoded PGN>`. For long games, fall back
to copying the PGN and opening `https://lichess.org/paste`.

---

## 9. Persistence

`localStorage`, versioned key `lichess-otb/v1`:

- calibration (corners, frame size, deviceId), orientation σ, piece-size scale;
- settings (debug, thresholds);
- the current game (PGN + FEN), so a reload doesn't lose a game in progress.

Every access is wrapped in try/catch. The app works without storage. The model
itself is cached by the service worker (Cache Storage).

---

## 10. Performance budget

| Step | When it runs | Budget |
|---|---|---|
| Stability gate (64×64 warp + diff) | every frame, about 10 fps | ~2 ms |
| Crop + letterbox to 640² | stable and changed only | ~3 ms |
| YOLO nano, WebGPU | stable and changed only | est. 15–40 ms |
| YOLO nano, WASM | stable and changed only | est. 150–400 ms |
| Square mapping + move inference | stable and changed only | <1 ms (H1), ~10 ms (H2) |
| Board auto-detect | on demand | 100–300 ms |

The YOLO numbers are estimates and are measured in the detector step. Because
the detector runs only after a move, even the WASM path keeps up: a move is
confirmed about 1 s after the hand leaves.

---

## 11. Testing strategy

- **Unit tests (Vitest, Node):**
  - homography round-trips;
  - camera pose recovery from synthetic cameras;
  - cylinder projection;
  - all 8 orientation symmetries plus the parity rule;
  - square mapping;
  - board likelihood and move inference on synthetic `P[64][13]` covering every
    move kind (castling both sides, en passant, every promotion, H2) and
    adversarial noise.
- **Fixture tests (real pixels):** frames from `test/fixtures/samples/` with
  hand-labelled corners. They check calibration geometry, base-point mapping
  and the stability gate (the sample clip has hands entering and leaving).
- **Model evaluation:** `training/evaluate.py` on held-out games (section 7.4).
- **E2E (Playwright, Chromium):**
  `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=game.y4m`
  runs the real app on a held-out game video. The test asserts that the PGN
  produced equals the true PGN.
- **Field debugging:** a "save snapshot" button exports the current crop,
  detections and probabilities. Any field failure becomes a test fixture.

---

## 12. Build plan (one reviewable commit per step)

Each step leaves the app runnable and adds tests for what it introduces.
**Steps 3–11 need no model**, so they can be built while recordings are made.

| # | Commit | What you can review or try |
|---|---|---|
| 1 | ✅ `docs: technical design` | This document. |
| 2 | ✅ `test: sample footage` | `test/fixtures/samples/`. |
| 3 | `chore: scaffold Vite + TS + Vitest + lint, AGPL license, Pages workflow` | Empty app shell deploys. |
| 4 | `feat(capture): webcam + video-file source, camera picker` | Live camera or a loaded clip in the page. |
| 5 | `feat(geometry): homography, warp, camera pose from homography + tests` | Pure math, unit tested. |
| 6 | `feat(calibration): draggable corners, grid + height overlay, orientation buttons, persistence` | Lock onto your board, and see the grid and piece-height guides line up. |
| 7 | `feat(vision): worker, stability gate, change trigger, debug view` | Status flips between moving and stable as hands come and go. |
| 8 | `feat(game): chess.js tracker, board likelihood, move inference (H0/H1/H2) + tests` | Synthetic tests for every move kind. |
| 9 | `feat(ui): 2D board, move list, drag-to-move, undo, PGN + lichess export` | A working manual game recorder on top of the camera view. |
| 10 | `feat(dataset): piece-shape priors, projected boxes, fit-piece-sizes step` | Boxes drawn over your pieces in the sample clip. |
| 11 | `feat(dataset): video + PGN → stable segments → labelled crops, review grid, zip export` | **Turn your recorded games into a dataset.** |
| — | *you: record 8–10 games (section 7.3)* | |
| 12 | `feat(training): merge, train, evaluate, export ONNX + model card` | Run on your GPU, get `pieces-v1.onnx`. |
| 13 | `feat(detect): ONNX Runtime Web detector, crop, NMS, square mapping, debug overlay, model caching` | Live boxes and per-square probabilities over the camera view. |
| 14 | `feat(game): wire detector into move inference` | **First end-to-end:** play moves, get the PGN. |
| 15 | `feat(setup): snapshot FEN, setup check, auto orientation` | Setup turns green, and orientation needs no buttons. |
| 16 | `feat(detect): automatic board detection (X-corners + lattice)` | One-click board finding. |
| 17 | `test(e2e): Playwright fake-camera replay of a held-out game` | Regression safety net. |

---

## 13. Next phase

- **Lichess Board API.** OAuth2 PKCE (`board:play` scope) in the browser,
  `GET /api/board/game/stream/{id}` (NDJSON over fetch) and
  `POST /api/board/game/{id}/move/{uci}`. `GameTracker` already emits moves, so
  lichess is just another sink. Opponent moves from lichess are shown for the
  user to make on the physical board, and inference then expects exactly that
  move.
- **Data flywheel.** Every game recorded with the app is, once the PGN is
  confirmed, a new labelled dataset, now with the detector's own boxes instead
  of priors.
- **Board-corner keypoint model,** if classical detection proves fragile
  (section 5.2).

---

## 14. Open questions

1. **Games for the dataset:** about 10 games from your setup, varied as in
   section 7.3, each with its PGN. This is needed before step 12.
2. **Piece-size priors:** the defaults assume a standard Staunton set. Step 10
   includes a fitting step, so no measurements are needed from you.
