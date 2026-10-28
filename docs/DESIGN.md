# lichess-otb — Technical Design

Status: **draft for review** · Scope: v0 (webcam → recognized moves → PGN), fully client-side.

---

## 1. Goal

A static web app (no backend) that watches a physical chessboard through a webcam
and turns what happens on the board into a stream of legal chess moves.

v0 deliverable:

1. Open the page, grant camera access (or load a recorded video file).
2. Locate the board in the image (automatic, with a manual 4-corner fallback).
3. Figure out the board orientation (which corner is a1, mirrored or not).
4. Verify the starting position is set up.
5. Detect each move as it is played, validate it with the rules of chess, and
   show it on a 2D board plus a move list.
6. Export the game as PGN or open it in the lichess analysis board.

Everything runs in the browser and is hosted as static files (e.g. GitHub Pages).

### Primary camera placement

The camera sits **at one player's end of the board, raised above that player**
(behind and above White or Black), looking down the files toward the opponent.
In the image, one back rank is near the camera and the other is far away.

This is the setup the app is built and tested for first. Existing tools such as
ChessCam assume a camera at the **side** of the board, looking across the files,
and fail in this setup. That is the main reason this project exists.

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

### Non-goals for v0

- **Recognizing piece *types* from pixels.** We track the game from a known
  position instead (section 5.1). An ML piece classifier is a planned extension
  with a slot already designed for it (section 11).
- Live play on lichess (Board API). It is also server-less (OAuth PKCE plus
  streaming HTTP from the browser), so it is a natural phase 2 (section 11).
- Clocks, multi-board, mobile-first UI.

---

## 2. What the sample footage tells us

The four photos and the 24 s clip (1264×862, about 47 fps) set the working
conditions:

| Observation | Consequence for the design |
|---|---|
| Camera sits at one player's end of the board (rank 1 or rank 8 nearest), about 45–55° above it, and the whole board is in frame. Near ranks look about 1.6× taller than far ranks. | Strong perspective: we need a full homography, not a rotated rectangle. This placement is the primary target (section 1). |
| Pieces are tall. A king's silhouette covers its own square **and about one square behind it** (away from the camera). | Per-square sampling must account for lean (section 4.5). Never classify a square by its full warped area alone. |
| This sample's image is **mirrored**: rank 1 is near the camera, but the h-file is on the left and h1 is a *light* square. On a real board a1 is always dark. Other webcams, apps and OS settings deliver an unmirrored image. | **Both mirrored and unmirrored input are first-class.** Mirroring is treated as part of orientation and detected automatically from the rule "a1 is dark" (section 4.4), with a manual toggle as a fallback. Nothing assumes either case. |
| The camera can sit on any side: photo 1 has rank 1 near the camera, photos 2–4 have rank 8 near. | All four rotations must be supported. |
| Folding board: a hinge seam runs across the middle, and the two halves are slightly misaligned. One square on the 2nd rank has chipped veneer that shows white. | Detection must tolerate one bad square and a small grid misfit. Comparing against a reference image is more robust than comparing against an "ideal" square colour. |
| Hands and forearms cover large parts of the board while moving. There is a bright white table, glare on skin, and auto-exposure drift. | We only analyse the board when it is **stable** (section 4.6). Comparisons need gain compensation. |
| Light pieces on light squares and dark pieces on dark squares have low contrast. | Use texture/edge features and change detection, not raw colour thresholds alone. |
| The user warns that boards, pieces, colours and sizes will vary. | No hard-coded colours. Everything is calibrated per session from the board itself. |

---

## 3. Architecture overview

```
 ┌──────────────── main thread (UI) ─────────────────┐        ┌────────── vision worker ──────────┐
 │                                                   │        │                                   │
 │  FrameSource ──(ImageBitmap @ ~10 fps)───────────────────▶ │  Rectifier (homography + warp)    │
 │   · webcam (getUserMedia)                         │        │        │                          │
 │   · video file (for testing / replay)             │        │        ▼                          │
 │                                                   │        │  SquareSampler (lean-aware ROIs)  │
 │  CalibrationView  ◀── corners / orientation ────────────── │        │                          │
 │   (video + draggable corners + grid overlay)      │        │        ▼                          │
 │                                                   │        │  StabilityGate (motion, hands)    │
 │  GameView                                         │        │        │ stable frame             │
 │   · 2D board (SVG)                                │        │        ▼                          │
 │   · move list / PGN / lichess export              │ ◀───── │  ChangeDetector  ─▶ diff[64]      │
 │   · status + manual correction                    │ events │  OccupancyClassifier ─▶ occ[64]   │
 │                                                   │        │        │                          │
 │  GameTracker (chess.js) ◀───────────────────────────────── │  (observations only, no rules)    │
 │   · MoveInference: observation → legal move       │        │                                   │
 │   · position history, undo                        │        │  BoardDetector (on demand)        │
 │                                                   │        │                                   │
 │  Settings / persistence (localStorage)            │        └───────────────────────────────────┘
 └───────────────────────────────────────────────────┘
```

**Why split it this way.** The worker produces *observations* (per-square
change scores, occupancy, stability). It knows nothing about chess rules. The
main thread owns the *game*, and `MoveInference` turns observations into moves
using chess.js. This keeps the two hard problems apart, and each can be tested
on its own:

- vision: "did square e2 change?"
- inference: "given these changes, which legal move happened?"

### 3.1 Tech stack

| Concern | Choice | Why |
|---|---|---|
| Build | **Vite + TypeScript**, static output in `dist/` | Fast dev loop, trivial static hosting, typed vision code. |
| Rules | **chess.js** (BSD-2) | Legal move generation, SAN, PGN, FEN. Small and well tested. |
| Vision | **Hand-written TypeScript** on `ImageData` / typed arrays | We need only a few primitives: homography, warp, blur, Sobel, box statistics. OpenCV.js is an 8–10 MB wasm download, harder to debug, and its built-in chessboard finder fails on a board with pieces on it. |
| Concurrency | One **Web Worker** with `OffscreenCanvas` | Keeps the UI at 60 fps while vision runs at about 10 fps. Frames reach the worker as transferable `ImageBitmap`s. |
| UI | Plain TS + small DOM/SVG components, no framework | Small surface. The board is about 64 SVG rects plus glyphs. |
| Tests | **Vitest** (unit, fixtures in Node), **Playwright** (e2e with Chromium's fake camera fed by a video file) | Algorithms are tested on real frames from the sample clip. |
| Hosting | GitHub Pages through a GitHub Actions workflow | Zero server. |

Browser support target: current Chrome, Edge, Firefox and Safari 16.4+ (for
`OffscreenCanvas` in workers). `getUserMedia` requires HTTPS or localhost.
GitHub Pages provides HTTPS.

---

## 4. Vision pipeline (worker)

All geometry uses **board coordinates** `(u, v) ∈ [0, 8]²`, where `(0, 0)` is
the first calibration corner, C0. Orientation (section 4.4) later maps board
coordinates to chess squares.

### 4.1 Frame acquisition (`FrameSource`)

- `getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })`,
  with a camera picker (`enumerateDevices`) and a video-file source behind the
  same interface.
- Frames are scheduled with `requestVideoFrameCallback`, falling back to
  `requestAnimationFrame`, and throttled to **about 10 fps**. Chess does not
  need more, and this leaves CPU headroom.
- Each frame is sent as `createImageBitmap(video)` and transferred to the
  worker, at most one frame in flight. If the worker is busy, frames are
  dropped. They are never queued.
- **Mirrored vs. unmirrored cameras.** The source may arrive mirrored (some
  webcams, virtual cameras, OS "mirror my video" settings, recordings from
  FaceTime or Photo Booth) or not. We cannot know which from the stream
  metadata, and we don't need to. The worker analyses the pixels exactly as
  delivered and never un-mirrors them itself. Whether the board appears
  mirrored is resolved as part of orientation (section 4.4), automatically, so
  both kinds of camera behave identically from the user's point of view. The
  on-screen preview is shown as delivered, with an optional "mirror preview"
  toggle that is display-only CSS and never affects analysis.

### 4.2 Geometry (`geometry/`)

- `Homography` maps board coordinates to image coordinates, as a 3×3 matrix.
  - With 4 points: direct linear transform, an 8×8 linear solve by Gaussian
    elimination with partial pivoting.
  - With N > 4 points: normalized DLT least squares (Hartley normalization),
    solved through the normal equations. Used to refine from many detected grid
    corners.
  - `invert()`, `apply(u, v)`.
- `warp(image, H, outSize)`: inverse-maps every output pixel through H, with
  bilinear sampling. It produces the **rectified board** `R`, 256×256 RGB
  (32 px per square) plus a margin of one square on each side (320×320 total).
  The margin is needed for lean-aware sampling and for border checks during
  detection.
- Cost: 320² ≈ 100k pixels × bilinear is about 1–2 ms per frame. Fine.

### 4.3 Board localization

#### 4.3.1 Manual calibration (always available, built first)

The user drags 4 handles onto the **outer corners of the 8×8 playing area**
(not the frame). A live 9×9 line grid is drawn through the homography so they
can see misalignment immediately. Optional *refine* snaps each handle to the
strongest nearby corner (section 4.3.2 primitives).

The result (`corners[4]`, frame size, camera `deviceId`) is persisted to
`localStorage`, because the camera usually stays put between sessions.

#### 4.3.2 Automatic detection (`BoardDetector`)

Classic X-corner lattice fitting. It runs on demand ("Detect board" button) and
when calibration is missing. It is not run every frame.

1. **Downscale** to about 640 px wide, convert to grayscale, Gaussian blur
   (σ ≈ 1.5).
2. **Saddle response.** Compute the Hessian `Ixx, Iyy, Ixy`.
   `S = Ixy² − Ixx·Iyy` is large and positive at checkerboard inner corners
   (X-corners). Keep local maxima above a threshold set relative to the
   image's 99th-percentile response, about 100–400 candidates.
3. **Seed quads.** For each candidate, take its k = 8 nearest neighbours. Form
   quads of 4 mutually adjacent candidates whose 4 surrounding patches
   alternate dark/light. Each quad defines a homography from the unit square to
   the image.
4. **Grow the lattice (RANSAC-style).** Project integer lattice points
   `(i, j)` for i, j in −8…8 through the seed homography. Count candidates
   within a tolerance of 0.15 square. Refit H with all inliers using the
   normalized DLT, then repeat. Keep the seed with the most inliers. 49 inner
   corners exist, and in the initial position about 21–35 are visible.
5. **Find the 8×8 window.** The lattice fixes the grid but not *which* 8×8 cells
   are the board: pieces hide the edges, and the frame continues the lines.
   Score every candidate window offset with:
   - checker alternation: neighbouring cell means alternate in luminance inside
     the window;
   - frame uniformity: the ring of cells just outside the window does not
     alternate;
   - a strong straight edge along the window border (playing area to frame).

   Pick the best window. This gives the 4 outer corners.
6. **Validate.** Accept only if at least 16 inliers, the reprojection RMS is
   below 0.08 square, and the alternation score passes a threshold. Otherwise
   report "couldn't find the board" and fall back to manual calibration, with
   the best guess pre-filled.

The **hinge seam** looks like an extra straight line, but it falls on a grid
line, so it reinforces the fit. The slight misalignment between halves is
absorbed by the inlier tolerance. The chipped square is just one bad cell in
the alternation score, which is a sum over 64 cells.

> Risk: auto-detection is the most fragile component, which is why it comes
> *after* the end-to-end loop works with manual calibration (section 10).

### 4.4 Orientation (`Orientation`)

The calibrated quad gives a board-coordinate frame. To name squares we need one
of the **8 symmetries** of the square: 4 rotations × {normal, mirrored}.

Automatic resolution happens during setup:

1. **Parity → mirror.** Classify the light/dark pattern of the empty middle
   squares in the rectified image. a1 must be dark. Given a rotation, exactly
   one of {normal, mirrored} puts a dark square at a1. So parity removes the
   mirror ambiguity.
2. **Start position → rotation.** In the starting position, two opposite edges
   hold 16 pieces each. The edge whose pieces are lighter (section 4.7 colour
   score) is White's back side, which fixes rank 1. That leaves 2 of the
   4 rotations, and parity (step 1) picks one.

The result is a symmetry `σ`. `square(σ, cellX, cellY) → "e4"`. The user can
override it with **Rotate 90°** and **Mirror** buttons while watching the
labelled grid overlay. Photo 1, for example, resolves to rank 1 near the camera
and mirrored. The same board through an unmirrored webcam resolves to rank 1
near the camera and not mirrored, with no user action.

Why this works for both camera types: a mirror flips the handedness of the
square pattern. With rank 1 found, a mirrored image puts the dark corner of
rank 1 on the "h side" of the image, and an unmirrored image puts it on the
"a side". Parity alone therefore decides mirrored vs. not. Each of the 8
symmetries is covered by unit tests on synthetic boards rendered both ways.

If a calibration is restored from storage but the camera's mirroring changed
in the meantime (another app toggled it), the stored σ is wrong. The setup
check catches this, because the parity of the dark squares no longer matches
σ, and re-resolves σ.

When the game starts from a FEN instead of the initial position, step 2 isn't
available. Parity still decides the mirror for any assumed rotation, so the
user only confirms which side White is on.

We also record the **camera direction** in board coordinates: the image "down"
vector `dv` mapped into the board frame. Lean-aware sampling uses it.

### 4.5 Lean-aware square sampling (`SquareSampler`)

A piece standing on square *s* covers *s* and leans **away from the camera**
into the square behind it. Pieces from the square in front of *s* lean into the
near part of *s*.

For each square we define two regions in the rectified image:

- **Base ROI.** The part of the square nearest the camera, inset from the grid
  lines. In board units relative to the square, it is the band from 0.15 to
  0.65 of the square depth measured from the near edge, and 0.15 to 0.85 across.
  The base and lower body of the square's own piece are almost always visible
  here. It is the primary signal for "is something standing here?".
- **Body ROI.** The base ROI extended one square away from the camera, used for
  piece **colour**. The body of a piece occupies it, which gives many more
  pixels than the base.

"Near" and "away" come from the camera direction `dv` (section 4.4). For a
camera roughly perpendicular to one board edge, the ROIs are axis-aligned bands.
In general they are rotated rectangles in board coordinates.

The sampler outputs per square, for the base ROI: mean RGB/Lab, luminance
standard deviation, mean Sobel gradient magnitude, and a 16×16 grayscale
thumbnail (for diffing). It outputs mean Lab of the body ROI as well. This is
the only per-frame data that leaves the rectifier.

### 4.6 Stability gate (`StabilityGate`)

We analyse only frames where nothing is moving.

- **Motion** is the mean absolute difference between consecutive 64×64
  downsampled grayscale rectified frames, including the margin. Hands usually
  enter from outside the board, so the margin catches them early.
- **State machine:** `MOVING → SETTLING → STABLE`. The board is STABLE when
  motion stays below `τ_motion` for `N = 6` consecutive processed frames
  (about 0.6 s). Any frame above `τ_motion` returns it to MOVING.
- **Lighting drift:** before diffing, each frame's luminance is normalized by
  matching the median of the *frame margin* (wood) to the reference. This
  compensates auto-exposure and gain changes.
- `τ_motion` adapts to the camera's noise floor. It is measured during the
  first seconds as 4× the median motion while the board is idle.

A hand held perfectly still over the board can produce a "stable" frame.
Change detection handles that case: a large unexplained change is rejected
(section 5.3).

### 4.7 Observations: change and occupancy

Two complementary signals are computed on every STABLE frame.

**(a) Change vector `d[64]` (primary).** We keep a **reference snapshot**: the
square features at the last accepted position. For each square,

```
d[s] = fraction of base-ROI thumbnail pixels whose |ΔL| > τ_px
       after 3×3 blur and gain normalization, in [0, 1]
```

Change detection compares the board against *itself*, so it is independent of
board style, piece style and colours. That makes it the most robust signal and
it drives move inference. When a stable frame shows no significant change, the
reference is refreshed with it, which absorbs slow lighting drift.

**(b) Occupancy `occ[64] ∈ {empty, white, black}` with confidence (secondary).**
It is used for the setup check, the orientation check, the colour of a moved
piece, and sanity checks. It is self-calibrating:

- During setup, before a game is known, the classifier is unsupervised. For
  each square colour class (light/dark by parity), the 8 squares with the
  lowest texture are assumed empty and form the empty-square model (median Lab,
  MAD). A square is *occupied* when its texture or its colour distance from that
  model exceeds k·MAD. Occupied squares are then split into two clusters by
  body-ROI luminance (2-means). The lighter cluster is White.
- During a game, the tracked position *tells us* which squares are empty and
  which hold white or black pieces. After each accepted move the models are
  refit from those labels: empty-light, empty-dark, white-on-light,
  white-on-dark, black-on-light and black-on-dark, each a nearest-centroid
  model over the base and body features. This adapts the classifier to the
  actual pieces, board and light at no cost to the user.

The `Classifier` interface (`classify(features) → {empty | white | black, p}`)
is where an ML model can replace or augment the heuristics later (section 11).

---

## 5. From observations to moves (main thread)

### 5.1 Why tracking instead of recognition

Pixel-level piece-type recognition (telling a bishop from a pawn at this angle,
on any piece set) needs a trained model and a labelled dataset we don't have
yet. A tracked game doesn't need it: the starting position is known, and each
move only has to be one of about 20–40 legal moves. Matching a change pattern
against that short list is far more reliable than recognizing 32 pieces from
pixels.

### 5.2 Move inference (`MoveInference`)

Inputs are the current position `P` (chess.js), the change vector `d[64]`, and
optionally `occ[64]`.

For each legal move `m` in `P`, build its **expected change mask** `E_m[64]`:

| Move kind | Squares with expected change 1 |
|---|---|
| normal / capture | from, to |
| castling | king from/to, rook from/to |
| en passant | from, to, captured-pawn square |
| promotion | from, to (promotion piece defaults to queen, editable in UI) |

The lean also changes the square behind each involved square a little, so
`E_m` gets weight `w_lean ≈ 0.3` on the square directly behind each involved
square (from the camera direction).

Score each move:

```
score(m) = Σ_s E_m[s]·d[s]  −  α · Σ_{s ∉ support(E_m)} d[s]  +  β · occConsistency(m)
```

- The first term rewards change where the move predicts it.
- The second term penalizes unexplained change elsewhere (`α ≈ 1`).
- `occConsistency` (`β` small) checks that after `m` the destination holds the
  right colour and the origin is empty according to `occ`. This term
  disambiguates captures and recaptures.

**Decision:**

- Accept the best move `m*` if `score(m*) ≥ τ_accept` **and** its margin over
  the runner-up is at least `τ_margin`.
- If not, try **2-ply** (`m1` then `m2`, about 1,000 pairs, cheap). This covers
  both sides moving between two stable frames, as in blitz. Accept with stricter
  thresholds.
- If still nothing fits, the change is **unexplained**. Do not touch the game,
  keep the old reference, and show "Board doesn't match — fix the board or
  correct the move". The changed squares are highlighted on the 2D board.

When a move is accepted, apply it, append it to history, refresh the reference
snapshot from the current stable frame, and refit occupancy models
(section 4.7b).

**Why it is robust.** Unrelated noise (a shadow, a slightly nudged piece)
produces small `d` on squares no legal move involves, and the α term absorbs it.
A large wrong change (a hand resting on the board) has big unexplained mass and
gets rejected instead of being turned into a nonsense move.

### 5.3 Edge cases

| Case | Handling |
|---|---|
| Piece nudged but not moved | Small `d` on one square. No legal move matches with margin, so it is ignored. After a few seconds of persistent stability, the reference silently adopts the new frame. |
| Touch-move then return (piece lifted, put back) | No net change, so nothing happens. |
| Capture where the capturing piece briefly sits next to the board | Captured pieces leave the board. Only the final stable state matters. |
| Promotion | Auto-queen, with a one-click change to N, B or R in the move list. |
| Two moves between stable states | 2-ply search (section 5.2). |
| Wrong inference | **Undo** button, and drag pieces on the 2D board to enter the correct move manually. The reference re-syncs on the next stable frame. |
| Position not starting from the initial setup | "Start from FEN" input. Orientation then needs manual confirmation. |
| Camera bumped | A large global change with no legal-move fit, plus a grid misalignment check: the alternation score drops. Prompts the user to recalibrate. |

### 5.4 Setup flow

```
 [Camera] ─▶ [Locate board] ─▶ [Set up pieces] ─▶ [Playing] ─▶ [Finished]
               auto / manual     occ[] matches start position?
                                 orientation auto-resolved
                                 (user confirms grid labels)
```

"Set up pieces" shows live occupancy on the 2D board and turns green when ranks
1, 2, 7 and 8 are occupied, ranks 3–6 are empty, and the colours are on the
correct sides. Then **Start game** begins tracking.

---

## 6. UI

Single page with three panels:

1. **Camera panel.** Live video with a canvas overlay showing the grid, square
   labels, the corner handles in calibration mode, and a per-square change
   heatmap in debug mode.
2. **Board panel.** 2D board in SVG, in the player's orientation, with Unicode
   glyphs (no image assets). It shows the tracked position, highlights the last
   move, and, in setup mode, shows live occupancy dots. In correction mode it
   supports drag-to-move.
3. **Game panel.** Status line (`Waiting for stable board` / `Move detected: e4`
   / `Board doesn't match`), move list in SAN, buttons for Undo, Flip, New game,
   Copy PGN, Download PGN and **Open in lichess analysis**, plus a debug toggle.

**Lichess export** without a server: lichess supports importing a PGN through
its analysis URL, `https://lichess.org/analysis/pgn/<url-encoded PGN>`. For long
games we fall back to copying the PGN and opening `https://lichess.org/paste`.

---

## 7. Persistence

`localStorage`, versioned key `lichess-otb/v1`:

- calibration (corners, frame size, deviceId) and orientation `σ`;
- settings (thresholds overrides, debug, auto-queen);
- the current game (PGN + FEN), so a reload doesn't lose a game in progress.

Every access is wrapped in try/catch. The app works without storage.

---

## 8. Performance budget (per processed frame, mid-range laptop)

| Step | Budget |
|---|---|
| ImageBitmap → worker → `getImageData` (1280×720) | ~3 ms |
| Warp to 320×320 | ~2 ms |
| Motion (64×64) | <0.5 ms |
| Square features (64 ROIs) | ~2 ms |
| Change + occupancy (stable frames only) | ~1 ms |
| Move inference (main thread, stable frames only) | <1 ms (1-ply), ~10 ms (2-ply) |
| Board auto-detect (on demand) | 100–300 ms |

At 10 fps that is about 10% of one core.

---

## 9. Testing strategy

- **Unit (Vitest, Node):**
  - homography round-trips;
  - warp against known synthetic checkerboards;
  - orientation symmetries (all 8, including the "a1 is dark" rule);
  - `MoveInference` on synthetic `d[64]` vectors for every move type (normal,
    capture, castling both sides, en passant, promotion, 2-ply), including
    noisy and adversarial cases.
- **Fixture tests (Node, real pixels):** frames extracted from the sample clip
  and photos, stored as small JPEGs in `test/fixtures/` with hand-labelled JSON
  (corners and occupancy). Tests cover detector corner error, occupancy
  accuracy, and change detection when a piece is added (the clip adds pieces
  one at a time, so each addition is a labelled "change at square X" event).
  Decoding uses `jpeg-js` (a dev dependency).
- **E2E (Playwright, Chromium):**
  `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=clip.y4m`
  runs the real app against a recorded game, and the test asserts on the final
  PGN.
- **Debug tooling in the app:** debug mode shows the rectified board, the ROIs,
  `d[64]` as a heatmap, and the top-5 move scores. Recording a stable-frame
  snapshot (JSON + PNG) turns any field failure into a new fixture.

**Data gap.** The sample clip shows pieces being *placed* on an empty board, not
a game being played. Before the inference step is merged, we need a short
recording of a real game (about 10–20 moves including a capture, and ideally
castling) from the same camera position.

---

## 10. Build plan (one reviewable commit per step)

Each step leaves the app runnable and adds tests for what it introduces.

| # | Commit | What you can review / try |
|---|---|---|
| 1 | `docs: technical design` | This document. |
| 2 | `chore: scaffold Vite + TS + Vitest + lint, GitHub Pages workflow` | Empty app shell deploys. |
| 3 | `feat(capture): webcam + video-file frame source, camera picker` | See the live camera or a loaded clip in the page. |
| 4 | `feat(geometry): homography (DLT/normalized), warp + tests` | Pure math, unit tested. |
| 5 | `feat(calibration): draggable corners, grid overlay, persistence` | Manually lock onto the board and see the grid. |
| 6 | `test: fixture frames from sample clip + labels, jpeg loader` | Ground truth for the next steps. |
| 7 | `feat(vision): worker, rectified board, lean-aware square sampler, debug view` | See the rectified board and ROIs live. |
| 8 | `feat(vision): stability gate + change vector` | Heatmap lights up only on squares that changed once the hand leaves. |
| 9 | `feat(game): chess.js tracker + move inference (1-ply/2-ply) + tests` | Synthetic tests for all move types. |
| 10 | `feat(ui): 2D board, move list, undo/manual correction, PGN + lichess export` | **First end-to-end:** play moves, get PGN. |
| 11 | `feat(vision): self-calibrating occupancy classifier + setup check` | Board panel shows live occupancy, "start position OK". |
| 12 | `feat(orientation): auto rotation/mirror from parity + start position` | No manual orientation needed. |
| 13 | `feat(detect): automatic board detection (X-corners + lattice)` | One-click board finding. |
| 14 | `test(e2e): Playwright fake-camera run on recorded game` | Regression safety net. |

Phase 2 candidates, not in v0: Lichess Board API (play a live lichess game
through the physical board), ML piece-type classifier (arbitrary position
recognition), clocks, voice announcements.

---

## 11. Extension points

- **ML piece classifier.** `Classifier` (section 4.7) takes per-square crops.
  A small CNN (for example a MobileNet-style network at 64×128 crops, exported
  to ONNX and run with `onnxruntime-web`) could output
  `{empty, wP, wN, …, bK}` and enable "recognize any position" plus a FEN
  snapshot. Training data can come from this app itself: every tracked game
  yields labelled crops for free (we know every piece on every square). This is
  the main reason to build tracking first.
- **Lichess Board API.** OAuth2 PKCE (`board:play` scope) in the browser,
  `GET /api/board/game/stream/{id}` (NDJSON over fetch) and
  `POST /api/board/game/{id}/move/{uci}`. The `GameTracker` emits moves, so
  this is an additional sink. Opponent moves arriving from lichess would be
  shown for the user to execute on the physical board, and the tracker then
  expects exactly that move.

---

## 12. Open questions for review

1. **Real game footage.** Can you record a short game (10–20 moves, with a
   capture and castling) from the usual camera position? It is the key test
   asset for steps 8–10.
2. **Primary use case.** Recording OTB games for later analysis (v0 as
   specified), or playing live lichess games on the physical board (pulls the
   Board API into scope sooner)?
3. **Starting positions.** Is "start from the initial position" enough for v0,
   or do you need "start from any position" early? The latter needs the ML
   classifier, or manual FEN entry, which v0 supports.
4. **Hosting.** GitHub Pages from this repo OK?
