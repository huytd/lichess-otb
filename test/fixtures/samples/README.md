# Sample footage

The original sample images and video, kept as reference and as the source for
test fixtures. These files are the originals and are never edited. Derived
fixtures (extracted frames, labels) go elsewhere under `test/fixtures/`.

All of them come from the same wooden folding board and piece set, photographed
from one side of the board at about 45–55° elevation. The image is **mirrored**
(h1 is a light square). See `docs/DESIGN.md` section 2.

| File | Contents |
|---|---|
| `photo-01-rank1-near.webp` | 1264×868. Rank 1 near the camera, h-file on the left. Mid-game position, few pieces. |
| `photo-02-rank8-near.webp` | 1264×868. Camera on the other side (rank 8 near). Mid-game position. |
| `photo-03-rank8-near.webp` | Same position as photo 02, a moment later (the hand on the right edge has moved). Useful as a near-duplicate pair for stability and noise tests. |
| `photo-04-rank8-near.webp` | 1264×868. Rank 8 near, different position with more pieces. |
| `clip-01-placing-pieces.mov` | 1264×862, H.264, about 47 fps, 24 s. Rank 1 near (same view as photo 01). Starts with an empty board, then pieces are placed one at a time, with hands entering and leaving the frame. This is **not** a played game. |

Board details worth knowing when testing:
- A hinge seam runs across the middle of the board (between ranks 4 and 5), and
  the two halves are slightly misaligned.
- One square on the 2nd rank has chipped veneer that shows white.
