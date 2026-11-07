# Matched 2x3 composites (rockets, relative Dubins, murmuration)

Each composite has two rows.

- **Top row:** saved value-function strip (ground truth, HJ-Gauss, absolute error), cropped from the original animation GIF.
- **Bottom row:** vehicles under the benchmark (reference) field, vehicles under the HJ-Gauss (full) field, and a state-norm gap chart.

Vehicle controls come from the sign of ∇v of each field, not from heuristics (`fieldcontrol.py`). Rockets and Dubins vehicles are the Fig. 4 shuttle from `levtoms.pdf`. Starlings are quadrotors.

## Files

| File | Role |
|---|---|
| `decals.py` | Extracts the Fig. 4 shuttle from `levtoms.pdf` and writes the pursuer and evader decals to `assets/`. |
| `assets/` | Decal PNGs (magenta background = key). Generated, but committed so the composites can be rebuilt without re-extracting. |
| `fieldcontrol.py` | Loads a value archive, interpolates v and ∇v in time, and integrates the vehicle dynamics with RK4. |
| `matched3col.py` | Renders the composite (MuJoCo bottom row, chart, BRT annotation). |
| `make_composites.sh` | End-to-end driver: decals, value solves, composites. |

## Requirements

- Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`, `pillow`, `mujoco`, and this package on `PYTHONPATH`.
- `pdftoppm` (poppler), used by `decals.py`.
- The original animation GIFs (`roysoc_rockets.gif`, `roysoc_dubins.gif`, `roysoc_murmuration.gif`) for the top rows.

## Reproduce

From `monte_carlo/`:

```sh
./make_composites.sh \
  --pdf ~/Documents/Papers/MSRYears/TOMS/levtoms.pdf \
  --anim-dir ~/Documents/blog/downloads/Papers/assets/anim/originals_1x3 \
  --work /path/to/scratch
```

This writes `WORK/fields/*.npz` (value fields, about 130 MB each at n=160) and `WORK/composites/roysoc_{rockets,dubins,murmuration}.gif` plus `_poster.png` files. The solves take tens of minutes each on a laptop; the three renders take a few minutes each.

To run a single step by hand:

```sh
export PYTHONPATH=$PWD
python -m hjgauss.render.decals --pdf ~/Documents/Papers/MSRYears/TOMS/levtoms.pdf
python -m hjgauss.validation.cases --suite games --cases rockets --n 160 --steps 400 --kind reference --out WORK/fields
python -m hjgauss.validation.cases --suite games --cases rockets --n 160 --steps 400 --kind full --out WORK/fields
python -m hjgauss.render.matched3col --case rockets --hj-root WORK/fields --anim-dir ANIM_DIR --out WORK/composites/roysoc_rockets.gif
```

## Settings that are hard-coded

These were chosen by test, not derived. Change them in `matched3col.py` with care.

- **Control signs** (`SIGN`): −1 for rockets, +1 for Dubins and murmuration. Chosen because the pursuer-driven trajectory lowers v for those signs.
- **Starting states** (`STARTS`): two per game, picked where the reference and HJ-Gauss fields disagree most. Found by random search over the initial region.
- **Grid and horizon:** n = 160, 400 steps, T = 0.5 (rockets), 1.0 (Dubins), 2.0 (murmuration). Value archives must use the same settings as the top-row GIFs.
- **Decal and camera layout** (`DECAL_HALF`, `CAMS`, `GZ`, `TOP_X`): tuned by eye.

## Known issues

- Once, a render crashed with exit code 139 (segfault) and then succeeded on retry. Four later runs were clean. The cause is unknown; if it recurs, rerun the failing case.
- Quadrotor rotors are plain translucent discs.
- Starlings' starting states come from a search over the seven-disk initial condition, which is an illustrative choice, not a reproduction of the paper's experiment.
- The composites are not published. Copying them into `assets/anim/` is a separate, manual step.
