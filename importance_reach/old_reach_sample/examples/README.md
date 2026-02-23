# Examples

Example scripts demonstrating the MC Cole-Hopf reachability solver on different systems.

## Scripts

| Script | System | State dim | Description |
|--------|--------|-----------|-------------|
| `ex_double_integrator.py` | Double integrator | 2 | Position-velocity system with bounded acceleration. Plots BRS at multiple time snapshots. |
| `ex_dubins.py` | Dubins vehicle | 3 | Constant-speed vehicle with bounded turn rate. Plots (x, y) slices at fixed heading angles. |
| `ex_rockets.py` | Two-rockets pursuit-evasion | 3 | Relative-coordinate rocket game. Plots (x, z) slices of the backward reachable tube. |
| `ex_dubins_3d_comparison.py` | Dubins (relative) | 3 | MC vs levelsetpy comparison: 2D slices with error maps. |
| `ex_rockets_3d_comparison.py` | Two-rockets | 3 | MC vs levelsetpy comparison: 2D slices + 3D BRT isosurface overlay. |

## Running

From the project root:

```bash
JAX_PLATFORMS=cpu python examples/ex_double_integrator.py
JAX_PLATFORMS=cpu python examples/ex_dubins.py
JAX_PLATFORMS=cpu python examples/ex_rockets.py
```

The comparison scripts require levelsetpy (grid-based solver):

```bash
JAX_PLATFORMS=cpu python examples/ex_dubins_3d_comparison.py
JAX_PLATFORMS=cpu python examples/ex_rockets_3d_comparison.py
```

Each script saves PNG files in this directory.

## Output

- `double_integrator.png` -- BRS contours at 4 time snapshots
- `dubins.png` -- BRS slices at 4 heading angles
- `rockets.png` -- BRT cross-sections at 4 relative thrust angles
- `dubins_3d_comparison.png` -- MC vs levelsetpy 2D slice comparison (3x3 grid)
- `rockets_3d_slices.png` -- MC vs levelsetpy 2D slice comparison for rockets
- `rockets_3d_brt.png` -- Side-by-side 3D BRT isosurface comparison
- `rockets_3d_overlay.png` -- 3D BRT overlay (both meshes in same axes)
