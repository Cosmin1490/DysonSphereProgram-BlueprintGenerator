# TODO

## CP cost formula
- [ ] Decompile current game version's `DysonShell.cs` to get the exact CP (Cell Point / solar sail) formula
- [ ] Our estimate in `script_cost.py` is ~16% off: 453,440 estimated vs 527,612 in-game at R=10000 for C60
- [ ] Once correct, optimize C60 rotation to minimize stress (`max_stress` / `absolute_latitude`) and maximize CP count

## SP seg count formula (fixed)
The game's seg count formula is `round(angle * radius / 600) * 2` — the `* 2` must be **outside** the rounding:
```python
# Correct
seg_count = int(float(angle * radius / 600 + 0.5)) * 2

# Wrong (was producing odd seg counts, 300 SP too high for C60)
seg_count = int(float((angle * radius / 600 + 0.5) * 2))
```

## Node count record
- [ ] Current best: 2,582 nodes (Fibonacci + ThresholdPenaltyOptimizer)
- [ ] Record: linlin's 2,724 nodes (jammed Tammes packing)
- [ ] Range 2,583-2,723 unexplored
- [ ] Approaches to try: simulated annealing, longer optimizer runs, hybrid strategies

## Proven results
- C60 (truncated icosahedron) is the **mathematically optimal** Dyson Sphere for SP cost
  - V <= 56 impossible (geometric theorem using spherical solid angle bounds)
  - V = 58 computationally ruled out (only +2.1% theoretical margin, no valid polyhedron found)
  - V = 60: only C60's icosahedral symmetry passes shell_size validation (cd2 = 0.156 vs 0.161 limit)
- SP cost at R=10000: **14,400 SP** (1,800 node + 12,600 frame)
- Published: https://www.dysonsphereblueprints.com/blueprints/dyson-sphere-best-cost-efficiency-optimized-sphere-design-60-nodes-15-cheaper-than-football
