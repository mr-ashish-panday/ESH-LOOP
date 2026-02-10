# ESH-Loop: Adaptive Compute via Entropy-Gated Pondering

**Paper #2 in the ESH Series**

## Core Idea
ESH-Loop extends the Entropy-Steered Hybridization (ESH) architecture with **adaptive-depth pondering**. Instead of processing every token with the same number of layers, ESH-Loop uses the entropy router to decide how many times to re-process each token:

- **Easy tokens** (low α) → 1 pass → fast
- **Medium tokens** (mid α) → 2 passes → balanced
- **Hard tokens** (high α) → 3 passes → thorough

## Key Innovation
Unlike PonderNet (Banino et al., 2021), which requires a separate halting network, ESH-Loop **reuses the entropy router** as the halting signal. The same mechanism that routes between SSM and Attention also controls computational depth.

## Architecture
```
Input → [ESHLoopBlock × N]
              ↓
         Router(x) → α
              ↓
    α < thresh → 1 pass (halt)
    α > thresh → re-process (ponder)
              ↓
         Output + ponder_cost
```

## Results (Coming Soon)
- FLOPs saved on easy tokens
- Accuracy gained on hard tokens
- Comparison with fixed-depth ESH baseline

## Citation
```
@article{pandey2026eshloop,
  title={ESH-Loop: Adaptive Compute via Entropy-Gated Pondering},
  author={Ashish Pandey},
  year={2026}
}
```
