"""Models: shared per-modality encoders, gated fusion, swappable temporal cores
(selective SSM / causal Transformer), and the assembled beam predictor.

The Phase-0 design principle (CLAUDE.md): encoders/fusion/head are IDENTICAL
across the SSM and Transformer variants, so only the temporal core differs and
their core parameter counts are matched within +/-10%.
"""
