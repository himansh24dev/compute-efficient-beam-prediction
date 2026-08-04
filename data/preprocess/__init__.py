"""Pure-numpy per-modality preprocessing.

Nothing here imports torch, so preprocessing can be unit-tested and run on the
raw files with only numpy/pandas/opencv installed. The torch Datasets in
`data/dataset.py` call these and wrap the outputs in tensors.
"""
