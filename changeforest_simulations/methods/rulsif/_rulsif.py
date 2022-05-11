from pathlib import Path


def rulsif(X, minimal_relative_segment_length, **kwargs):
    # This requires octave to be installed and on the PATH
    # (see https://blink1073.github.io/oct2py/source/installation.html)
    # and oct2py (`mamba install -y oct2py`).
    from oct2py import octave

    octave.addpath(str(Path(__file__).parent.absolute()))

    n = 50
    k = int(2 * n * minimal_relative_segment_length)
    alpha = 0.01
    n_folds = 5
    out = octave.change_detection(X.T, n, k, alpha, n_folds)
    return out
