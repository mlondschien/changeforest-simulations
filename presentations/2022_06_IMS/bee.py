from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from changeforest import Control, changeforest

dataset_id = 2

btf = Path("applications") / "data" / f"sequence{dataset_id}" / "btf"

ximage = np.loadtxt(btf / "ximage.btf")
yimage = np.loadtxt(btf / "yimage.btf")
timage = np.loadtxt(btf / "timage.btf")
labels = np.loadtxt(btf / "label0.btf", dtype="str")

start = 220
stop = 400

X = np.stack([ximage, yimage, timage], axis=1)

dX = X[1:, :] - X[:-1, :]
# timage is the angle. Limit its delta to lie somewhere in [-pi/2, pi/2].
dX[:, 2] = np.fmod(dX[:, 2] + 7 * np.pi / 2, np.pi) - np.pi / 2

dX = dX[start:stop, :]


# Plot 1: Plot raw features
fig, axes = plt.subplots(nrows=dX.shape[1])
for idx in range(len(axes)):
    axes[idx].plot(dX[:, idx])
    ymin, ymax = axes[idx].get_ylim()

plt.tight_layout()
plt.savefig("applications/bee_time_series.png")


# Plot 2: BinarySegmentationResult
result = changeforest(
    dX, "random_forest", "bs", Control(minimal_relative_segment_length=0.1)
)
print(result)
result.plot()
plt.tight_layout()
plt.savefig("applications/bee_binary_segmentation_result.png")
