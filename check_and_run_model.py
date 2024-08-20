import onnx
import sys
import onnxruntime as ort
import numpy as np

filename = sys.argv[1]
model = onnx.load(filename)

print("Checking model..")
onnx.checker.check_model(model)
print("Checking model complete!")


print("Running inference...")
kpts0 = np.load("/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/kpts0.npy")
desc0 = np.load("/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/desc0.npy")

kpts1 = np.load("/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/kpts1.npy")
desc1 = np.load("/srv/calibrations/GlobalFootball/01_11v11_soccer_amfb_green/desc1.npy")


ort_sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
scores = ort_sess.run(None, {'kpts0': kpts0, 'kpts1': kpts1, 'desc0': desc0, 'desc1': desc1})

# Print Result
print(scores[0])
print(scores[0].shape)
matches = np.argwhere(scores[0][0] > 0.98)
print(matches)

print(f"Found {np.sum(scores[0] > 0.98)} matches")


from lightglue import viz2d
from lightglue_onnx.utils import load_image
import matplotlib.pyplot as plt

# Sample images for tracing
img0_path = sys.argv[2] + "/image0-synchronized.jpg"
img1_path = sys.argv[2] + "/image1-synchronized.jpg"
image0 = load_image(img0_path).permute(1, 2, 0)
image1 = load_image(img1_path).permute(1, 2, 0)


m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]

def unnormalize_keypoints(
    kpts: np.array,
    h: int,
    w: int,
) -> np.array:
    size = np.array([w, h])
    shift = size / 2
    scale = size.max() / 2
    kpts = kpts * scale + shift
    return kpts

m_kpts0 = unnormalize_keypoints(m_kpts0[:, :2], 3046, 410)
m_kpts1 = unnormalize_keypoints(m_kpts1[:, :2], 3046, 410)
m_kpts0[:, 0] += 0.9 * 4104

print(m_kpts0)

axes = viz2d.plot_images([image0, image1], adaptive=False)
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
plt.show()