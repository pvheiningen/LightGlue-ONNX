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

# max_num_keypoints = 2048
# kpts0 = np.pad(kpts0, ((0, 0), (0, max_num_keypoints - kpts0.shape[1]), (0, 0)), mode='constant')
# kpts1 = np.pad(kpts1, ((0, 0), (0, max_num_keypoints - kpts1.shape[1]), (0, 0)), mode='constant')
# desc0 = np.pad(desc0, ((0, 0), (0, max_num_keypoints - desc0.shape[1]), (0, 0)), mode='constant')
# desc1 = np.pad(desc1, ((0, 0), (0, max_num_keypoints - desc1.shape[1]), (0, 0)), mode='constant')

ort_sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
scores = ort_sess.run(None, {'kpts0': kpts0, 'kpts1': kpts1, 'desc0': desc0, 'desc1': desc1})

# Print Result
print(scores[0])
print(scores[0].shape)
print(np.argwhere(scores[0][0] > 0.98))

print(f"Found {np.sum(scores[0] > 0.98)} matches")
