import onnx
import sys
import onnxruntime as ort
import numpy as np

model = onnx.load(sys.argv[1])

print("Checking model..")
onnx.checker.check_model(model)
print("Checking model complete!")


print("Running inference...")
kpts0 = np.load("kpts0.npy")
desc0 = np.load("desc0.npy")

kpts1 = np.load("kpts1.npy")
desc1 = np.load("desc1.npy")

# max_num_keypoints = 2048
# kpts0 = np.pad(kpts0, ((0, 0), (0, max_num_keypoints - kpts0.shape[1]), (0, 0)), mode='reflect')
# kpts1 = np.pad(kpts1, ((0, 0), (0, max_num_keypoints - kpts1.shape[1]), (0, 0)), mode='reflect')
# desc0 = np.pad(desc0, ((0, 0), (0, max_num_keypoints - desc0.shape[1]), (0, 0)), mode='reflect')
# desc1 = np.pad(desc1, ((0, 0), (0, max_num_keypoints - desc1.shape[1]), (0, 0)), mode='reflect')

# print(kpts0[0][:10])
# print(kpts0[0][-10:])

ort_sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
matches, scores = ort_sess.run(None, {'kpts0': kpts0, 'kpts1': kpts1, 'desc0': desc0, 'desc1': desc1})

# Print Result
print(matches)
print(scores)

print(f"Found {len(matches)} matches")
