import h5py, numpy as np

arr = {}
def visit(name, obj):
    if isinstance(obj, h5py.Dataset) and "optimizer" not in name:
        arr[name] = obj[()]

with h5py.File("tools/fan_autoencoder_linear.h5", "r") as f:
    f.visititems(visit)

for k, v in arr.items():
    print(k, v.shape)

# shapes are unique: W1 (3,2), B1 (2,), W2 (2,3), B2 (3,)
by_shape = {v.shape: v for v in arr.values()}
np.savez("tools/weights.npz",
         W1=by_shape[(3, 2)], B1=by_shape[(2,)],
         W2=by_shape[(2, 3)], B2=by_shape[(3,)])
print("saved weights.npz")