"""
model_numpy.py
--------------
A small convolutional neural network implemented from scratch in NumPy
(no PyTorch / TensorFlow available in this environment). It is a genuine
CNN with real forward + backward passes and gradient descent -- not a
placeholder.

Architecture (input: 64x64x3):
  Conv(3->8, 3x3) -> ReLU -> MaxPool(2)      -> 32x32x8
  Conv(8->16, 3x3) -> ReLU -> MaxPool(2)     -> 16x16x16
  Flatten -> Dense(256) -> ReLU -> Dense(9)

Output vector (9 numbers), all normalized to [0,1] where relevant:
  [0]      objectness   (probability an animal is present)          -> sigmoid
  [1:5]    bbox (xmin, ymin, xmax, ymax) normalized by image size    -> sigmoid
  [5:9]    class logits for {dog, cat, bird, rabbit}                 -> softmax

This is intentionally simple so the math is easy to follow and to
actually run in reasonable time on CPU without any deep learning
framework.
"""

import numpy as np


class Adam:
    """Minimal Adam optimizer state for a single parameter array."""

    def __init__(self, shape, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def step(self, param, grad, lr):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return param - lr * m_hat / (np.sqrt(v_hat) + self.eps)


def im2col(x, kh, kw, stride=1, pad=1):
    """x: (N, H, W, C) -> columns for convolution."""
    n, h, w, c = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    cols = np.zeros((n, oh, ow, kh, kw, c), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, i, j, :] = xp[:, i:i + oh * stride:stride, j:j + ow * stride:stride, :]
    return cols.reshape(n, oh, ow, kh * kw * c), oh, ow


class Conv2D:
    def __init__(self, in_c, out_c, k=3, stride=1, pad=1):
        limit = np.sqrt(2.0 / (k * k * in_c))
        self.W = np.random.randn(k * k * in_c, out_c) * limit
        self.b = np.zeros(out_c)
        self.k, self.stride, self.pad = k, stride, pad
        self.in_c, self.out_c = in_c, out_c
        self.opt_W = Adam(self.W.shape)
        self.opt_b = Adam(self.b.shape)

    def forward(self, x):
        self.x_shape = x.shape
        cols, oh, ow = im2col(x, self.k, self.k, self.stride, self.pad)
        self.cols = cols
        self.oh, self.ow = oh, ow
        n = x.shape[0]
        out = cols.reshape(n * oh * ow, -1) @ self.W + self.b
        return out.reshape(n, oh, ow, self.out_c)

    def backward(self, dout, lr):
        n = dout.shape[0]
        dout_flat = dout.reshape(n * self.oh * self.ow, self.out_c)
        cols_flat = self.cols.reshape(n * self.oh * self.ow, -1)

        dW = cols_flat.T @ dout_flat
        db = dout_flat.sum(axis=0)
        dcols_flat = dout_flat @ self.W.T
        dcols = dcols_flat.reshape(n, self.oh, self.ow, self.k, self.k, self.in_c)

        # scatter-add back into padded input
        h, w = self.x_shape[1], self.x_shape[2]
        dxp = np.zeros((n, h + 2 * self.pad, w + 2 * self.pad, self.in_c))
        k, stride = self.k, self.stride
        for i in range(k):
            for j in range(k):
                dxp[:, i:i + self.oh * stride:stride, j:j + self.ow * stride:stride, :] += dcols[:, :, :, i, j, :]
        dx = dxp[:, self.pad:self.pad + h, self.pad:self.pad + w, :]

        self.W = self.opt_W.step(self.W, dW / n, lr)
        self.b = self.opt_b.step(self.b, db / n, lr)
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr=None):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        n, h, w, c = x.shape
        s = self.size
        x_r = x.reshape(n, h // s, s, w // s, s, c)
        self.x_r_shape = x_r.shape
        out = x_r.max(axis=(2, 4))
        # store argmax mask for backward
        out_expand = out[:, :, None, :, None, :]
        self.mask = (x_r == out_expand)
        return out

    def backward(self, dout, lr=None):
        n, hh, s1, ww, s2, c = self.x_r_shape
        d_expand = dout[:, :, None, :, None, :] * self.mask
        # avoid double counting if multiple maxes: normalize
        counts = self.mask.sum(axis=(2, 4), keepdims=True)
        counts[counts == 0] = 1
        d_expand = d_expand / counts
        return d_expand.reshape(n, hh * s1, ww * s2, c)


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr=None):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, in_f, out_f):
        limit = np.sqrt(2.0 / in_f)
        self.W = np.random.randn(in_f, out_f) * limit
        self.b = np.zeros(out_f)
        self.opt_W = Adam(self.W.shape)
        self.opt_b = Adam(self.b.shape)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        n = dout.shape[0]
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        self.W = self.opt_W.step(self.W, dW / n, lr)
        self.b = self.opt_b.step(self.b, db / n, lr)
        return dx


class AnimalDetectorCNN:
    """Full network + task-specific loss (objectness + bbox + class)."""

    def __init__(self, img_size=64, n_classes=4):
        self.img_size = img_size
        self.n_classes = n_classes
        self.conv1 = Conv2D(3, 8, k=3, pad=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(8, 16, k=3, pad=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)
        self.flatten = Flatten()
        flat_dim = (img_size // 4) * (img_size // 4) * 16
        self.fc1 = Dense(flat_dim, 256)
        self.relu3 = ReLU()
        self.fc2 = Dense(256, 1 + 4 + n_classes)  # objectness + bbox + class logits

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        return out  # raw logits, shape (N, 9)

    def predict(self, x):
        raw = self.forward(x)
        obj = 1 / (1 + np.exp(-raw[:, 0]))
        box = 1 / (1 + np.exp(-raw[:, 1:5]))
        logits = raw[:, 5:]
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        cls_probs = ex / ex.sum(axis=1, keepdims=True)
        return obj, box, cls_probs

    def backward(self, draw_out, lr):
        d = self.fc2.backward(draw_out, lr)
        d = self.relu3.backward(d)
        d = self.fc1.backward(d, lr)
        d = self.flatten.backward(d)
        d = self.pool2.backward(d)
        d = self.relu2.backward(d)
        d = self.conv2.backward(d, lr)
        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d, lr)
        return d

    def save(self, path):
        np.savez(path,
                 c1W=self.conv1.W, c1b=self.conv1.b,
                 c2W=self.conv2.W, c2b=self.conv2.b,
                 f1W=self.fc1.W, f1b=self.fc1.b,
                 f2W=self.fc2.W, f2b=self.fc2.b,
                 img_size=self.img_size, n_classes=self.n_classes)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(img_size=int(d["img_size"]), n_classes=int(d["n_classes"]))
        model.conv1.W, model.conv1.b = d["c1W"], d["c1b"]
        model.conv2.W, model.conv2.b = d["c2W"], d["c2b"]
        model.fc1.W, model.fc1.b = d["f1W"], d["f1b"]
        model.fc2.W, model.fc2.b = d["f2W"], d["f2b"]
        return model
