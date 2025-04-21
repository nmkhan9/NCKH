import numpy as np
import cv2
import matplotlib.pyplot as plt


# 1️⃣ Load images
def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không thể tải file: {filename}")
    return img.astype(np.float32) / 255


img = load_image('pixel.jpg')
img_missing = load_image('pixel_missing.jpg')

# 2️⃣ Create mask
mask = img_missing < 0.99  # Known pixels


# 3️⃣ Initialize missing values
def initialize_missing(img_missing, mask):
    img_init = img_missing.copy()
    for i in range(img_init.shape[0]):
        row_mask = mask[i, :]
        if np.any(row_mask) and np.sum(row_mask) > 1:  # Need at least 2 known points
            known_idx = np.where(row_mask)[0]
            unknown_idx = np.where(~row_mask)[0]
            img_init[i, unknown_idx] = np.interp(unknown_idx, known_idx, img_init[i, known_idx])
    return img_init


img_init = initialize_missing(img_missing, mask)


# 4️⃣ Nuclear norm subgradient
def nuclear_norm_subgradient(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.where(S > 1e-10, S, 0)  # Loại bỏ singular values quá nhỏ
    return U @ Vt

# 5️⃣ Subgradient Descent
def subgradient_descent_nuclear(y, mask, lambd=0.05, alpha=0.1, max_iter=500, tol=1e-5):
    x = y.copy()
    best_x = x.copy()
    best_loss = float('inf')
    for i in range(1, max_iter + 1):
        current_alpha = alpha / np.sqrt(i)

        grad_f = np.zeros_like(x)
        grad_f[mask] = x[mask] - y[mask]

        grad_nuclear = nuclear_norm_subgradient(x)

        x -= current_alpha * (grad_f + lambd * grad_nuclear)
        x = np.clip(x, 0, 1)

        # Compute loss (only for monitoring)
        data_fidelity = 0.5 * np.sum((x[mask] - y[mask]) ** 2)
        singular_values = np.linalg.svd(x, compute_uv=False)
        nuclear_norm = lambd * np.sum(singular_values)
        current_loss = data_fidelity + nuclear_norm

        if current_loss < best_loss:
            best_loss = current_loss
            best_x = x.copy()

        if i % 100 == 0:
            print(f"Iter {i}: Loss={current_loss:.4f}, Nuclear Norm={nuclear_norm / lambd:.4f}")

        if current_loss < tol:
            print(f"Converged at iteration {i}")
            break

    return best_x


# 6️⃣ Image recovery
img_recovered = subgradient_descent_nuclear(
    img_init,
    mask,
    lambd=0.05,
    alpha=0.5,
    max_iter=1000,
    tol=1e-5
)

# 7️⃣ Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Missing image
img_missing_disp = img_missing.copy()
img_missing_disp[~mask] = 1.0
axes[0].imshow(img_missing_disp, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Damaged Image")

# Recovered image
axes[1].imshow(img_recovered, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Recovered Image")

# Original image
axes[2].imshow(img, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Original Image")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()