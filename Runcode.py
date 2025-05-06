import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import svd


# 1️⃣ Hàm đọc ảnh dưới dạng ảnh xám (grayscale) và chuẩn hóa về [0, 1]
def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không thể tải file: {filename}")
    return img.astype(np.float32) / 255


# Đọc ảnh gốc và ảnh đã bị làm mất dữ liệu
img = load_image('pixel.jpg')  # Ảnh gốc
img_missing = load_image('pixel_missing.png')  # Ảnh có vùng bị mất

print(f'Kích thước ảnh : {img.shape[0]}x{img.shape[1]}', )
# 2️⃣ Tạo mặt nạ cho vùng bị mất dữ liệu

# Giả sử pixel >= 0.99 là vùng bị "trắng xoá" → coi là bị mất
img_missing[img_missing >= 0.99] = np.nan  # Gán các pixel bị mất thành NaN

# Tạo mặt nạ: False nếu pixel bị mất (NaN), True nếu còn dữ liệu
mask = ~np.isnan(img_missing)
print(f'% ảnh bị mất {round(np.sum(~mask) * 100 / mask.size, 2)}%')


def frob_norm(A):
    return np.linalg.norm(A, ord="fro")


def svt_solver(M, mask, tol=1e-4, delta=None, tau=None, n_iters=10000):
    # Kiểm tra nếu không có vùng quan sát
    if np.sum(mask) == 0:
        return "Mask không chứa vị trí quan sát nào (toàn bộ là NaN)"

    M_filled = np.copy(M)
    M_filled[np.isnan(M_filled)] = 0

    if not delta:
        delta = 1.2 * np.prod(M.shape) / np.sum(mask)

    if not tau:
        tau = np.sqrt(np.prod(M.shape))  # sqrt(m*n)

    # Tính k0
    norm_M = np.linalg.norm(M_filled, ord=2)
    k0 = np.ceil(tau / (delta * norm_M)) if norm_M > 0 else 1

    # Khởi tạo
    X = np.zeros_like(M_filled)
    Y = k0 * delta * mask * M_filled  # Dùng M_filled để khởi tạo Y
    print(f"Initial k0: {k0}")

    for i in range(n_iters):
        # Dùng np.linalg.svd
        u, s, vh = np.linalg.svd(Y, full_matrices=False)
        # Áp dụng ngưỡng tau lên các giá trị đơn
        shrink_s = np.maximum(s - tau, 0)
        r = np.count_nonzero(shrink_s)
        X = u @ np.diag(shrink_s) @ vh
        Y += delta * mask * (M_filled - X)  # Dùng M_filled trong cập nhật
        # Tiêu chí dừng: chỉ tính sai số trên các vị trí quan sát
        # Dùng M_filled thay vì M để tránh NaN
        err = frob_norm(mask * (X - M_filled)) / frob_norm(mask * M_filled)
        if err <= tol:
            print(f"Hội tụ lại vòng lặp {i + 1} với err = {round(err, 6)}")
            break
    else:
        print(f"Vòng lặp tối đa ({n_iters}) với err = {round(err, 6)}")
    return X


# 4️⃣ Thực hiện khôi phục ảnh
img_recovered = svt_solver(img_missing, mask)

# 5️⃣ Tính toán sai số MSE (chỉ tính trên vùng bị mất)
rmse = np.sqrt(np.mean((img_missing[mask] - img_recovered[mask]) ** 2))
print('rmse :', round(rmse, 6))

# 6️⃣ Hiển thị ảnh: gồm 3 ảnh (ảnh bị mất, ảnh khôi phục, ảnh gốc)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ảnh bị mất dữ liệu (ẩn vùng NaN khỏi hiển thị)
axes[0].imshow(np.ma.masked_array(img_missing), cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Ảnh bị mất", fontsize=20)

# Ảnh sau khi khôi phục bằng SVT
axes[1].imshow(img_recovered, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("Ảnh khôi phục (SVT)", fontsize=20)

# Ảnh gốc ban đầu
axes[2].imshow(img, cmap='gray', vmin=0, vmax=1)
axes[2].set_title("Ảnh ban đầu", fontsize=20)

# Tắt trục và hiển thị
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()