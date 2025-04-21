import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1️⃣ Đọc ảnh gốc (ảnh xám)
img = cv2.imread('pixel.jpg', cv2.IMREAD_GRAYSCALE)

img = img.astype(np.float32)

# 2️⃣ Làm mất dữ liệu ngẫu nhiên cùng vùng nhỏ xung quanh
img_missing = img.copy()
h, w = img.shape
num_points = 100  # Số điểm ngẫu nhiên cần làm mất
patch_size = 10  # Kích thước vùng bị mất xung quanh mỗi điểm

for _ in range(num_points):
    x = np.random.randint(patch_size, w - patch_size)  # Chọn điểm ngẫu nhiên
    y = np.random.randint(patch_size, h - patch_size)

    # Làm mất dữ liệu bằng cách đặt pixel về màu trắng (255)
    img_missing[y - patch_size // 2: y + patch_size // 2 + 1,
    x - patch_size // 2: x + patch_size // 2 + 1] = 255

# 3️⃣ Lưu ảnh bị mất dữ liệu
cv2.imwrite("pixel_missing.jpg", img_missing)

# 4️⃣ Hiển thị ảnh gốc và ảnh bị mất dữ liệu
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Ảnh gốc")
axes[0].axis("off")

axes[1].imshow(img_missing, cmap="gray")
axes[1].set_title("Ảnh bị mất dữ liệu")
axes[1].axis("off")

plt.show()