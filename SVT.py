def frob_norm(A):
    return np.linalg.norm(A, ord="fro")


def svt_solver(M, mask, tol=1e-4, delta=None, tau=None, n_iters=500):
    """
    The Singular Value Thresholding solver.
    The primary paper is https://arxiv.org/pdf/0810.3286.pdf
    """
    # Kiểm tra nếu không có vùng quan sát
    if np.sum(mask) == 0:
        raise ValueError("Mask không chứa vị trí quan sát nào (toàn bộ là NaN)")

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
            print(f'err :{err}')
            break

    return X
