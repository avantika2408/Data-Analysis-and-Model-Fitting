import numpy as np

def svd_inv(A):
    U, sig, VT = np.linalg.svd(A)
    siginv = np.diag(1 / sig)
    Ainv = VT.T @ siginv @ U.T
    detA = np.prod(sig)  # Determinant from SVD
    return Ainv, detA

def cov(x):
    np.random.seed(100)

    while True:
        A = np.random.rand(len(x), len(x))
        cov = A @ A.T + 1e-6 * np.eye(len(x))  # Ensure positive-definite
        cov = (cov + cov.T) / 2
        try:
            inv_cov, _ = svd_inv(cov)
            break
        except np.linalg.LinAlgError:
            continue

    return cov


# def cov_legit(x, epsilon=1e-4, max_attempts=100):
#     np.random.seed(1)

#     for _ in range(max_attempts):
#         A = np.random.rand(len(x), len(x))
#         cov = A @ A.T

#         # Add small value to diagonal to ensure positive definiteness
#         cov += epsilon * np.eye(len(x))

#         # Force symmetry (important in finite precision)
#         cov = (cov + cov.T) / 2

#         # Legit check: Cholesky decomposition
#         try:
#             _ = np.linalg.cholesky(cov)
#             return cov
#         except np.linalg.LinAlgError:
#             continue

#     raise RuntimeError("Failed to generate SPD covariance matrix.")
