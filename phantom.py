import numpy as np

# Lookup tables indexed by tissue label:
# 0 = Background (dummy 1s prevent division by zero in background pixels)
# 1 = White Matter (WM)
# 2 = Gray Matter (GM)
# 3 = Cerebrospinal Fluid (CSF)
# 4 = Lesion
T1_LUT = np.array([1.0, 800.0, 1300.0, 4000.0, 1800.0], dtype=float)
S0_LUT = np.array([1.0, 0.70, 0.85, 1.00, 0.90], dtype=float)

TISSUE_NAMES = {
    0: "Background",
    1: "WM",
    2: "GM",
    3: "CSF",
    4: "Lesion",
}


def inside_ellipse(X, Y, cx, cy, a, b):
    """Return a True/False grid marking points inside an ellipse."""
    return ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0


def create_phantom(size=64):
    """Create a (size, size) integer label map of a synthetic brain slice."""
    # Build two (size, size) coordinate grids spanning [-1, 1] x [-1, 1]
    ruler = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(ruler, ruler, indexing="xy")

    # Start with all pixels set to 0 (Background)
    labels = np.zeros((size, size), dtype=int)

    # Paint outermost regions first; inner regions overwrite outer ones
    # 1. Gray Matter (label 2): large outer oval
    gm = inside_ellipse(X, Y, cx=0.0, cy=0.0, a=0.85, b=0.72)
    labels[gm] = 2

    # 2. White Matter (label 1): slightly smaller oval inside Gray Matter
    wm = inside_ellipse(X, Y, cx=0.0, cy=0.0, a=0.70, b=0.56)
    labels[wm] = 1

    # 3. CSF Ventricles (label 3): two tall ovals near the center
    left_ventricle = inside_ellipse(X, Y, cx=-0.18, cy=0.0, a=0.10, b=0.24)
    right_ventricle = inside_ellipse(X, Y, cx=0.18, cy=0.0, a=0.10, b=0.24)
    labels[left_ventricle | right_ventricle] = 3

    # 4. Lesion (label 4): small oval inside the right side of White Matter
    lesion = inside_ellipse(X, Y, cx=0.42, cy=-0.20, a=0.14, b=0.12)
    labels[lesion] = 4

    return labels


def simulate_phantom_data(labels, t, snr=20):
    """Map tissue labels to true T1/S0 and simulate clean + noisy MRI signals."""
    t = np.asarray(t, dtype=float)
    H, W = labels.shape
    m = len(t)

    # Look up ground-truth parameter maps of shape (H, W)
    T1_true = T1_LUT[labels]
    S0_true = S0_LUT[labels]

    # Reshape to (H, W, 1) and (1, 1, m) so NumPy computes all (H, W, m) values at once
    T1_3d = T1_true.reshape(H, W, 1)
    S0_3d = S0_true.reshape(H, W, 1)
    t_3d = t.reshape(1, 1, m)

    clean = S0_3d * (1.0 - np.exp(-t_3d / T1_3d))

    # Dedicated RNG seed per SNR so adding new SNRs never alters existing datasets
    rng = np.random.default_rng(1000 + int(snr))
    noise_std = S0_3d / float(snr)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(H, W, m))

    # Keep negative noisy values unclipped so noise stays zero-mean
    noisy = clean + noise
    return T1_true, S0_true, clean, noisy


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    labels = create_phantom(size=64)
    counts = np.bincount(labels.ravel())
    print("Pixel counts per class:", counts)
    for class_id, count in enumerate(counts):
        print(f"  {class_id} ({TISSUE_NAMES[class_id]}): {count} pixels")

    # Verify that empirical noise std per tissue matches S0 / SNR
    t_delays = np.linspace(100.0, 3000.0, 10)
    snr_value = 20
    T1_true, S0_true, clean, noisy = simulate_phantom_data(labels, t_delays, snr=snr_value)

    print(f"\nNoise standard deviation check (SNR = {snr_value}):")
    for class_id in range(1, 5):
        tissue_pixels = labels == class_id
        measured_std = np.std((noisy - clean)[tissue_pixels])
        expected_std = S0_LUT[class_id] / snr_value
        print(f"  {TISSUE_NAMES[class_id]:6s}: measured={measured_std:.5f}, expected={expected_std:.5f}")

    plt.figure(figsize=(5, 5))
    plt.imshow(labels, cmap="viridis", origin="upper")
    plt.colorbar(ticks=[0, 1, 2, 3, 4], label="Tissue Class")
    plt.title("Synthetic Brain Phantom (64x64)")
    plt.tight_layout()
    plt.show()