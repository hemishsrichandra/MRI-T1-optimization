import numpy as np

# A background T1 of 1 keeps the division safe. Its S0 of 0 keeps it dark.
T1_LUT = np.array([1.0, 800.0, 1300.0, 4000.0, 1800.0])
S0_LUT = np.array([0.0, 0.70, 0.85, 1.00, 0.90])
TISSUE_NAMES = {
    0: "Background",
    1: "WM",
    2: "GM",
    3: "CSF",
    4: "Lesion",
}


def make_delays(m=10, t_min=100.0, t_max=5000.0):
    """Return geometrically spaced recovery delays in milliseconds."""
    if m < 2 or not np.isfinite([t_min, t_max]).all() or not 0 < t_min < t_max:
        raise ValueError("Use at least two delays and 0 < t_min < t_max.")

    return np.geomspace(t_min, t_max, m)


def inside_ellipse(X, Y, cx, cy, a, b):
    """Mark points inside an ellipse centered at (cx, cy) with radii a and b."""
    return ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0


def create_phantom(size=64):
    """Draw the project phantom as a square grid of tissue labels."""
    if size < 1:
        raise ValueError("size must be positive.")

    axis = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(axis, axis)
    labels = np.zeros((size, size), dtype=int)

    # Paint the big regions first so the smaller ones can replace their pixels.
    grey_matter = inside_ellipse(x, y, 0.0, 0.0, 0.85, 0.72)
    white_matter = inside_ellipse(x, y, 0.0, 0.0, 0.70, 0.56)
    labels[grey_matter] = 2
    labels[white_matter] = 1

    left_ventricle = inside_ellipse(x, y, -0.18, 0.0, 0.10, 0.24)
    right_ventricle = inside_ellipse(x, y, 0.18, 0.0, 0.10, 0.24)
    labels[left_ventricle | right_ventricle] = 3

    lesion = inside_ellipse(x, y, 0.42, -0.20, 0.14, 0.12)
    labels[lesion] = 4
    return labels


def create_labels(size=64):
    """Draw the original slice using pixel-centered coordinates."""
    if size < 1:
        raise ValueError("size must be positive.")

    # Place the pixel centers on a square that runs from -1 to 1.
    pixel_centers = (np.arange(size) + 0.5) * (2 / size) - 1
    x, y = np.meshgrid(pixel_centers, pixel_centers)
    brain_ellipse = (x / 0.78) ** 2 + (y / 0.84) ** 2

    # Fill the outer ellipse with grey matter and the inner part with white matter.
    labels = np.zeros((size, size), dtype=np.uint8)
    labels[brain_ellipse <= 1] = 2
    labels[brain_ellipse <= 0.72] = 1

    ventricle_shape = ((np.abs(x) - 0.16) / 0.10) ** 2 + ((y + 0.03) / 0.25) ** 2
    labels[(labels == 1) & (ventricle_shape <= 1)] = 3

    lesion_shape = ((x - 0.37) / 0.11) ** 2 + ((y - 0.20) / 0.14) ** 2
    labels[(labels == 1) & (lesion_shape <= 1)] = 4

    return labels


def tissue_maps(labels):
    """Look up T1 in milliseconds and relative S0 for each pixel."""
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 4)):
        raise ValueError("labels must contain integers from 0 through 4.")

    return T1_LUT[labels], S0_LUT[labels]


def make_noisy_data(T1_true, S0_true, t, snr, seed):
    """Return signals as (row, column, delay). Use snr=np.inf for clean data."""
    _, noisy = _simulate_signals(T1_true, S0_true, t, snr, seed)
    return noisy


def simulate_phantom_data(labels, t, snr=20):
    """Return the T1 map, S0 map, clean signal, and noisy signal."""
    t1_map, s0_map = tissue_maps(labels)

    # Use the same seed every time an SNR is repeated in the experiment.
    seed = 1000 + int(snr) if np.isfinite(snr) and snr > 0 else 1000
    clean, noisy = _simulate_signals(t1_map, s0_map, t, snr, seed)
    return t1_map, s0_map, clean, noisy


def _simulate_signals(T1_true, S0_true, t, snr, seed):
    T1_true = np.asarray(T1_true, dtype=float)
    S0_true = np.asarray(S0_true, dtype=float)
    delays_ms = np.asarray(t, dtype=float)

    if T1_true.ndim != 2 or T1_true.shape != S0_true.shape:
        raise ValueError("T1_true and S0_true must have matching 2-D shapes.")
    if not np.isfinite(T1_true).all() or np.any(T1_true <= 0):
        raise ValueError("T1_true must be positive and finite.")
    if not np.isfinite(S0_true).all() or np.any(S0_true < 0):
        raise ValueError("S0_true must be nonnegative and finite.")
    if (
        delays_ms.ndim != 1
        or delays_ms.size == 0
        or not np.isfinite(delays_ms).all()
        or np.any(delays_ms < 0)
    ):
        raise ValueError("t must be a nonempty vector of finite, nonnegative delays.")
    if not snr > 0:
        raise ValueError("snr must be positive.")

    # Add a delay axis so each pixel has a signal value at every recovery time.
    pixel_amplitude = S0_true[..., None]
    recovery = -np.expm1(-delays_ms / T1_true[..., None])
    clean_signal = pixel_amplitude * recovery

    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=pixel_amplitude / snr, size=clean_signal.shape)
    return clean_signal, clean_signal + noise


def brain_pixels(labels):
    """Return the row and column indices of pixels with a tissue label."""
    return np.nonzero(np.asarray(labels) > 0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    labels = create_phantom()
    counts = np.bincount(labels.ravel(), minlength=len(TISSUE_NAMES))
    print("Pixel counts")
    for label, name in TISSUE_NAMES.items():
        print(f"  {name:10s} {counts[label]:4d}")

    delays = np.linspace(100.0, 3000.0, 10)
    snr = 20
    _, _, clean, noisy = simulate_phantom_data(labels, delays, snr)
    noise = noisy - clean

    print(f"\nNoise standard deviation at SNR {snr}")
    for label, name in TISSUE_NAMES.items():
        measured = np.std(noise[labels == label])
        expected = S0_LUT[label] / snr
        print(f"  {name:10s} measured {measured:.5f} | expected {expected:.5f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.get_cmap("viridis", len(TISSUE_NAMES))
    image = ax.imshow(labels, cmap=colors, vmin=-0.5, vmax=4.5)
    colorbar = fig.colorbar(image, ax=ax, ticks=list(TISSUE_NAMES))
    colorbar.set_ticklabels(list(TISSUE_NAMES.values()))
    ax.set_title("Synthetic brain phantom")
    fig.tight_layout()
    plt.show()
