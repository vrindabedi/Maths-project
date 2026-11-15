import io
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import map_coordinates

# ==============================================================================
# 1. DICOM (.dcm) LOADER & TABLE 1 BENCHMARK PHANTOM
# ==============================================================================

def load_dicom_mu(file_bytes: bytes, slice_idx: int = 0, mu_water: float = 0.19):
    """
    Loads a .dcm file, converts pixel values -> Hounsfield Units (HU) ->
    linear attenuation coefficient mu(x, y) in cm^-1, and maps to physical
    box [-D/2, D/2]^2 in cm.

    Note: With mu_water = 0.19 cm^-1:
      - Air (-1000 HU) -> 0.00 cm^-1
      - Brain (~+50 HU) -> ~0.20 cm^-1
      - +50 HU jump -> +0.0095 cm^-1 (Clear tumor in Table 1)
      - +10 HU jump -> +0.0019 cm^-1 (Faint tumor in Table 1)
    """
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    arr = ds.pixel_array.astype(np.float64)

    # Handle multi-frame / 3D DICOM
    num_slices = 1
    if arr.ndim == 3:
        num_slices = arr.shape[0]
        slice_idx = int(np.clip(slice_idx, 0, num_slices - 1))
        arr = arr[slice_idx]

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept

    # Convert HU to linear attenuation mu (cm^-1)
    mu_img = np.maximum(0.0, mu_water * (1.0 + hu / 1000.0))

    # Physical FOV D in cm from PixelSpacing (mm -> cm), default 30 cm
    rows, cols = mu_img.shape
    if hasattr(ds, "PixelSpacing") and len(ds.PixelSpacing) >= 2:
        dy_cm = float(ds.PixelSpacing[0]) / 10.0
        dx_cm = float(ds.PixelSpacing[1]) / 10.0
        D = max(rows * dy_cm, cols * dx_cm)
    else:
        D = 30.0

    # Pad to square if non-square
    if rows != cols:
        n = max(rows, cols)
        square = np.zeros((n, n), dtype=np.float64)
        r0 = (n - rows) // 2
        c0 = (n - cols) // 2
        square[r0:r0 + rows, c0:c0 + cols] = mu_img
        mu_img = square

    return mu_img, D, num_slices, hu


def build_table1_phantom(grid_n: int = 301, D: float = 30.0):
    """
    Builds the exact 2D reference head slice from Table 1 (Section 6.1):
      - Air: mu = 0
      - Skull: ring r in [8.4, 9.0], mu = 0.45
      - Brain: disc r < 8.4, mu = 0.20
      - Clear tumor: disc r <= 0.5 at (-3, -2), mu = 0.20 + 0.0095
      - Faint tumor: disc r <= 0.5 at ( 3,  2), mu = 0.20 + 0.0019
    """
    coords = np.linspace(-D / 2, D / 2, grid_n)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)

    mu = np.zeros_like(R)
    mu[(R >= 8.4) & (R <= 9.0)] = 0.45
    mu[R < 8.4] = 0.20
    mu[(X + 3.0)**2 + (Y + 2.0)**2 <= 0.5**2] = 0.20 + 0.0095
    mu[(X - 3.0)**2 + (Y - 2.0)**2 <= 0.5**2] = 0.20 + 0.0019
    return mu, X, Y


# ==============================================================================
# 2. RAY GEOMETRY & LINE INTEGRAL MEASUREMENTS (Sections 2.1, 2.2, 4, 6.2)
# ==============================================================================

def clip_ray_to_box(ox, oy, dx, dy, half_D):
    """
    Clips line (ox, oy) + tau * (dx, dy) to [-half_D, half_D]^2.
    Returns (xs, ys, xd, yd, L) or None if the ray misses the box.
    """
    tmin, tmax = -1e9, 1e9
    for o, d in ((ox, dx), (oy, dy)):
        if abs(d) < 1e-12:
            if o < -half_D or o > half_D:
                return None
        else:
            t1 = (-half_D - o) / d
            t2 = (half_D - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin >= tmax:
                return None
    xs, ys = ox + tmin * dx, oy + tmin * dy
    xd, yd = ox + tmax * dx, oy + tmax * dy
    L = np.hypot(xd - xs, yd - ys)
    return xs, ys, xd, yd, L


def generate_rays(n_angles: int, n_rays_per_angle: int, D: float):
    """
    Generates parallel-beam CT rays across [0, pi) clipped to [-D/2, D/2]^2.
    Parameterized as r(t) = S + t(D - S), t in [0, 1] (Eq. 3).
    """
    half_D = D / 2.0
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    offsets = np.linspace(-half_D * 0.98, half_D * 0.98, n_rays_per_angle)

    rays = []
    for theta in angles:
        # Offset normal vector & ray unit direction
        nx, ny = np.cos(theta), np.sin(theta)
        dx, dy = -np.sin(theta), np.cos(theta)
        for s in offsets:
            clipped = clip_ray_to_box(s * nx, s * ny, dx, dy, half_D)
            if clipped is not None:
                rays.append(clipped)

    rays = np.array(rays, dtype=np.float64)
    return rays[:, 0], rays[:, 1], rays[:, 2], rays[:, 3], rays[:, 4]


def measure_attenuation_from_image(mu_img, D, xs, ys, xd, yd, L, n_quad=300):
    """
    Computes Beer-Lambert line integral p = L * int_0^1 mu(r(t)) dt (Eq. 2, 4)
    using bilinear interpolation on the reference slice mu_img.
    """
    H, W = mu_img.shape
    t = np.linspace(0.0, 1.0, n_quad)[None, :]  # (1, Q)
    p = np.zeros(len(L), dtype=np.float64)

    batch = 1000
    for i in range(0, len(L), batch):
        sl = slice(i, i + batch)
        rx = xs[sl, None] + t * (xd[sl, None] - xs[sl, None])
        ry = ys[sl, None] + t * (yd[sl, None] - ys[sl, None])

        # Map physical [-D/2, D/2] -> pixel indices [0, W-1], [0, H-1]
        col = (rx + D / 2.0) / D * (W - 1)
        row = (D / 2.0 - ry) / D * (H - 1)  # y-up convention

        vals = map_coordinates(
            mu_img, [row.ravel(), col.ravel()], order=1, mode="constant", cval=0.0
        ).reshape(rx.shape)
        # Trapezoidal integration on [0, 1]
        p[sl] = L[sl] * np.trapezoid(vals, t[0], axis=1)
    return p


def circle_chord_length(xs, ys, xd, yd, L, cx, cy, r):
    """Exact chord length inside disc (cx, cy, r) for Table 1 analytical rays."""
    dx = xd - xs
    dy = yd - ys
    fx = xs - cx
    fy = ys - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    out = np.zeros_like(L)
    mask = disc > 0
    sqrt_d = np.sqrt(np.maximum(disc[mask], 0.0))
    t1 = np.clip((-b[mask] - sqrt_d) / (2.0 * a[mask]), 0.0, 1.0)
    t2 = np.clip((-b[mask] + sqrt_d) / (2.0 * a[mask]), 0.0, 1.0)
    out[mask] = np.maximum(0.0, t2 - t1) * L[mask]
    return out


def measure_attenuation_table1_exact(xs, ys, xd, yd, L):
    """Exact Eq. (15) p = sum(mu * l) for the Table 1 phantom."""
    l_outer = circle_chord_length(xs, ys, xd, yd, L, 0.0, 0.0, 9.0)
    l_brain = circle_chord_length(xs, ys, xd, yd, L, 0.0, 0.0, 8.4)
    l_skull = l_outer - l_brain
    l_clear = circle_chord_length(xs, ys, xd, yd, L, -3.0, -2.0, 0.5)
    l_faint = circle_chord_length(xs, ys, xd, yd, L, 3.0, 2.0, 0.5)
    return (
        0.45 * l_skull
        + 0.20 * l_brain
        + 0.0095 * l_clear
        + 0.0019 * l_faint
    )


# ==============================================================================
# 3. PLANE-WAVE MODEL & REGULARIZED LEAST SQUARES (Sections 2.3, 2.4, 4, 6.1)
# ==============================================================================

def build_wave_frequencies(N_max: int, D: float):
    """
    Generates unique half-plane wave frequency pairs (a_k, b_k):
      a_k = n_x * (2*pi / D), b_k = n_y * (2*pi / D)
    For N_max = 30, this gives exactly K = 1860 terms (1 + 2K = 3721 unknowns).
    """
    pairs = []
    for ny in range(0, N_max + 1):
        nx_start = 1 if ny == 0 else -N_max
        for nx in range(nx_start, N_max + 1):
            pairs.append((nx, ny))
    pairs = np.array(pairs, dtype=np.float64)
    scale = 2.0 * np.pi / D
    a_k = pairs[:, 0] * scale
    b_k = pairs[:, 1] * scale
    return a_k, b_k


def solve_plane_wave_ct(xs, ys, xd, yd, L, p_meas, a_k, b_k, D: float, lam: float):
    """
    Solves the regularized least-squares normal equations (Eq. 6, 7, 8, 13):
      min_w  sum_m (p_pred - p_meas)^2 + lam * integral |grad(mu)|^2 dA

    Because plane waves are orthogonal on [-D/2, D/2]^2:
      integral |grad(mu)|^2 dA = (D^2 / 2) * sum_k (a_k^2 + b_k^2)(u_k^2 + v_k^2)

    Normal equations:
      (A^T A + lam * (D^2 / 2) * diag(0, a_1^2+b_1^2, a_1^2+b_1^2, ...)) w = A^T p_meas
    Accumulated in batches to keep memory low.
    """
    K = len(a_k)
    n_unknowns = 1 + 2 * K
    M = len(L)

    ATA = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
    ATp = np.zeros(n_unknowns, dtype=np.float64)

    dx = xd - xs
    dy = yd - ys

    batch = 1200
    for i in range(0, M, batch):
        sl = slice(i, i + batch)
        m_b = len(L[sl])

        alpha = xs[sl, None] * a_k[None, :] + ys[sl, None] * b_k[None, :]
        beta = dx[sl, None] * a_k[None, :] + dy[sl, None] * b_k[None, :]

        # Exact ray integrals (Eq. 8 & Section 4), handling beta -> 0 safely
        small = np.abs(beta) < 1e-8
        beta_safe = np.where(small, 1.0, beta)

        C = np.where(
            small,
            L[sl, None] * np.cos(alpha),
            L[sl, None] * (np.sin(alpha + beta) - np.sin(alpha)) / beta_safe,
        )
        S = np.where(
            small,
            L[sl, None] * np.sin(alpha),
            L[sl, None] * (np.cos(alpha) - np.cos(alpha + beta)) / beta_safe,
        )

        A_b = np.empty((m_b, n_unknowns), dtype=np.float64)
        A_b[:, 0] = L[sl]
        A_b[:, 1::2] = C
        A_b[:, 2::2] = S

        ATA += A_b.T @ A_b
        ATp += A_b.T @ p_meas[sl]

    # Smoothing penalty diagonal: (D^2 / 2) * (a_k^2 + b_k^2)
    freq_sq = a_k**2 + b_k**2
    reg = np.zeros(n_unknowns, dtype=np.float64)
    reg[1::2] = (D**2 / 2.0) * freq_sq
    reg[2::2] = (D**2 / 2.0) * freq_sq

    # Small ridge epsilon ensures uniqueness even when lam = 0 (Sec 6.3)
    eps = 1e-10
    ATA_reg = ATA + np.diag(lam * reg + eps)
    weights = np.linalg.solve(ATA_reg, ATp)

    w0 = weights[0]
    u_k = weights[1::2]
    v_k = weights[2::2]

    # Amplitude & phase representation (Section 4, page 12)
    w_k = np.sqrt(u_k**2 + v_k**2)
    phi_k = np.arctan2(v_k, u_k)
    return w0, u_k, v_k, w_k, phi_k


# ==============================================================================
# 4. CONTINUOUS EVALUATION, GRADIENT, BOUNDARY, AREA & CURVATURE (Sec 2.5-2.7, 6.7)
# ==============================================================================

def evaluate_surface_and_gradient(X, Y, w0, u_k, v_k, a_k, b_k):
    """
    Evaluates mu(x, y) and analytical partial derivatives dmu/dx, dmu/dy
    using Eq. (5) and Section 4.2.
    """
    pts_x = X.ravel()
    pts_y = Y.ravel()
    N_pts = len(pts_x)

    mu = np.full(N_pts, w0, dtype=np.float64)
    dmu_dx = np.zeros(N_pts, dtype=np.float64)
    dmu_dy = np.zeros(N_pts, dtype=np.float64)

    batch = 4000
    for i in range(0, N_pts, batch):
        sl = slice(i, i + batch)
        phase = pts_x[sl, None] * a_k[None, :] + pts_y[sl, None] * b_k[None, :]
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        mu[sl] += cos_p @ u_k + sin_p @ v_k
        # d/dx [u cos + v sin] = (-u sin + v cos) * a_k
        deriv_factor = -sin_p * u_k[None, :] + cos_p * v_k[None, :]
        dmu_dx[sl] = deriv_factor @ a_k
        dmu_dy[sl] = deriv_factor @ b_k

    mu = mu.reshape(X.shape)
    dmu_dx = dmu_dx.reshape(X.shape)
    dmu_dy = dmu_dy.reshape(X.shape)
    grad_mag = np.sqrt(dmu_dx**2 + dmu_dy**2)
    return mu, dmu_dx, dmu_dy, grad_mag


def evaluate_mu_points(x_pts, y_pts, w0, u_k, v_k, a_k, b_k):
    """Evaluates continuous mu(x, y) at arbitrary 1D arrays of (x, y)."""
    phase = x_pts[:, None] * a_k[None, :] + y_pts[:, None] * b_k[None, :]
    return w0 + np.cos(phase) @ u_k + np.sin(phase) @ v_k


def extract_tumor_metrics(
    xc: float,
    yc: float,
    r_max: float,
    w0,
    u_k,
    v_k,
    a_k,
    b_k,
    n_angles: int = 360,
):
    """
    Implements Section 6.7:
      1. Finds the boundary r = R(theta) in 360 directions where mu is halfway
         between tumor center mu(xc, yc) and surrounding brain mu_bg.
      2. Computes Area = 0.5 * int_0^{2pi} R(theta)^2 dtheta (Eq. 16).
      3. Computes Total attenuation = int_0^{2pi} int_0^{R(theta)} mu r dr dtheta (Eq. 12).
      4. Computes Curvature kappa(theta) along the closed curve (Eq. 11).
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    dtheta = 2.0 * np.pi / n_angles

    mu_center = float(evaluate_mu_points(np.array([xc]), np.array([yc]), w0, u_k, v_k, a_k, b_k)[0])

    # Estimate surrounding brain baseline on an outer ring [0.75*r_max, r_max]
    r_bg = np.linspace(0.75 * r_max, r_max, 5)
    bg_x = (xc + r_bg[None, :] * np.cos(thetas)[:, None]).ravel()
    bg_y = (yc + r_bg[None, :] * np.sin(thetas)[:, None]).ravel()
    mu_bg = float(np.median(evaluate_mu_points(bg_x, bg_y, w0, u_k, v_k, a_k, b_k)))

    jump = mu_center - mu_bg
    mu_half = 0.5 * (mu_center + mu_bg)

    # Sample radial profiles along 360 rays
    n_rad = 120
    r_vals = np.linspace(0.0, r_max, n_rad)
    rx = xc + np.cos(thetas)[:, None] * r_vals[None, :]
    ry = yc + np.sin(thetas)[:, None] * r_vals[None, :]
    prof = evaluate_mu_points(rx.ravel(), ry.ravel(), w0, u_k, v_k, a_k, b_k).reshape(n_angles, n_rad)

    R_theta = np.zeros(n_angles, dtype=np.float64)
    sign = 1.0 if jump >= 0 else -1.0
    for j in range(n_angles):
        diff = sign * (prof[j] - mu_half)
        # Find first crossing from positive to <= 0
        crossings = np.where(diff <= 0.0)[0]
        if len(crossings) > 0 and crossings[0] > 0:
            idx = crossings[0]
            d0, d1 = diff[idx - 1], diff[idx]
            frac = d0 / (d0 - d1 + 1e-12)
            R_theta[j] = r_vals[idx - 1] + frac * (r_vals[idx] - r_vals[idx - 1])
        else:
            R_theta[j] = r_max * 0.5

    # Polar Area (Eq. 16)
    area = 0.5 * np.sum(R_theta**2) * dtheta
    mean_radius = float(np.mean(R_theta))

    # Total attenuation inside R (Eq. 12)
    n_in = 40
    t_in = np.linspace(0.0, 1.0, n_in)
    rr = R_theta[:, None] * t_in[None, :]
    in_x = (xc + np.cos(thetas)[:, None] * rr).ravel()
    in_y = (yc + np.sin(thetas)[:, None] * rr).ravel()
    mu_in = evaluate_mu_points(in_x, in_y, w0, u_k, v_k, a_k, b_k).reshape(n_angles, n_in)
    dr_grid = R_theta[:, None] / (n_in - 1)
    radial_int = np.trapezoid(mu_in * rr, dx=1.0, axis=1) * (R_theta / (n_in - 1))
    total_attenuation = float(np.sum(radial_int) * dtheta)

    # Curvature kappa(theta) (Eq. 11) via periodic central differences
    bx = xc + R_theta * np.cos(thetas)
    by = yc + R_theta * np.sin(thetas)
    xp = (np.roll(bx, -1) - np.roll(bx, 1)) / (2.0 * dtheta)
    yp = (np.roll(by, -1) - np.roll(by, 1)) / (2.0 * dtheta)
    xpp = (np.roll(bx, -1) - 2.0 * bx + np.roll(bx, 1)) / (dtheta**2)
    ypp = (np.roll(by, -1) - 2.0 * by + np.roll(by, 1)) / (dtheta**2)
    denom = np.maximum((xp**2 + yp**2) ** 1.5, 1e-12)
    kappa = (xp * ypp - yp * xpp) / denom

    return {
        "thetas": thetas,
        "R_theta": R_theta,
        "bx": bx,
        "by": by,
        "area": float(area),
        "mean_radius": mean_radius,
        "total_attenuation": total_attenuation,
        "kappa": kappa,
        "mean_kappa": float(np.mean(np.abs(kappa))),
        "mu_center": mu_center,
        "mu_bg": mu_bg,
        "jump": float(jump),
    }


# ==============================================================================
# 5. STREAMLIT UI (UPLOAD .dcm OR RUN TABLE 1 BENCHMARK)
# ==============================================================================

st.set_page_config(page_title="Voxel-Free CT Reconstruction", layout="wide")
st.title("Voxel-Free Continuous CT Reconstruction & Calculus Analysis")
st.caption(
    "Implements Beer–Lambert line integrals, plane-wave least squares with "
    "|∇μ|² smoothing, gradient ridges, polar area, curvature κ, and CNR from the report."
)

with st.sidebar:
    st.header("1. Input Source")
    mode = st.radio(
        "Choose CT Slice Input",
        ["Upload DICOM (.dcm)", "Table 1 Benchmark Phantom (PDF Sec 6.1)"],
    )

    uploaded_file = None
    slice_idx = 0
    mu_water = 0.19
    if mode == "Upload DICOM (.dcm)":
        uploaded_file = st.file_uploader("Upload a CT slice (.dcm)", type=["dcm"])
        slice_idx = st.number_input("Slice index (if 3D DICOM)", min_value=0, value=0, step=1)
        mu_water = st.number_input("μ_water (cm⁻¹)", value=0.19, step=0.01, format="%.3f")

    st.header("2. Mathematical Parameters")
    N_max = st.slider("Max frequency N_max (K = 1860 at 30)", 10, 30, 20, step=2)
    n_angles = st.slider("Projection angles", 40, 120, 90, step=10)
    n_rays = st.slider("Rays per angle", 61, 151, 121, step=10)
    lam = st.select_slider(
        "Smoothing λ (Eq. 13)",
        options=[0.0, 0.001, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        value=10.0,
    )
    sigma = st.select_slider(
        "Measurement noise σ on p (Sec 6.2)",
        options=[0.0, 0.002, 0.005, 0.01],
        value=0.005,
    )
    grad_thresh = st.slider("Tiered smoothness |∇μ| threshold (Sec 4.2)", 0.01, 0.50, 0.08, step=0.01)

    st.header("3. Tumor ROI Seed (Sec 6.7)")
    xc = st.number_input("Center x_c (cm)", value=-3.0, step=0.5)
    yc = st.number_input("Center y_c (cm)", value=-2.0, step=0.5)
    r_search = st.number_input("Search radius r_max (cm)", value=1.5, min_value=0.3, step=0.2)

    run_btn = st.button("Run Reconstruction & Calculus Analysis", type="primary")

# Load reference image
if mode == "Upload DICOM (.dcm)":
    if uploaded_file is None:
        st.info("Upload a `.dcm` file in the sidebar (or switch to the Table 1 Benchmark Phantom) to start.")
        st.stop()
    mu_ref, D, num_slices, _ = load_dicom_mu(uploaded_file.read(), slice_idx=slice_idx, mu_water=mu_water)
    st.sidebar.success(f"Loaded DICOM: {mu_ref.shape[0]}×{mu_ref.shape[1]}, FOV D = {D:.2f} cm, Slices = {num_slices}")
else:
    D = 30.0
    mu_ref, _, _ = build_table1_phantom(grid_n=251, D=D)

if not run_btn:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mu_ref, extent=[-D/2, D/2, -D/2, D/2], origin="upper", cmap="gray")
    ax.set_title("Reference μ(x, y) [cm⁻¹]")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    st.pyplot(fig, use_container_width=False)
    st.stop()

# Run full pipeline
with st.spinner("Computing ray integrals, solving plane-wave system, and extracting calculus metrics..."):
    xs, ys, xd, yd, L = generate_rays(n_angles, n_rays, D)

    if mode == "Table 1 Benchmark Phantom (PDF Sec 6.1)":
        p_clean = measure_attenuation_table1_exact(xs, ys, xd, yd, L)
    else:
        p_clean = measure_attenuation_from_image(mu_ref, D, xs, ys, xd, yd, L)

    rng = np.random.default_rng(42)
    p_meas = p_clean + rng.normal(0.0, sigma, size=p_clean.shape)

    a_k, b_k = build_wave_frequencies(N_max, D)
    w0, u_k, v_k, w_k, phi_k = solve_plane_wave_ct(xs, ys, xd, yd, L, p_meas, a_k, b_k, D, lam)

    grid_n = 201
    coords = np.linspace(-D / 2, D / 2, grid_n)
    X, Y = np.meshgrid(coords, coords)
    # y-up for mathematical coordinates
    Y_up = np.flipud(Y)

    mu_rec, dmu_dx, dmu_dy, grad_mag = evaluate_surface_and_gradient(X, Y_up, w0, u_k, v_k, a_k, b_k)

    # Resample mu_ref onto evaluation grid for error & CNR (Eq. 14)
    H, W = mu_ref.shape
    col_idx = (X + D / 2.0) / D * (W - 1)
    row_idx = (D / 2.0 - Y_up) / D * (H - 1)
    mu_ref_eval = map_coordinates(mu_ref, [row_idx.ravel(), col_idx.ravel()], order=1).reshape(X.shape)

    # Background RMS error B (Eq. 14) inside central soft-tissue region (avoiding skull jump)
    R_grid = np.sqrt(X**2 + Y_up**2)
    if mode == "Table 1 Benchmark Phantom (PDF Sec 6.1)":
        roi_bg = (R_grid < 7.0) & (((X + 3)**2 + (Y_up + 2)**2) > 1.0) & (((X - 3)**2 + (Y_up - 2)**2) > 1.0)
    else:
        # Interior tissue where true gradient is small
        gy, gx = np.gradient(mu_ref_eval, D / grid_n)
        roi_bg = (R_grid < 0.35 * D) & (np.hypot(gx, gy) < 0.02) & ((X - xc)**2 + (Y_up - yc)**2 > r_search**2)

    if np.any(roi_bg):
        B_rms = float(np.sqrt(np.mean((mu_rec[roi_bg] - mu_ref_eval[roi_bg])**2)))
    else:
        B_rms = float(np.sqrt(np.mean((mu_rec - mu_ref_eval)**2)))

    tumor = extract_tumor_metrics(xc, yc, r_search, w0, u_k, v_k, a_k, b_k)
    cnr = abs(tumor["jump"]) / (B_rms + 1e-12)

# Display key metrics
st.subheader("Calculus & Reconstruction Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unknowns (1 + 2K)", f"{1 + 2 * len(a_k)}")
c2.metric("Measured Tumor Jump", f"{tumor['jump']:.5f} cm⁻¹")
c3.metric("Background RMS B", f"{B_rms:.5f} cm⁻¹")
c4.metric("CNR (Visible ≥ 3)", f"{cnr:.2f}", delta="Detected" if cnr >= 3 else "Hidden (< 3)")
c5.metric("Polar Area (Eq. 16)", f"{tumor['area']:.4f} cm²")

c6, c7, c8 = st.columns(3)
c6.metric("Mean Radius", f"{tumor['mean_radius']:.4f} cm")
c7.metric("Mean |Curvature κ|", f"{tumor['mean_kappa']:.3f} cm⁻¹")
c8.metric("Total Attenuation ∬μ dA", f"{tumor['total_attenuation']:.4f} cm")

# Visualizations
tab1, tab2, tab3 = st.tabs(["Reconstruction & Edges (∇μ)", "Ray Profile p(x) (Fig. 3)", "Tumor Boundary & Curvature κ(θ)"])

with tab1:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = axes[0].imshow(mu_ref_eval, extent=[-D/2, D/2, -D/2, D/2], cmap="gray", vmin=0, vmax=max(0.45, mu_ref_eval.max()))
    axes[0].set_title("Reference Slice μ_ref(x, y)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(mu_rec, extent=[-D/2, D/2, -D/2, D/2], cmap="gray", vmin=0, vmax=max(0.45, mu_ref_eval.max()))
    axes[1].plot(tumor["bx"], tumor["by"], "r-", lw=1.5, label="Extracted boundary")
    axes[1].set_title(f"Plane-Wave Reconstruction (λ={lam})")
    axes[1].legend(loc="upper right", fontsize=8)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Ridge / tiered smoothness edge visualization (Sec 2.5 & 4.2)
    im2 = axes[2].imshow(grad_mag, extent=[-D/2, D/2, -D/2, D/2], cmap="inferno")
    axes[2].contour(X, Y_up, grad_mag, levels=[grad_thresh], colors="cyan", linewidths=0.8)
    axes[2].set_title(f"Gradient |∇μ| & Tiered Threshold ({grad_thresh})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    # Vertical rays (first n_rays correspond to theta = 0, i.e., vertical rays along y)
    x_ray = xs[:n_rays]
    p_vert = p_meas[:n_rays]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x_ray, p_vert, lw=1.5, color="navy")
    ax.set_title("Vertical Ray Attenuation p vs x (cf. Figure 3a)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("p = -ln(I_out / I_in)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(tumor["bx"], tumor["by"], "r-", lw=2, label="Half-max curve r = R(θ)")
    axes[0].plot(xc, yc, "ko", ms=4, label="Seed center")
    axes[0].set_aspect("equal")
    axes[0].set_title("Extracted Tumor Boundary (Sec 6.7)")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.degrees(tumor["thetas"]), tumor["kappa"], color="purple")
    axes[1].set_title("Boundary Curvature κ(θ) (Eq. 11)")
    axes[1].set_xlabel("Angle θ (degrees)")
    axes[1].set_ylabel("κ (cm⁻¹)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
