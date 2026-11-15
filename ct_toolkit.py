"""
ct_toolkit.py  -  load ANY CT-type file, run the Group-14 cosine-model pipeline, analyse tumours
=================================================================================================
Accepted inputs (give files, folders or .zip; mix them freely):

  RAW RAY TABLES   .csv .tsv .txt .xlsx .xlsm .xls     columns = source/detector positions, I_in, I_out, p ...
  IMAGE VOLUMES    .nii .nii.gz   (NIfTI)              CT in Hounsfield units
                   .dcm / no extension / folder / zip  (DICOM CT series, many slices)
                   .npy                                (3-D array, assumed HU)
  SEGMENTATIONS    NIfTI label files (name contains seg/mask/label, or few integer values)
                   DICOM-SEG objects (SOP class 1.2.840.10008.5.1.4.1.1.66.4)

What it does
  rays table  -> cosine-basis least squares (report eq. 5-8, 13) -> reconstructed volume
  image vol.  -> (optional) simulate a scan: forward-project to rays, add noise, then reconstruct
  segmentation-> volume, per-slice area, polar/Green's-theorem area, boundary curvature, CNR

Command line:
  python ct_toolkit.py ct_dataset/ct_raw_rays.csv --seg ct_dataset/seg.nii
  python ct_toolkit.py ct_series_folder/ seg.dcm --simulate --lam 10
  python ct_toolkit.py scan.nii.gz tumour_seg.nii.gz --simulate

NOTE: real scanners do not export raw rays in an open format (vendor-specific projection data).
Clinical exports are reconstructed DICOM/NIfTI images, which is why --simulate exists.
"""
import argparse, glob, gzip, hashlib, json, os, re, struct, sys, tempfile, zipfile
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

MU_WATER = 0.19                         # cm^-1  (so that +0.0095 = +50 HU, as in the report)
SEG_SOP = "1.2.840.10008.5.1.4.1.1.66.4"


# =============================================================== containers
@dataclass
class Volume:
    data: np.ndarray                    # (nx, ny, nz); HU for images, ints for segmentations
    spacing: tuple                      # mm  (sx, sy, sz)
    origin: tuple = (0.0, 0.0, 0.0)     # mm  position of voxel (0,0,0)
    name: str = "volume"
    kind: str = "image"                 # "image" | "seg"
    label_names: dict = field(default_factory=dict)
    source: str = ""
    z_pos: list = None                  # DICOM slice z positions [mm] if known


@dataclass
class Loaded:
    rays: list = field(default_factory=list)       # standardised DataFrames
    volumes: list = field(default_factory=list)
    segs: list = field(default_factory=list)
    log: list = field(default_factory=list)


# ============================================================ NIfTI I/O
_NII_DT = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8,
           512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def read_nifti(path):
    """Returns (data[nx,ny,nz], spacing_mm, origin_mm). Uses nibabel if installed, else a built-in reader."""
    try:
        import nibabel as nib
        img = nib.as_closest_canonical(nib.load(path))
        data = np.asanyarray(img.dataobj)
        if data.ndim > 3:
            data = data[..., 0]
        aff = img.affine
        return data, tuple(np.sqrt((aff[:3, :3] ** 2).sum(0))), tuple(aff[:3, 3])
    except ImportError:
        pass
    raw = (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")).read()
    end = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    if struct.unpack(end + "i", raw[:4])[0] != 348:
        raise ValueError("not a NIfTI-1 file")
    dim = struct.unpack(end + "8h", raw[40:56])
    shape = dim[1:1 + dim[0]]
    dt = struct.unpack(end + "h", raw[70:72])[0]
    pixdim = struct.unpack(end + "8f", raw[76:108])
    off = int(struct.unpack(end + "f", raw[108:112])[0])
    slope, inter = struct.unpack(end + "2f", raw[112:120])
    units = raw[123] & 7
    scale = {1: 1000.0, 2: 1.0, 3: 0.001}.get(units, 1.0)
    sform = struct.unpack(end + "h", raw[254:256])[0]
    srow = struct.unpack(end + "12f", raw[280:328])
    q = struct.unpack(end + "3f", raw[268:280])
    arr = np.frombuffer(raw, np.dtype(_NII_DT[dt]).newbyteorder(end), count=int(np.prod(shape)),
                        offset=off).reshape(shape, order="F")
    arr = arr[..., 0] if arr.ndim > 3 else arr
    if slope not in (0.0, 1.0) or inter != 0.0:
        arr = arr * (slope or 1.0) + inter
    sp = [pixdim[1] * scale, pixdim[2] * scale, pixdim[3] * scale]
    org = [srow[3], srow[7], srow[11]] if sform > 0 else list(q)
    org = [o * scale for o in org]
    if sform > 0:                                   # undo simple axis flips (e.g. LAS)
        for i in range(3):
            if srow[4 * i + i] < 0:
                arr = np.flip(arr, i)
                org[i] = org[i] - (arr.shape[i] - 1) * sp[i]
    return np.array(arr), tuple(sp), tuple(org)


def write_nifti(path, data, spacing_mm, origin_mm=(0, 0, 0), dtype=np.int16):
    data = np.asarray(data)
    nx, ny, nz = data.shape
    h = bytearray(352)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<h", h, 70, {np.int16: 4, np.float32: 16, np.uint8: 2}[dtype])
    struct.pack_into("<h", h, 72, np.dtype(dtype).itemsize * 8)
    struct.pack_into("<8f", h, 76, 1, *spacing_mm, 0, 0, 0, 0)
    struct.pack_into("<f", h, 108, 352.0)
    struct.pack_into("<f", h, 112, 1.0)
    h[123] = 2
    struct.pack_into("<h", h, 254, 1)
    for i in range(3):
        row = [0.0] * 4
        row[i] = float(spacing_mm[i]); row[3] = float(origin_mm[i])
        struct.pack_into("<4f", h, 280 + 16 * i, *row)
    h[344:348] = b"n+1\x00"
    with open(path, "wb") as f:
        f.write(bytes(h))
        f.write(data.astype(np.dtype(dtype).newbyteorder("<")).ravel(order="F").tobytes())


# ============================================================== DICOM I/O
def _dicom_series(paths, log):
    try:
        import pydicom
    except ImportError:
        log.append("DICOM files found but `pydicom` is not installed:  pip install pydicom")
        return [], []
    groups = {}
    for p in paths:
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue
        if "SOPClassUID" not in ds:
            continue
        groups.setdefault(str(getattr(ds, "SeriesInstanceUID", "unknown")), []).append(ds)
    vols, segs = [], []
    for uid, dss in groups.items():
        try:
            if str(dss[0].SOPClassUID) == SEG_SOP:
                for ds in dss:
                    segs.append(_dicom_seg(ds, log))
            elif "PixelData" in dss[0]:
                vols.append(_dicom_image_series(dss, log))
            else:
                log.append(f"skipped DICOM series {uid[-8:]}: no pixel data "
                           f"(vendor raw projection data is not readable here)")
        except Exception as e:                                    # keep going with other series
            log.append(f"could not read DICOM series {uid[-8:]}: {e}")
    return vols, segs


def _slice_sort_key(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None:
        r, c = np.array(iop[:3], float), np.array(iop[3:], float)
        return float(np.dot(np.cross(r, c), np.array(ipp, float)))
    return float(getattr(ds, "InstanceNumber", 0))


def _dicom_image_series(dss, log):
    dss = sorted(dss, key=_slice_sort_key)
    mats = []
    for ds in dss:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        mats.append(ds.pixel_array.astype(np.float32) * slope + inter)
    vol = np.stack(mats, axis=2)                                  # (rows, cols, nz)
    data = vol[::-1, :, :].transpose(1, 0, 2)                     # -> (x=cols, y=up, z)
    ps = [float(v) for v in getattr(dss[0], "PixelSpacing", [1.0, 1.0])]
    zs = [_slice_sort_key(d) for d in dss]
    sz = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else float(getattr(dss[0], "SliceThickness", 1.0))
    mod = str(getattr(dss[0], "Modality", "?"))
    if mod != "CT":
        log.append(f"DICOM modality is {mod}, not CT: voxel values are NOT Hounsfield units")
    ipp = [float(v) for v in getattr(dss[0], "ImagePositionPatient", [0, 0, 0])]
    return Volume(data, (ps[1], ps[0], sz), tuple(ipp), name=f"DICOM {mod} series ({len(dss)} slices)",
                  kind="image", source="dicom", z_pos=[float(getattr(d, "ImagePositionPatient", [0, 0, z])[2]) for d, z in zip(dss, zs)])


def _dicom_seg(ds, log):
    arr = np.asarray(ds.pixel_array)
    if arr.ndim == 2:
        arr = arr[None]
    pf = ds.PerFrameFunctionalGroupsSequence
    zpos, segnum = [], []
    for fg in pf:
        zpos.append(float(fg.PlanePositionSequence[0].ImagePositionPatient[2]))
        segnum.append(int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber))
    uz = sorted(set(round(z, 3) for z in zpos))
    lab = np.zeros((arr.shape[1], arr.shape[2], len(uz)), np.int16)
    for i, (z, s) in enumerate(zip(zpos, segnum)):
        lab[:, :, uz.index(round(z, 3))][arr[i] > 0] = s
    names = {int(sg.SegmentNumber): str(getattr(sg, "SegmentLabel", sg.SegmentNumber))
             for sg in getattr(ds, "SegmentSequence", [])}
    shared = ds.SharedFunctionalGroupsSequence[0]
    pm = shared.PixelMeasuresSequence[0]
    ps = [float(v) for v in pm.PixelSpacing]
    sz = float(np.median(np.diff(uz))) if len(uz) > 1 else float(getattr(pm, "SliceThickness", 1.0))
    ipp = [float(v) for v in pf[0].PlanePositionSequence[0].ImagePositionPatient]
    data = lab[::-1, :, :].transpose(1, 0, 2)
    return Volume(data, (ps[1], ps[0], sz), tuple(ipp), name="DICOM-SEG", kind="seg", label_names=names,
                  source="dicom-seg", z_pos=list(uz))


# ============================================================ table loader
def _norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


ALIAS = {
    "sx": ["source_x_cm", "source_x_mm", "source_x", "src_x", "sx", "xs", "x_source", "focal_x", "tube_x"],
    "sy": ["source_y_cm", "source_y_mm", "source_y", "src_y", "sy", "ys", "y_source", "focal_y", "tube_y"],
    "sz": ["source_z_cm", "source_z_mm", "source_z", "src_z", "sz", "zs", "z_source", "focal_z", "tube_z"],
    "dx": ["detector_x_cm", "detector_x_mm", "detector_x", "det_x", "dx", "xd", "x_detector"],
    "dy": ["detector_y_cm", "detector_y_mm", "detector_y", "det_y", "dy", "yd", "y_detector"],
    "dz": ["detector_z_cm", "detector_z_mm", "detector_z", "det_z", "dz", "zd", "z_detector"],
    "p": ["p_measured", "p", "p_meas", "attenuation", "line_integral", "projection"],
    "i_in": ["i_in_counts", "i_in", "i0", "i_0", "incident_intensity", "incident_counts"],
    "i_out": ["i_out_measured_counts", "i_out", "i_out_counts", "detected_intensity", "detected_counts", "intensity_out"],
    "angle": ["gantry_angle_deg", "gantry_angle", "angle_deg", "angle", "theta"],
    "offset": ["detector_offset_cm", "detector_offset_mm", "detector_offset", "offset", "detector_position", "det_pos"],
    "slice_id": ["slice_id", "slice", "slice_index"],
    "z": ["z_cm", "z_mm", "z", "slice_z", "z_position"],
}


def standardise_table(df, units="cm", gantry_radius=15.5, log=None):
    """Map arbitrary column names to sx,sy,sz,dx,dy,dz,p,slice_id,z (lengths in cm). Returns DataFrame or None."""
    log = log if log is not None else []
    cols = {_norm(c): c for c in df.columns}
    pick, scale = {}, {}
    for key, names in ALIAS.items():
        for n in names:
            if n in cols:
                pick[key] = cols[n]
                scale[key] = 0.1 if (n.endswith("_mm") or (units == "mm" and key in
                                     "sx sy sz dx dy dz offset z".split() and not n.endswith("_cm"))) else 1.0
                break
    out = pd.DataFrame(index=df.index)
    have_geo = all(k in pick for k in ("sx", "sy", "dx", "dy"))
    if have_geo:
        for k in ("sx", "sy", "dx", "dy"):
            out[k] = pd.to_numeric(df[pick[k]], errors="coerce") * scale[k]
        for k in ("sz", "dz"):
            out[k] = pd.to_numeric(df[pick[k]], errors="coerce") * scale[k] if k in pick else 0.0
    elif "angle" in pick and "offset" in pick:
        th = np.deg2rad(pd.to_numeric(df[pick["angle"]], errors="coerce").to_numpy())
        s = pd.to_numeric(df[pick["offset"]], errors="coerce").to_numpy() * scale["offset"]
        h = np.sqrt(np.clip(gantry_radius ** 2 - s ** 2, 1e-6, None))
        u = np.c_[np.cos(th), np.sin(th)]; n = np.c_[-np.sin(th), np.cos(th)]
        S = s[:, None] * n - h[:, None] * u; D = s[:, None] * n + h[:, None] * u
        out["sx"], out["sy"], out["dx"], out["dy"] = S[:, 0], S[:, 1], D[:, 0], D[:, 1]
        out["sz"] = out["dz"] = 0.0
        log.append(f"no source/detector columns: built parallel-beam geometry from angle + offset (radius {gantry_radius} cm)")
    else:
        log.append("table has no ray geometry (need source/detector x,y or gantry angle + detector offset)")
        return None
    if "p" in pick:
        out["p"] = pd.to_numeric(df[pick["p"]], errors="coerce")
    elif "i_in" in pick and "i_out" in pick:
        i_in = pd.to_numeric(df[pick["i_in"]], errors="coerce")
        i_out = pd.to_numeric(df[pick["i_out"]], errors="coerce").clip(lower=0.5)
        out["p"] = -np.log(i_out / i_in)
        log.append("p computed as -ln(I_out / I_in)")
    else:
        log.append("table has no p, and no I_in / I_out pair")
        return None
    if "slice_id" in pick:
        out["slice_id"] = pd.factorize(df[pick["slice_id"]], sort=True)[0]
    else:
        out["slice_id"] = pd.factorize(out["sz"].round(3), sort=True)[0]
    out["z"] = (pd.to_numeric(df[pick["z"]], errors="coerce") * scale["z"]) if "z" in pick else out["sz"]
    out = out.dropna(subset=["sx", "sy", "dx", "dy", "p"]).reset_index(drop=True)
    ext = float(np.abs(out[["sx", "sy", "dx", "dy"]].to_numpy()).max())
    log.append(f"ray table OK: {len(out)} rays, {out.slice_id.nunique()} slice(s), largest |coordinate| = {ext:.1f} cm")
    return out


def read_table(path, units="cm", log=None):
    ext = os.path.splitext(path)[1].lower()
    log = log if log is not None else []
    if ext in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        for name, df in sheets.items():
            std = standardise_table(df, units, log=log)
            if std is not None:
                log.append(f"used sheet '{name}'")
                return std
        return None
    df = pd.read_csv(path, sep=None, engine="python") if ext in (".tsv", ".txt") else pd.read_csv(path)
    return standardise_table(df, units, log=log)


# ============================================================ any-file loader
def _expand(paths, tmp):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += [f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True) if os.path.isfile(f)]
        elif p.lower().endswith(".zip"):
            d = tempfile.mkdtemp(dir=tmp)
            with zipfile.ZipFile(p) as z:
                z.extractall(d)
            out += _expand([d], tmp)
        else:
            out.append(p)
    return out


def _looks_like_seg(name, data):
    if re.search(r"seg|mask|label|roi|tumou?r", os.path.basename(name).lower()):
        return True
    vals = np.unique(data)
    return len(vals) <= 12 and np.allclose(vals, np.round(vals)) and vals.min() >= 0 and vals.max() <= 255 and 0 in vals


def load_any(paths, units="cm", gantry_radius=15.5, npy_spacing_mm=1.0, force_seg=()):
    """Load files/folders/zips of any supported kind -> Loaded."""
    L = Loaded()
    tmp = tempfile.mkdtemp()
    files = _expand(list(paths), tmp)
    dicom_candidates = []
    for f in files:
        low = f.lower()
        try:
            if low.endswith((".nii", ".nii.gz")):
                data, sp, org = read_nifti(f)
                is_seg = f in force_seg or _looks_like_seg(f, data)
                v = Volume(np.rint(data).astype(np.int16) if is_seg else data.astype(np.float32), sp, org,
                           name=os.path.basename(f), kind="seg" if is_seg else "image", source="nifti")
                (L.segs if is_seg else L.volumes).append(v)
                L.log.append(f"{os.path.basename(f)}: NIfTI {'segmentation' if is_seg else 'image'} {data.shape}, "
                             f"voxel {sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.2f} mm")
            elif low.endswith((".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls")):
                sub = []
                t = read_table(f, units, sub)
                L.log += [f"{os.path.basename(f)}: {m}" for m in sub]
                if t is not None:
                    L.rays.append(t)
            elif low.endswith(".npy"):
                a = np.load(f)
                if a.ndim == 2:
                    a = a[:, :, None]
                is_seg = _looks_like_seg(f, a)
                v = Volume(a.astype(np.int16 if is_seg else np.float32), (npy_spacing_mm,) * 3, name=os.path.basename(f),
                           kind="seg" if is_seg else "image", source="npy")
                (L.segs if is_seg else L.volumes).append(v)
                L.log.append(f"{os.path.basename(f)}: numpy array {a.shape}, spacing assumed {npy_spacing_mm} mm")
            elif low.endswith((".json", ".png", ".jpg", ".md", ".py", ".vox")):
                continue
            else:
                dicom_candidates.append(f)
        except Exception as e:
            L.log.append(f"{os.path.basename(f)}: could not read ({e})")
    if dicom_candidates:
        v, s = _dicom_series(dicom_candidates, L.log)
        L.volumes += v
        L.segs += s
        L.log.append(f"DICOM: {len(v)} image series, {len(s)} segmentation object(s) from {len(dicom_candidates)} file(s)")
    return L


# ======================================================= cosine-model maths
def hu_to_mu(hu):
    return np.clip(MU_WATER * (1.0 + np.asarray(hu, float) / 1000.0), 0.0, None)


def mu_to_hu(mu):
    return 1000.0 * (np.asarray(mu, float) - MU_WATER) / MU_WATER


def frequencies(nmax, box):
    pairs = [(nx, ny) for nx in range(0, nmax + 1) for ny in range(-nmax, nmax + 1) if nx > 0 or ny > 0]
    n = np.array(pairs, float)
    return 2 * np.pi * n[:, 0] / box, 2 * np.pi * n[:, 1] / box


def design_rows(S, D, a, b):
    """[L | L*C_k | L*S_k]: exact line integral of each cos/sin plane wave along each ray (report eq. 8)."""
    d = D - S
    L = np.hypot(d[:, 0], d[:, 1])
    alpha = np.outer(S[:, 0], a) + np.outer(S[:, 1], b)
    beta = np.outer(d[:, 0], a) + np.outer(d[:, 1], b)
    small = np.abs(beta) < 1e-9
    bs = np.where(small, 1.0, beta)
    C = np.where(small, np.cos(alpha), (np.sin(alpha + beta) - np.sin(alpha)) / bs)
    Sn = np.where(small, np.sin(alpha), (np.cos(alpha) - np.cos(alpha + beta)) / bs)
    return np.hstack([L[:, None], L[:, None] * C, L[:, None] * Sn])


def evaluate_mu(w, a, b, X, Y, chunk=1000):
    K = len(a)
    xf, yf = X.ravel(), Y.ravel()
    out = np.empty(xf.size)
    for i in range(0, xf.size, chunk):
        ph = np.outer(xf[i:i + chunk], a) + np.outer(yf[i:i + chunk], b)
        out[i:i + chunk] = w[0] + np.cos(ph) @ w[1:1 + K] + np.sin(ph) @ w[1 + K:]
    return out.reshape(X.shape)


def reconstruct(rays, X, Y, nmax=30, lam=10.0, box=30.0, log=print, chunk=1500):
    """Smoothed least squares  min sum (p_pred-p)^2 + lam*int|grad mu|^2 ; returns mu stack (nx,ny,ns), z list [cm]."""
    from scipy.linalg import cho_factor, cho_solve
    a, b = frequencies(nmax, box)
    K = len(a)
    pen = np.concatenate([[0.0], np.tile(lam * (a ** 2 + b ** 2) * box ** 2 / 2, 2)])
    groups = {}
    for sid, sub in rays.groupby("slice_id"):
        key = hashlib.md5(np.round(sub[["sx", "sy", "dx", "dy"]].to_numpy(), 4).tobytes()).hexdigest()
        groups.setdefault(key, []).append(sid)
    mus, zs = {}, {}
    for key, sids in groups.items():
        subs = [rays[rays.slice_id == s] for s in sids]
        S = subs[0][["sx", "sy"]].to_numpy(); D = subs[0][["dx", "dy"]].to_numpy()
        P = np.stack([s["p"].to_numpy() for s in subs], 1)
        AtA = np.zeros((1 + 2 * K, 1 + 2 * K)); AtP = np.zeros((1 + 2 * K, len(sids)))
        for i in range(0, len(S), chunk):
            A = design_rows(S[i:i + chunk], D[i:i + chunk], a, b)
            AtA += A.T @ A
            AtP += A.T @ P[i:i + chunk]
        AtA[np.diag_indices_from(AtA)] += pen + 1e-10
        try:
            W = cho_solve(cho_factor(AtA), AtP)
        except Exception:
            W = np.linalg.lstsq(AtA, AtP, rcond=None)[0]
        log(f"  solved {len(sids)} slice(s) sharing one geometry ({len(S)} rays, {1 + 2 * K} unknowns)")
        for j, sid in enumerate(sids):
            mus[sid] = evaluate_mu(W[:, j], a, b, X, Y)
            zs[sid] = float(subs[j]["z"].mean())
    order = sorted(mus, key=lambda s: zs[s])
    return np.stack([mus[s] for s in order], 2), [zs[s] for s in order]


# ===================================================== forward projection
def parallel_geometry(box=30.0, n_angles=120, n_rays=151, arc=180.0, radius=None):
    radius = radius or box / 2 + 0.5
    th = np.deg2rad(np.arange(n_angles) * arc / n_angles)
    s = (np.arange(n_rays) - (n_rays - 1) / 2) * (box / (n_rays - 1))
    th, s = np.meshgrid(th, s, indexing="ij")
    th, s = th.ravel(), s.ravel()
    u = np.c_[np.cos(th), np.sin(th)]; n = np.c_[-np.sin(th), np.cos(th)]
    h = np.sqrt(radius ** 2 - s ** 2)
    return s[:, None] * n - h[:, None] * u, s[:, None] * n + h[:, None] * u, th, s


def forward_project(mu2d, sp_cm, S, D, ds=0.05, chunk=1500):
    """Line integral of a 2-D mu image (image centre at the isocentre) along rays S->D."""
    from scipy.ndimage import map_coordinates
    nx, ny = mu2d.shape
    L = np.hypot(*(D - S).T)
    ns = int(np.ceil(L.max() / ds))
    t = (np.arange(ns) + 0.5) / ns
    p = np.zeros(len(S))
    for i in range(0, len(S), chunk):
        s, d = S[i:i + chunk], D[i:i + chunk]
        x = s[:, 0:1] + t[None] * (d[:, 0:1] - s[:, 0:1])
        y = s[:, 1:2] + t[None] * (d[:, 1:2] - s[:, 1:2])
        ci = x / sp_cm[0] + (nx - 1) / 2
        cj = y / sp_cm[1] + (ny - 1) / 2
        vals = map_coordinates(mu2d, [ci.ravel(), cj.ravel()], order=1, mode="constant", cval=0.0).reshape(x.shape)
        p[i:i + chunk] = vals.mean(1) * L[i:i + chunk]
    return p


def simulate_scan(mu_slices, sp_cm, z_cm, box=30.0, n_angles=120, n_rays=151, sigma=0.005, i_in=1e7, seed=1):
    rng = np.random.default_rng(seed)
    S, D, th, s = parallel_geometry(box, n_angles, n_rays)
    frames = []
    for k, (mu, z) in enumerate(zip(mu_slices, z_cm)):
        p_true = forward_project(mu, sp_cm, S, D)
        p_meas = p_true + rng.normal(0, sigma, p_true.shape)
        i_out = np.rint(i_in * np.exp(-p_meas))
        frames.append(pd.DataFrame(dict(
            slice_id=k, z_cm=z, gantry_angle_deg=np.rad2deg(th), detector_offset_cm=s,
            source_x_cm=S[:, 0], source_y_cm=S[:, 1], source_z_cm=z, detector_x_cm=D[:, 0], detector_y_cm=D[:, 1],
            detector_z_cm=z, path_length_L_cm=np.hypot(*(D - S).T), I_in_counts=i_in,
            I_out_measured_counts=i_out, p_true=p_true, p_measured=-np.log(i_out / i_in))))
    return pd.concat(frames, ignore_index=True)


# ================================================== segmentation analysis
def contour_stats(mask2d, sx, sy, n=360):
    """Boundary r=R(theta)-style analysis: Green's-theorem area, perimeter, curvature kappa (report eq. 11)."""
    from skimage import measure
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    sm = np.pad(gaussian_filter(mask2d.astype(float), 0.7), 1)
    cs = measure.find_contours(sm, 0.5)
    if not cs:
        return {}
    c = max(cs, key=len) - 1
    pts = c * np.array([sx, sy])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    d = np.r_[0, np.cumsum(seg)]
    per = d[-1]
    s = np.linspace(0, per, n, endpoint=False)
    x = gaussian_filter1d(np.interp(s, d, pts[:, 0]), 3, mode="wrap")
    y = gaussian_filter1d(np.interp(s, d, pts[:, 1]), 3, mode="wrap")
    h = per / n
    xp = (np.roll(x, -1) - np.roll(x, 1)) / (2 * h); yp = (np.roll(y, -1) - np.roll(y, 1)) / (2 * h)
    xpp = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / h ** 2; ypp = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / h ** 2
    kappa = (xp * ypp - yp * xpp) / (xp ** 2 + yp ** 2) ** 1.5
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    cx, cy = x.mean(), y.mean()
    R = np.hypot(x - cx, y - cy)
    return dict(green_area_mm2=area, perimeter_mm=per, kappa_mean_abs=float(np.abs(kappa).mean()),
                kappa_max_abs=float(np.abs(kappa).max()), R_mean_mm=float(R.mean()), R_min_mm=float(R.min()),
                R_max_mm=float(R.max()))


def analyze_segmentation(seg, img=None):
    rows = []
    sx, sy, sz = seg.spacing
    vox = sx * sy * sz
    for lab in [int(v) for v in np.unique(seg.data) if v != 0]:
        m = seg.data == lab
        idx = np.argwhere(m)
        areas = m.sum((0, 1)) * sx * sy
        k = int(np.argmax(areas))
        row = dict(label=lab, name=seg.label_names.get(lab, f"label {lab}"), voxels=int(m.sum()),
                   volume_cm3=m.sum() * vox / 1000.0, slices=int((areas > 0).sum()), largest_slice=k,
                   max_slice_area_mm2=float(areas[k]), eq_radius_mm=float(np.sqrt(areas[k] / np.pi)),
                   centroid_x_mm=seg.origin[0] + idx[:, 0].mean() * sx, centroid_y_mm=seg.origin[1] + idx[:, 1].mean() * sy,
                   centroid_z_mm=seg.origin[2] + idx[:, 2].mean() * sz)
        row.update(contour_stats(m[:, :, k], sx, sy))
        if row.get("R_mean_mm"):
            row["kappa_of_equiv_circle"] = 1.0 / row["eq_radius_mm"]
        if img is not None and img.data.shape == seg.data.shape:
            row["mean_HU"] = float(img.data[m].mean()); row["std_HU"] = float(img.data[m].std())
        rows.append(row)
    return pd.DataFrame(rows)


def cnr_table(rec_mu, seg2d, truth_mu=None, ring_iters=(1, 4), margin=5):
    """Contrast-to-noise (report eq. 14): CNR = tumour jump / B,  B = RMS error (or robust noise) in surrounding tissue."""
    from scipy.ndimage import binary_dilation, binary_erosion
    rec_hu = mu_to_hu(rec_mu)
    ref_hu = mu_to_hu(truth_mu) if truth_mu is not None else rec_hu
    tissue = (ref_hu > -300) & (ref_hu < 300)
    anyseg = seg2d > 0
    region = binary_erosion(tissue, iterations=margin) & ~binary_dilation(anyseg, iterations=6)
    if region.sum() < 20:
        return pd.DataFrame()
    if truth_mu is not None:
        B = float(np.sqrt(np.mean((rec_mu - truth_mu)[region] ** 2))); how = "RMS error vs truth"
    else:
        r = rec_mu[region]; B = float(1.4826 * np.median(np.abs(r - np.median(r)))); how = "robust std in tissue"
    rows = []
    for lab in [int(v) for v in np.unique(seg2d) if v != 0]:
        m = seg2d == lab
        ring = binary_dilation(m, iterations=ring_iters[1]) & ~binary_dilation(m, iterations=ring_iters[0]) & ~anyseg & tissue
        if ring.sum() == 0:
            continue
        jr = float(rec_mu[m].mean() - rec_mu[ring].mean())
        jt = float(truth_mu[m].mean() - truth_mu[ring].mean()) if truth_mu is not None else np.nan
        rows.append(dict(label=lab, true_jump=jt, recon_jump=jr, B=B, B_definition=how, CNR=jr / B, visible_CNR_ge_3=bool(jr / B >= 3)))
    return pd.DataFrame(rows)


# ================================================================ alignment
def seg_slice_at(seg, z_mm, shape):
    """Segmentation slice nearest to height z_mm, resampled (nearest neighbour) to in-plane `shape`."""
    from scipy.ndimage import zoom
    if seg.z_pos is not None and seg.source != "nifti":
        k = int(np.argmin(np.abs(np.array(seg.z_pos) - z_mm)))
    else:
        k = int(np.clip(round((z_mm - seg.origin[2]) / seg.spacing[2]), 0, seg.data.shape[2] - 1))
    sl = seg.data[:, :, k]
    if sl.shape != tuple(shape):
        sl = zoom(sl, (shape[0] / sl.shape[0], shape[1] / sl.shape[1]), order=0)
    return sl


# =============================================================== figures
def _fig_slice(items, ext, suptitle=""):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5 * len(items), 4.4))
    for i, (title, img, kw, seg) in enumerate(items):
        ax = fig.add_subplot(1, len(items), i + 1)
        im = ax.imshow(np.asarray(img).T, origin="lower", extent=ext, **kw)
        ax.set_title(title, fontsize=10); ax.set_xlabel("x [cm]")
        fig.colorbar(im, ax=ax, fraction=0.046)
        if seg is not None:
            for lab in np.unique(seg):
                if lab:
                    ax.contour(np.linspace(ext[0], ext[1], seg.shape[0]), np.linspace(ext[2], ext[3], seg.shape[1]),
                               (seg == lab).T.astype(float), [0.5], colors=["yellow", "cyan", "lime", "magenta"][int(lab) % 4], linewidths=0.8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def edge_map(hu2d, sp_mm):
    gx, gy = np.gradient(hu2d.astype(float), sp_mm[0], sp_mm[1])
    return np.hypot(gx, gy)


# ================================================================ pipeline
def run_pipeline(L, lam=10.0, nmax=30, box=30.0, simulate=False, sigma=0.005, max_slices=5, grid_half=10.0,
                 grid_vox=0.2, log=print):
    """Main entry point (used by the CLI and the Streamlit app). Returns a dict of results."""
    R = dict(report=[], seg_table=None, cnr=None, recon=None, rays=None, figs={}, notes=[])
    note = lambda m: (R["notes"].append(m), log(m))
    img = L.volumes[0] if L.volumes else None
    seg = L.segs[0] if L.segs else None

    # ---- segmentation-only measurements on the data as supplied
    if seg is not None:
        R["seg_table"] = analyze_segmentation(seg, img if (img is not None and img.data.shape == seg.data.shape) else None)

    rec = truth_stack = None
    if L.rays:
        rays = L.rays[0]
        note(f"reconstructing from raw rays table ({len(rays)} rays)")
        n = int(round(2 * grid_half / grid_vox)); c = (np.arange(n) + 0.5) * grid_vox - grid_half
        X, Y = np.meshgrid(c, c, indexing="ij")
        mu_stack, zs = reconstruct(rays, X, Y, nmax, lam, box, log=note)
        sp_cm = (grid_vox, grid_vox)
        dz = float(np.median(np.diff(zs))) if len(zs) > 1 else grid_vox
        R["rays"] = rays
    elif img is not None and simulate:
        st = max(1, int(np.ceil(max(img.data.shape[:2]) / 256)))
        data = img.data[::st, ::st]
        sp_cm = (img.spacing[0] * st / 10, img.spacing[1] * st / 10)
        ext = max(data.shape[0] * sp_cm[0], data.shape[1] * sp_cm[1])
        if ext > box:
            note(f"image field of view {ext:.1f} cm is larger than the model box {box:.0f} cm: parts outside are ignored "
                 f"(crop the head or raise box)")
        if seg is not None and seg.data.any():
            zs_idx = np.where(seg.data.any((0, 1)))[0]
            mid = zs_idx[np.argmax((seg.data > 0).sum((0, 1))[zs_idx])]
            cand = sorted(set(np.clip(np.arange(mid - max_slices // 2, mid - max_slices // 2 + max_slices), 0, img.data.shape[2] - 1)))
        else:
            mid = img.data.shape[2] // 2
            cand = [mid]
        note(f"simulating a scan of slices {cand} (sigma={sigma}); forward-projecting along 120 x 151 rays each")
        mu_in = [hu_to_mu(data[:, :, k]) for k in cand]
        zs = [(img.origin[2] + k * img.spacing[2]) / 10 for k in cand]
        rays = simulate_scan(mu_in, sp_cm, zs, box=box, sigma=sigma)
        R["rays"] = rays
        std = standardise_table(rays, log=[])
        nx, ny = data.shape[:2]
        X, Y = np.meshgrid((np.arange(nx) - (nx - 1) / 2) * sp_cm[0], (np.arange(ny) - (ny - 1) / 2) * sp_cm[1], indexing="ij")
        mu_stack, zs = reconstruct(std, X, Y, nmax, lam, box, log=note)
        outside = (np.abs(X) > box / 2) | (np.abs(Y) > box / 2)
        mu_stack[outside] = 0.0
        truth_stack = np.stack([m for m in mu_in], 2)
        dz = img.spacing[2] / 10
        seg_s = None
    elif img is not None:
        note("image volume loaded; tick 'simulate scan' to run the reconstruction pipeline on it")
    if rec is None and 'mu_stack' in locals():
        origin_mm = ((-(mu_stack.shape[0] - 1) / 2) * sp_cm[0] * 10 if truth_stack is not None else -grid_half * 10 + sp_cm[0] * 5,
                     (-(mu_stack.shape[1] - 1) / 2) * sp_cm[1] * 10 if truth_stack is not None else -grid_half * 10 + sp_cm[1] * 5,
                     zs[0] * 10)
        R["recon"] = Volume(mu_to_hu(mu_stack).astype(np.float32), (sp_cm[0] * 10, sp_cm[1] * 10, max(dz * 10, 0.1)),
                            origin_mm, name="reconstruction (HU)", kind="image", source="recon")
        ext = [(-(mu_stack.shape[0]) / 2) * sp_cm[0], (mu_stack.shape[0] / 2) * sp_cm[0],
               (-(mu_stack.shape[1]) / 2) * sp_cm[1], (mu_stack.shape[1] / 2) * sp_cm[1]]
        k = len(zs) // 2
        z_mm = zs[k] * 10
        seg2d = None
        if seg is not None:
            seg2d = seg_slice_at(seg, z_mm, mu_stack.shape[:2])
            if truth_stack is None:                                # raw-table route: seg grid assumed to match
                note("segmentation resampled to the reconstruction grid by height z and nearest neighbour")
            else:
                seg2d = seg.data[::max(1, int(np.ceil(max(img.data.shape[:2]) / 256))), ::max(1, int(np.ceil(max(img.data.shape[:2]) / 256))), cand[k]]
        truth2d = truth_stack[:, :, k] if truth_stack is not None else None
        if seg2d is not None and seg2d.any():
            R["cnr"] = cnr_table(mu_stack[:, :, k], seg2d, truth2d)
        items = []
        if truth2d is not None:
            items.append(("Input slice (HU)", mu_to_hu(truth2d), dict(cmap="gray", vmin=-100, vmax=300), seg2d))
        items.append((f"Reconstruction, lambda={lam:g} (HU)", mu_to_hu(mu_stack[:, :, k]), dict(cmap="gray", vmin=-100, vmax=300), seg2d))
        if truth2d is not None:
            items.append(("Error  mu_rec - mu_true", mu_stack[:, :, k] - truth2d, dict(cmap="RdBu_r", vmin=-0.05, vmax=0.05), None))
        else:
            items.append(("Gradient magnitude |grad mu| (edges)", edge_map(mu_to_hu(mu_stack[:, :, k]), (sp_cm[0] * 10, sp_cm[1] * 10)),
                          dict(cmap="magma"), None))
        R["figs"]["reconstruction"] = _fig_slice(items, ext, f"slice z = {z_mm:.1f} mm")

    # ---- figure for a supplied image/seg (no reconstruction)
    if img is not None and "reconstruction" not in R["figs"]:
        k = img.data.shape[2] // 2
        if seg is not None and seg.data.shape == img.data.shape and seg.data.any():
            k = int(np.argmax((seg.data > 0).sum((0, 1))))
        sx, sy = img.spacing[0] / 10, img.spacing[1] / 10
        ext = [-img.data.shape[0] * sx / 2, img.data.shape[0] * sx / 2, -img.data.shape[1] * sy / 2, img.data.shape[1] * sy / 2]
        s2 = seg.data[:, :, k] if (seg is not None and seg.data.shape == img.data.shape) else None
        R["figs"]["input"] = _fig_slice([(f"{img.name}  slice {k}", img.data[:, :, k], dict(cmap="gray", vmin=-100, vmax=300), s2),
                                         ("|grad| edge map", edge_map(img.data[:, :, k], img.spacing), dict(cmap="magma"), None)], ext)
    return R


# ==================================================================== CLI
def save_results(R, out):
    os.makedirs(out, exist_ok=True)
    if R["rays"] is not None:
        R["rays"].round(6).to_csv(os.path.join(out, "rays_used.csv"), index=False)
    if R["recon"] is not None:
        v = R["recon"]
        write_nifti(os.path.join(out, "reconstruction.nii"), np.rint(v.data), v.spacing, v.origin)
    if R["seg_table"] is not None:
        R["seg_table"].to_csv(os.path.join(out, "segmentation_measurements.csv"), index=False)
    if R["cnr"] is not None and len(R["cnr"]):
        R["cnr"].to_csv(os.path.join(out, "cnr.csv"), index=False)
    for k, f in R["figs"].items():
        f.savefig(os.path.join(out, f"{k}.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="files, folders or zips of any supported format")
    ap.add_argument("--seg", nargs="*", default=[], help="segmentation file(s) (also auto-detected)")
    ap.add_argument("--simulate", action="store_true", help="forward-project the image into raw rays, then reconstruct")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--nmax", type=int, default=30)
    ap.add_argument("--box", type=float, default=30.0, help="model box width D [cm]")
    ap.add_argument("--sigma", type=float, default=0.005)
    ap.add_argument("--units", default="cm", choices=["cm", "mm"], help="length units of a ray table")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    L = load_any(a.inputs + a.seg, units=a.units, force_seg=tuple(a.seg))
    print("\n".join(L.log))
    R = run_pipeline(L, a.lam, a.nmax, a.box, a.simulate, a.sigma)
    save_results(R, a.out)
    for name in ("seg_table", "cnr"):
        if R[name] is not None and len(R[name]):
            print(f"\n{name}:\n{R[name].round(4).T.to_string()}")
    print(f"\nresults saved in {a.out}/")


if __name__ == "__main__":
    main()
