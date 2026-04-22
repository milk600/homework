import json
import math
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage, stats
from skimage.registration import phase_cross_correlation
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

CPU_COUNT = os.cpu_count() or 4
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
NUM_THREADS = min(16, CPU_COUNT)
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATE_TAG = datetime.now().strftime("%Y%m%d")
ROOT = Path(r"e:\Latex")
DATA_DIR = ROOT / "workspace" / "data"
RESULTS_DIR = ROOT / "workspace" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "124046": {
        2021: DATA_DIR / "LC08_L2SP_124046_20210101_20210308_02_T1",
        2025: DATA_DIR / "LC08_L2SP_124046_20250112_20250122_02_T1",
    },
    "125047": {
        2021: DATA_DIR / "LC08_L2SP_125047_20210124_20210305_02_T1",
        2025: DATA_DIR / "LC08_L2SP_125047_20250103_20250111_02_T1",
    },
}

BANDS_SR = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
RES_DEG = 30.0 / 111320.0


def parse_mtl(mtl_path: Path) -> dict:
    meta = {}
    pattern = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)$")
    with open(mtl_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            key, value = m.groups()
            meta[key] = value.strip().strip('"')
    return meta


def scene_band_path(scene_dir: Path, suffix: str) -> Path:
    matches = list(scene_dir.glob(f"*_{suffix}.TIF"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{scene_dir} missing unique {suffix}.TIF")
    return matches[0]


def decode_masks(scene_dir: Path):
    qa_pixel_path = scene_band_path(scene_dir, "QA_PIXEL")
    qa_radsat_path = scene_band_path(scene_dir, "QA_RADSAT")
    with rasterio.open(qa_pixel_path) as src:
        qa_pixel = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
    with rasterio.open(qa_radsat_path) as src:
        qa_radsat = src.read(1)

    fill_ok = ((qa_pixel >> 0) & 1) == 0
    dilated_ok = ((qa_pixel >> 1) & 1) == 0
    cirrus_ok = ((qa_pixel >> 2) & 1) == 0
    cloud_ok = ((qa_pixel >> 3) & 1) == 0
    shadow_ok = ((qa_pixel >> 4) & 1) == 0
    snow_ok = ((qa_pixel >> 5) & 1) == 0
    water = ((qa_pixel >> 7) & 1) == 1
    unsat = qa_radsat == 0

    good = fill_ok & dilated_ok & cirrus_ok & cloud_ok & shadow_ok & snow_ok & unsat
    land = good & (~water)
    return good.astype(np.uint8), land.astype(np.uint8), transform, crs, profile


def read_scaled_band(scene_dir: Path, band: str, meta: dict):
    suffix = f"SR_{band}" if band != "ST_B10" else "ST_B10"
    path = scene_band_path(scene_dir, suffix)
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
    arr[arr == 0] = np.nan
    if band.startswith("B"):
        band_num = band.replace("B", "")
        arr = arr * float(meta[f"REFLECTANCE_MULT_BAND_{band_num}"]) + float(meta[f"REFLECTANCE_ADD_BAND_{band_num}"])
    else:
        arr = arr * float(meta["TEMPERATURE_MULT_BAND_ST_B10"]) + float(meta["TEMPERATURE_ADD_BAND_ST_B10"])
    return arr.astype(np.float32), transform, crs, profile


def reproject_to_match(src_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs, resampling):
    out = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
        num_threads=NUM_THREADS,
    )
    return out


def reproject_mask(src_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    out = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=src_arr.astype(np.uint8),
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
        num_threads=NUM_THREADS,
    )
    return out.astype(bool)


def estimate_shift(base, moving, valid, max_side=1024, max_abs_shift_px=20.0):
    finite_valid = valid & np.isfinite(base) & np.isfinite(moving)
    if np.count_nonzero(finite_valid) < 1000:
        return (0.0, 0.0), np.nan

    rows, cols = np.where(finite_valid)
    r0 = max(int(rows.min()) - 32, 0)
    r1 = min(int(rows.max()) + 33, base.shape[0])
    c0 = max(int(cols.min()) - 32, 0)
    c1 = min(int(cols.max()) + 33, base.shape[1])
    a = base[r0:r1, c0:c1]
    b = moving[r0:r1, c0:c1]
    v = finite_valid[r0:r1, c0:c1]

    med_a = np.nanmedian(a[v])
    med_b = np.nanmedian(b[v])
    if not np.isfinite(med_a) or not np.isfinite(med_b):
        return (0.0, 0.0), np.nan
    a = np.where(v, a, med_a).astype(np.float32, copy=False)
    b = np.where(v, b, med_b).astype(np.float32, copy=False)

    step = int(math.ceil(max(a.shape) / max_side)) if max(a.shape) > max_side else 1
    if step > 1:
        a = a[::step, ::step]
        b = b[::step, ::step]

    shift, error, _ = phase_cross_correlation(a, b, upsample_factor=5)
    shift = np.asarray(shift, dtype=np.float32) * step
    if np.any(np.abs(shift) > max_abs_shift_px):
        return (0.0, 0.0), float(error)
    return (float(shift[0]), float(shift[1])), float(error)


def apply_shift(arr, shift):
    filled = np.where(np.isfinite(arr), arr, 0.0)
    shifted = ndimage.shift(filled, shift=shift, order=1, mode="nearest")
    valid = ndimage.shift(np.isfinite(arr).astype(np.float32), shift=shift, order=0, mode="nearest") > 0.5
    shifted[~valid] = np.nan
    return shifted.astype(np.float32)


def compute_indices(bands):
    blue = bands["B2"]
    red = bands["B4"]
    nir = bands["B5"]
    swir1 = bands["B6"]
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1.0)
    savi = 1.5 * (nir - red) / (nir + red + 0.5)
    ndbi = (swir1 - nir) / (swir1 + nir)
    lst_c = bands["ST_B10"] - 273.15
    return {
        "NDVI": ndvi.astype(np.float32),
        "EVI": evi.astype(np.float32),
        "SAVI": savi.astype(np.float32),
        "NDBI": ndbi.astype(np.float32),
        "LST_C": lst_c.astype(np.float32),
    }


def local_ttest(arr1, arr2, valid_mask, size=5):
    x = np.where(valid_mask, arr1, 0.0)
    y = np.where(valid_mask, arr2, 0.0)
    m = valid_mask.astype(np.float32)
    n = ndimage.uniform_filter(m, size=size, mode="nearest") * (size ** 2)
    mean1 = ndimage.uniform_filter(x, size=size, mode="nearest") * (size ** 2) / np.maximum(n, 1)
    mean2 = ndimage.uniform_filter(y, size=size, mode="nearest") * (size ** 2) / np.maximum(n, 1)
    sq1 = ndimage.uniform_filter(x * x, size=size, mode="nearest") * (size ** 2) / np.maximum(n, 1)
    sq2 = ndimage.uniform_filter(y * y, size=size, mode="nearest") * (size ** 2) / np.maximum(n, 1)
    var1 = np.maximum(sq1 - mean1 * mean1, 0)
    var2 = np.maximum(sq2 - mean2 * mean2, 0)
    se = np.sqrt(var1 / np.maximum(n, 1) + var2 / np.maximum(n, 1))
    tstat = (mean2 - mean1) / np.where(se == 0, np.nan, se)
    df_num = (var1 / np.maximum(n, 1) + var2 / np.maximum(n, 1)) ** 2
    df_den = ((var1 / np.maximum(n, 1)) ** 2 / np.maximum(n - 1, 1)) + ((var2 / np.maximum(n, 1)) ** 2 / np.maximum(n - 1, 1))
    df = df_num / np.where(df_den == 0, np.nan, df_den)
    p = 2 * stats.t.sf(np.abs(tstat), np.where(np.isfinite(df), df, 1))
    p[(n < 4) | (~valid_mask)] = np.nan
    return p.astype(np.float32)


def compute_mad(bands2021, bands2025, valid_mask, sample_size=40000, chunk_size=80000):
    idx_all = np.flatnonzero(valid_mask.ravel())
    if idx_all.size < len(BANDS_SR) * 10:
        out = np.full(valid_mask.shape, np.nan, dtype=np.float32)
        mask = np.full(valid_mask.shape, -9999, dtype=np.int16)
        return out, mask, float("nan")

    rng = np.random.default_rng(42)
    if idx_all.size > sample_size:
        idx_fit = idx_all[rng.choice(idx_all.size, size=sample_size, replace=False)]
    else:
        idx_fit = idx_all

    x_fit = np.column_stack([bands2021[b].ravel()[idx_fit] for b in BANDS_SR]).astype(np.float32)
    y_fit = np.column_stack([bands2025[b].ravel()[idx_fit] for b in BANDS_SR]).astype(np.float32)
    finite_fit = np.isfinite(x_fit).all(axis=1) & np.isfinite(y_fit).all(axis=1)
    x_fit = x_fit[finite_fit]
    y_fit = y_fit[finite_fit]
    if x_fit.shape[0] < len(BANDS_SR) * 5:
        out = np.full(valid_mask.shape, np.nan, dtype=np.float32)
        mask = np.full(valid_mask.shape, -9999, dtype=np.int16)
        return out, mask, float("nan")

    n_comp = min(3, len(BANDS_SR))
    sx = StandardScaler().fit(x_fit)
    sy = StandardScaler().fit(y_fit)
    cca = CCA(n_components=n_comp, max_iter=300)
    cca.fit(sx.transform(x_fit), sy.transform(y_fit))

    intensity_flat = np.full(valid_mask.size, np.nan, dtype=np.float32)
    sum_vec = np.zeros(n_comp, dtype=np.float64)
    sumsq_vec = np.zeros(n_comp, dtype=np.float64)
    total_n = 0

    for start in range(0, idx_all.size, chunk_size):
        idx_chunk = idx_all[start : start + chunk_size]
        x = np.column_stack([bands2021[b].ravel()[idx_chunk] for b in BANDS_SR]).astype(np.float32)
        y = np.column_stack([bands2025[b].ravel()[idx_chunk] for b in BANDS_SR]).astype(np.float32)
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        if not np.any(finite):
            continue
        x = sx.transform(x[finite])
        y = sy.transform(y[finite])
        u, v = cca.transform(x, y)
        mad = (u - v).astype(np.float32)
        sum_vec += np.sum(mad, axis=0)
        sumsq_vec += np.sum(mad * mad, axis=0)
        total_n += mad.shape[0]
        intensity_flat[idx_chunk[finite]] = np.linalg.norm(mad, axis=1).astype(np.float32)

    if total_n == 0:
        out = np.full(valid_mask.shape, np.nan, dtype=np.float32)
        mask = np.full(valid_mask.shape, -9999, dtype=np.int16)
        return out, mask, float("nan")

    mean_vec = sum_vec / total_n
    std_vec = np.sqrt(np.maximum((sumsq_vec / total_n) - mean_vec * mean_vec, 1e-8))
    scale = float(np.sqrt(np.sum(std_vec * std_vec)))
    if scale <= 0:
        scale = 1.0
    intensity_flat[idx_all] = intensity_flat[idx_all] / scale
    intensity = intensity_flat[idx_all]
    threshold = float(np.nanmean(intensity) + 2 * np.nanstd(intensity))

    out = intensity_flat.reshape(valid_mask.shape)
    mask = np.where(out > threshold, 1, 0).astype(np.int16)
    mask[~valid_mask] = -9999
    return out, mask, threshold


def save_raster(path: Path, arr, profile, dtype="float32", nodata=np.nan):
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, compress="lzw", nodata=nodata if not (isinstance(nodata, float) and np.isnan(nodata)) else None)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(dtype), 1)


def pair_pixel_area_ha(transform, shape, crs):
    if crs is not None and getattr(crs, "is_projected", False):
        return np.full(shape, abs(transform.a * transform.e) / 10000.0, dtype=np.float32)
    _, _, _, ymax = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    rows = np.arange(shape[0])
    lats = np.abs(ymax - (rows + 0.5) * abs(transform.e))
    dy_m = abs(transform.e) * 111320.0
    dx_m = abs(transform.a) * 111320.0 * np.cos(np.deg2rad(lats))
    area_row = (dx_m * dy_m) / 10000.0
    return np.repeat(area_row[:, None], shape[1], axis=1).astype(np.float32)


def add_scale_bar(ax, extent):
    xmin, xmax, ymin, ymax = extent
    mean_lat = (ymin + ymax) / 2
    km = 20
    deg = km / (111.32 * math.cos(math.radians(mean_lat)))
    x0 = xmin + (xmax - xmin) * 0.06
    y0 = ymin + (ymax - ymin) * 0.06
    ax.plot([x0, x0 + deg], [y0, y0], color="k", linewidth=3)
    ax.text(x0 + deg / 2, y0 + (ymax - ymin) * 0.015, f"{km} km", ha="center", va="bottom", fontsize=9)


def add_north_arrow(ax, extent):
    xmin, xmax, ymin, ymax = extent
    x = xmax - (xmax - xmin) * 0.06
    y = ymax - (ymax - ymin) * 0.12
    ax.annotate("N", xy=(x, y), xytext=(x, y - (ymax - ymin) * 0.08), arrowprops=dict(facecolor="black", width=3, headwidth=10), ha="center", va="center", fontsize=12, fontweight="bold")


def make_map(arr, transform, out_png: Path, title, cmap, vmin=None, vmax=None, discrete=False, labels=None):
    left, bottom, right, top = rasterio.transform.array_bounds(arr.shape[0], arr.shape[1], transform)
    extent = [left, right, bottom, top]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(np.ma.masked_invalid(arr.astype(float)), extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_scale_bar(ax, extent)
    add_north_arrow(ax, extent)
    if discrete and labels:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=list(labels.keys()))
        cbar.ax.set_yticklabels(list(labels.values()))
    else:
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def process_pair(pair_id: str, scene2021: Path, scene2025: Path):
    meta2021 = parse_mtl(next(scene2021.glob("*_MTL.txt")))
    meta2025 = parse_mtl(next(scene2025.glob("*_MTL.txt")))

    good2021, land2021, _, _, _ = decode_masks(scene2021)
    good2025, land2025, qa25_transform, qa25_crs, _ = decode_masks(scene2025)

    b4_2021, base_transform, base_crs, base_profile = read_scaled_band(scene2021, "B4", meta2021)
    b4_2025, tr25, crs25, _ = read_scaled_band(scene2025, "B4", meta2025)
    shape = b4_2021.shape

    b4_2025 = reproject_to_match(b4_2025, tr25, crs25, shape, base_transform, base_crs, Resampling.bilinear)
    good25 = reproject_mask(good2025, qa25_transform, qa25_crs, shape, base_transform, base_crs)
    land25 = reproject_mask(land2025, qa25_transform, qa25_crs, shape, base_transform, base_crs)
    common_valid = np.isfinite(b4_2021) & np.isfinite(b4_2025) & land2021.astype(bool) & land25

    shift, reg_error = estimate_shift(b4_2021, b4_2025, common_valid)

    bands2021 = {}
    bands2025 = {}

    def _prepare_band(band):
        arr21, _, _, _ = read_scaled_band(scene2021, band, meta2021)
        arr25, tr25, crs25, _ = read_scaled_band(scene2025, band, meta2025)
        arr25 = reproject_to_match(arr25, tr25, crs25, shape, base_transform, base_crs, Resampling.bilinear)
        arr25 = apply_shift(arr25, shift)
        arr21[~good2021.astype(bool)] = np.nan
        arr25[~good25] = np.nan
        return band, arr21.astype(np.float32), arr25.astype(np.float32)

    with ThreadPoolExecutor(max_workers=min(6, NUM_THREADS, len(BANDS_SR) + 1)) as ex:
        futures = [ex.submit(_prepare_band, band) for band in (BANDS_SR + ["ST_B10"])]
        for fu in as_completed(futures):
            band, arr21, arr25 = fu.result()
            bands2021[band] = arr21
            bands2025[band] = arr25

    common_valid = np.ones(shape, dtype=bool)
    for band in BANDS_SR + ["ST_B10"]:
        common_valid &= np.isfinite(bands2021[band]) & np.isfinite(bands2025[band])
    common_valid &= land2021.astype(bool) & land25

    idx2021 = compute_indices(bands2021)
    idx2025 = compute_indices(bands2025)
    delta_lst = idx2025["LST_C"] - idx2021["LST_C"]
    delta_ndvi = idx2025["NDVI"] - idx2021["NDVI"]
    delta_evi = idx2025["EVI"] - idx2021["EVI"]
    delta_savi = idx2025["SAVI"] - idx2021["SAVI"]

    pvals = local_ttest(idx2021["LST_C"], idx2025["LST_C"], common_valid, size=5)
    signif_lst = np.full(shape, -9999, dtype=np.int16)
    signif_lst[(pvals < 0.05) & (delta_lst > 0)] = 1
    signif_lst[(pvals < 0.05) & (delta_lst < 0)] = -1
    signif_lst[(pvals >= 0.05) & common_valid] = 0

    rural_mask = common_valid & (idx2025["NDVI"] > 0.5) & (idx2025["NDBI"] < 0)
    rural_mean = float(np.nanmean(idx2025["LST_C"][rural_mask])) if np.any(rural_mask) else float(np.nanmean(idx2025["LST_C"][common_valid]))
    uhi = idx2025["LST_C"] - rural_mean
    uhi_class = np.full(shape, -9999, dtype=np.int16)
    uhi_class[(uhi <= 0) & common_valid] = 0
    uhi_class[(uhi > 0) & (uhi <= 2)] = 1
    uhi_class[(uhi > 2) & (uhi <= 4)] = 2
    uhi_class[uhi > 4] = 3

    mad_intensity, mad_mask, mad_thr = compute_mad(bands2021, bands2025, common_valid)

    pair_dir = RESULTS_DIR / f"{DATE_TAG}_{pair_id}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_lst_2021.tif", idx2021["LST_C"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_lst_2025.tif", idx2025["LST_C"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_delta_lst.tif", delta_lst, base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_ndvi_2021.tif", idx2021["NDVI"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_ndvi_2025.tif", idx2025["NDVI"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_delta_ndvi.tif", delta_ndvi, base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_evi_2021.tif", idx2021["EVI"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_evi_2025.tif", idx2025["EVI"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_savi_2021.tif", idx2021["SAVI"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_savi_2025.tif", idx2025["SAVI"], base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_uhi_2025.tif", uhi, base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_mad_intensity.tif", mad_intensity, base_profile)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_significant_lst_change.tif", signif_lst.astype(np.int16), base_profile, dtype="int16", nodata=-9999)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_uhi_class_2025.tif", uhi_class.astype(np.int16), base_profile, dtype="int16", nodata=-9999)
    save_raster(pair_dir / f"{DATE_TAG}_{pair_id}_mad_change_mask.tif", mad_mask.astype(np.int16), base_profile, dtype="int16", nodata=-9999)

    area = pair_pixel_area_ha(base_transform, shape, base_crs)
    stats_rows = [
        {"region_name": pair_id, "metric": "LST_C", "year": 2021, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2021["LST_C"][common_valid])), "std": float(np.nanstd(idx2021["LST_C"][common_valid]))},
        {"region_name": pair_id, "metric": "LST_C", "year": 2025, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2025["LST_C"][common_valid])), "std": float(np.nanstd(idx2025["LST_C"][common_valid]))},
        {"region_name": pair_id, "metric": "delta_LST_C", "year": "2021_2025", "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(delta_lst[common_valid])), "std": float(np.nanstd(delta_lst[common_valid]))},
        {"region_name": pair_id, "metric": "NDVI", "year": 2021, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2021["NDVI"][common_valid])), "std": float(np.nanstd(idx2021["NDVI"][common_valid]))},
        {"region_name": pair_id, "metric": "NDVI", "year": 2025, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2025["NDVI"][common_valid])), "std": float(np.nanstd(idx2025["NDVI"][common_valid]))},
        {"region_name": pair_id, "metric": "delta_NDVI", "year": "2021_2025", "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(delta_ndvi[common_valid])), "std": float(np.nanstd(delta_ndvi[common_valid]))},
        {"region_name": pair_id, "metric": "EVI", "year": 2021, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2021["EVI"][common_valid])), "std": float(np.nanstd(idx2021["EVI"][common_valid]))},
        {"region_name": pair_id, "metric": "EVI", "year": 2025, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2025["EVI"][common_valid])), "std": float(np.nanstd(idx2025["EVI"][common_valid]))},
        {"region_name": pair_id, "metric": "SAVI", "year": 2021, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2021["SAVI"][common_valid])), "std": float(np.nanstd(idx2021["SAVI"][common_valid]))},
        {"region_name": pair_id, "metric": "SAVI", "year": 2025, "area_ha": float(np.nansum(area[common_valid])), "mean": float(np.nanmean(idx2025["SAVI"][common_valid])), "std": float(np.nanstd(idx2025["SAVI"][common_valid]))},
    ]
    for code, label in [(0, "NoUHI"), (1, "WeakUHI"), (2, "ModerateUHI"), (3, "StrongUHI")]:
        mask = uhi_class == code
        if np.any(mask):
            stats_rows.append({"region_name": f"{pair_id}_{label}", "metric": "UHI_2025", "year": 2025, "area_ha": float(np.nansum(area[mask])), "mean": float(np.nanmean(uhi[mask])), "std": float(np.nanstd(uhi[mask]))})

    meta_rows = [
        {"pair": pair_id, "year": 2021, "scene": scene2021.name, "acquisition_date": meta2021.get("DATE_ACQUIRED"), "sun_elevation": float(meta2021.get("SUN_ELEVATION")), "sun_azimuth": float(meta2021.get("SUN_AZIMUTH")), "sr_mult": float(meta2021.get("REFLECTANCE_MULT_BAND_4")), "sr_add": float(meta2021.get("REFLECTANCE_ADD_BAND_4")), "st_mult": float(meta2021.get("TEMPERATURE_MULT_BAND_ST_B10")), "st_add": float(meta2021.get("TEMPERATURE_ADD_BAND_ST_B10")), "geom_rmse_model_m": float(meta2021.get("GEOMETRIC_RMSE_MODEL", "nan"))},
        {"pair": pair_id, "year": 2025, "scene": scene2025.name, "acquisition_date": meta2025.get("DATE_ACQUIRED"), "sun_elevation": float(meta2025.get("SUN_ELEVATION")), "sun_azimuth": float(meta2025.get("SUN_AZIMUTH")), "sr_mult": float(meta2025.get("REFLECTANCE_MULT_BAND_4")), "sr_add": float(meta2025.get("REFLECTANCE_ADD_BAND_4")), "st_mult": float(meta2025.get("TEMPERATURE_MULT_BAND_ST_B10")), "st_add": float(meta2025.get("TEMPERATURE_ADD_BAND_ST_B10")), "geom_rmse_model_m": float(meta2025.get("GEOMETRIC_RMSE_MODEL", "nan"))},
    ]

    summary = {
        "pair": pair_id,
        "pair_dir": str(pair_dir),
        "profile": base_profile,
        "crs": str(base_crs),
        "transform": base_transform,
        "shape": shape,
        "shift_row_px": shift[0],
        "shift_col_px": shift[1],
        "reg_error": reg_error,
        "lst_mean_2021": float(np.nanmean(idx2021["LST_C"][common_valid])),
        "lst_mean_2025": float(np.nanmean(idx2025["LST_C"][common_valid])),
        "lst_delta_mean": float(np.nanmean(delta_lst[common_valid])),
        "ndvi_improved_ha": float(np.nansum(area[common_valid & (delta_ndvi > 0.05)])),
        "ndvi_degraded_ha": float(np.nansum(area[common_valid & (delta_ndvi < -0.05)])),
        "strong_uhi_ha": float(np.nansum(area[uhi_class == 3])),
        "analysis_area_ha": float(np.nansum(area[common_valid])),
        "mad_threshold": mad_thr,
        "files": {
            "delta_lst": pair_dir / f"{DATE_TAG}_{pair_id}_delta_lst.tif",
            "delta_ndvi": pair_dir / f"{DATE_TAG}_{pair_id}_delta_ndvi.tif",
            "uhi_class": pair_dir / f"{DATE_TAG}_{pair_id}_uhi_class_2025.tif",
            "significant_lst_change": pair_dir / f"{DATE_TAG}_{pair_id}_significant_lst_change.tif",
            "mad_intensity": pair_dir / f"{DATE_TAG}_{pair_id}_mad_intensity.tif",
            "mad_change_mask": pair_dir / f"{DATE_TAG}_{pair_id}_mad_change_mask.tif",
            "lst_2021": pair_dir / f"{DATE_TAG}_{pair_id}_lst_2021.tif",
            "lst_2025": pair_dir / f"{DATE_TAG}_{pair_id}_lst_2025.tif",
        },
    }
    return summary, pd.DataFrame(stats_rows), pd.DataFrame(meta_rows)


def build_mosaic_grid(pair_outputs):
    bounds = []
    for item in pair_outputs:
        with rasterio.open(item["files"]["delta_lst"]) as src:
            bounds.append(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)
    width = int(math.ceil((xmax - xmin) / RES_DEG))
    height = int(math.ceil((ymax - ymin) / RES_DEG))
    transform = from_origin(xmin, ymax, RES_DEG, RES_DEG)
    return transform, (height, width)


def mosaic_result(files, dst_transform, dst_shape, method="mean"):
    sum_arr = np.zeros(dst_shape, dtype=np.float32)
    count_arr = np.zeros(dst_shape, dtype=np.uint16)
    latest_class = np.full(dst_shape, np.nan, dtype=np.float32)
    for path in files:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr == src.nodata] = np.nan if src.nodata is not None else arr[arr == src.nodata]
            dst = np.full(dst_shape, np.nan, dtype=np.float32)
            reproject(
                source=arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.nearest if "class" in path.name or "mask" in path.name else Resampling.bilinear,
                num_threads=NUM_THREADS,
            )
            if method == "first":
                fill = np.isnan(latest_class) & np.isfinite(dst)
                latest_class[fill] = dst[fill]
            else:
                valid = np.isfinite(dst)
                sum_arr[valid] += dst[valid]
                count_arr[valid] += 1
    if method == "first":
        return latest_class
    out = np.full(dst_shape, np.nan, dtype=np.float32)
    valid = count_arr > 0
    out[valid] = sum_arr[valid] / count_arr[valid]
    return out


def sample_validation_points_from_mosaic(delta_ndvi, delta_lst, valid_mask, transform, n=100):
    rng = np.random.default_rng(42)
    stable = valid_mask & (np.abs(delta_ndvi) < 0.05) & (np.abs(delta_lst) < 1.0)
    candidates = np.argwhere(stable)
    if len(candidates) == 0:
        return pd.DataFrame(columns=["point_id", "lon", "lat", "predicted_class", "ge_label"])
    chosen = candidates[rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)]
    rows = []
    for i, (r, c) in enumerate(chosen, start=1):
        x, y = rasterio.transform.xy(transform, r, c, offset="center")
        rows.append({"point_id": i, "lon": x, "lat": y, "predicted_class": "stable", "ge_label": ""})
    return pd.DataFrame(rows)


def compute_confusion_if_labeled(points_csv: Path):
    if not points_csv.exists():
        return None
    df = pd.read_csv(points_csv)
    if "ge_label" not in df.columns:
        return None
    ge_label = (
        df["ge_label"]
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )
    if ge_label.eq("").all():
        return None
    valid = ge_label != ""
    y_true = ge_label[valid]
    y_pred = (
        df.loc[valid, "predicted_class"]
        .astype(str)
        .str.strip()
        .replace({"nan": "unknown", "None": "unknown", "<NA>": "unknown", "": "unknown"})
    )
    labels = sorted([x for x in (set(y_true.tolist()) | set(y_pred.tolist())) if x not in ("", "nan", "None", "<NA>")], key=str)
    if len(labels) == 0:
        return None
    cm = pd.crosstab(y_true, y_pred, dropna=False).reindex(index=labels, columns=labels, fill_value=0)
    oa = np.trace(cm.values) / cm.values.sum() if cm.values.sum() else np.nan
    row_sum = cm.sum(axis=1).values
    col_sum = cm.sum(axis=0).values
    total = cm.values.sum()
    pe = np.sum(row_sum * col_sum) / (total * total) if total else np.nan
    kappa = (oa - pe) / (1 - pe) if total and (1 - pe) != 0 else np.nan
    return cm, oa, kappa


def main():
    pair_outputs = []
    stats_frames = []
    meta_frames = []
    max_workers = min(2, len(PAIRS), CPU_COUNT)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_pair, pair_id, scenes[2021], scenes[2025]): pair_id
            for pair_id, scenes in PAIRS.items()
        }
        for fu in as_completed(futures):
            summary, stats_df, meta_df = fu.result()
            pair_outputs.append(summary)
            stats_frames.append(stats_df)
            meta_frames.append(meta_df)

    metadata_df = pd.concat(meta_frames, ignore_index=True)
    metadata_csv = RESULTS_DIR / f"{DATE_TAG}_metadata_summary.csv"
    metadata_df.to_csv(metadata_csv, index=False, encoding="utf-8")

    dst_transform, dst_shape = build_mosaic_grid(pair_outputs)

    delta_lst = mosaic_result([p["files"]["delta_lst"] for p in pair_outputs], dst_transform, dst_shape)
    delta_ndvi = mosaic_result([p["files"]["delta_ndvi"] for p in pair_outputs], dst_transform, dst_shape)
    uhi_class = mosaic_result([p["files"]["uhi_class"] for p in pair_outputs], dst_transform, dst_shape, method="first")
    lst_2021 = mosaic_result([p["files"]["lst_2021"] for p in pair_outputs], dst_transform, dst_shape)
    lst_2025 = mosaic_result([p["files"]["lst_2025"] for p in pair_outputs], dst_transform, dst_shape)
    signif_lst = mosaic_result([p["files"]["significant_lst_change"] for p in pair_outputs], dst_transform, dst_shape, method="first")
    mad_intensity = mosaic_result([p["files"]["mad_intensity"] for p in pair_outputs], dst_transform, dst_shape)
    mad_mask = mosaic_result([p["files"]["mad_change_mask"] for p in pair_outputs], dst_transform, dst_shape, method="first")

    valid_mask = np.isfinite(delta_lst) & np.isfinite(delta_ndvi)
    area = np.full(dst_shape, (30.0 * 30.0) / 10000.0, dtype=np.float32)

    for code, label in [(0, "NoUHI"), (1, "WeakUHI"), (2, "ModerateUHI"), (3, "StrongUHI")]:
        mask = uhi_class == code
        if np.any(mask):
            stats_frames.append(pd.DataFrame([{"region_name": label, "metric": "UHI_2025", "year": 2025, "area_ha": float(np.nansum(area[mask])), "mean": float(np.nanmean(mask.astype(float))), "std": 0.0}]))

    mosaic_stats = pd.DataFrame(
        [
            {"region_name": "Mosaic", "metric": "LST_C", "year": 2021, "area_ha": float(np.nansum(area[valid_mask])), "mean": float(np.nanmean(lst_2021[valid_mask])), "std": float(np.nanstd(lst_2021[valid_mask]))},
            {"region_name": "Mosaic", "metric": "LST_C", "year": 2025, "area_ha": float(np.nansum(area[valid_mask])), "mean": float(np.nanmean(lst_2025[valid_mask])), "std": float(np.nanstd(lst_2025[valid_mask]))},
            {"region_name": "Mosaic", "metric": "delta_LST_C", "year": "2021_2025", "area_ha": float(np.nansum(area[valid_mask])), "mean": float(np.nanmean(delta_lst[valid_mask])), "std": float(np.nanstd(delta_lst[valid_mask]))},
            {"region_name": "Mosaic", "metric": "delta_NDVI", "year": "2021_2025", "area_ha": float(np.nansum(area[valid_mask])), "mean": float(np.nanmean(delta_ndvi[valid_mask])), "std": float(np.nanstd(delta_ndvi[valid_mask]))},
            {"region_name": "NDVI_Improved", "metric": "AreaSummary", "year": "2021_2025", "area_ha": float(np.nansum(area[valid_mask & (delta_ndvi > 0.05)])), "mean": np.nan, "std": np.nan},
            {"region_name": "NDVI_Degraded", "metric": "AreaSummary", "year": "2021_2025", "area_ha": float(np.nansum(area[valid_mask & (delta_ndvi < -0.05)])), "mean": np.nan, "std": np.nan},
            {"region_name": "LST_Warming_Significant", "metric": "AreaSummary", "year": "2021_2025", "area_ha": float(np.nansum(area[signif_lst == 1])), "mean": np.nan, "std": np.nan},
            {"region_name": "LST_Cooling_Significant", "metric": "AreaSummary", "year": "2021_2025", "area_ha": float(np.nansum(area[signif_lst == -1])), "mean": np.nan, "std": np.nan},
            {"region_name": "MAD_Changed", "metric": "AreaSummary", "year": "2021_2025", "area_ha": float(np.nansum(area[mad_mask == 1])), "mean": np.nan, "std": np.nan},
        ]
    )
    stats_df = pd.concat(stats_frames + [mosaic_stats], ignore_index=True)
    stats_csv = RESULTS_DIR / f"{DATE_TAG}_statistics.csv"
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8")

    profile_wgs84 = {
        "driver": "GTiff",
        "height": dst_shape[0],
        "width": dst_shape[1],
        "count": 1,
        "crs": "EPSG:4326",
        "transform": dst_transform,
        "compress": "lzw",
    }
    save_raster(RESULTS_DIR / f"{DATE_TAG}_delta_lst.tif", delta_lst, profile_wgs84)
    save_raster(RESULTS_DIR / f"{DATE_TAG}_delta_ndvi.tif", delta_ndvi, profile_wgs84)
    uhi_class_i16 = np.where(np.isfinite(uhi_class), uhi_class, -9999).astype(np.int16)
    signif_lst_i16 = np.where(np.isfinite(signif_lst), signif_lst, -9999).astype(np.int16)
    mad_mask_i16 = np.where(np.isfinite(mad_mask), mad_mask, -9999).astype(np.int16)
    save_raster(RESULTS_DIR / f"{DATE_TAG}_uhi_class_2025.tif", uhi_class_i16, profile_wgs84, dtype="int16", nodata=-9999)
    save_raster(RESULTS_DIR / f"{DATE_TAG}_significant_lst_change.tif", signif_lst_i16, profile_wgs84, dtype="int16", nodata=-9999)
    save_raster(RESULTS_DIR / f"{DATE_TAG}_mad_intensity.tif", mad_intensity, profile_wgs84)
    save_raster(RESULTS_DIR / f"{DATE_TAG}_mad_change_mask.tif", mad_mask_i16, profile_wgs84, dtype="int16", nodata=-9999)

    make_map(delta_lst, dst_transform, RESULTS_DIR / f"{DATE_TAG}_map_LST_change.png", "2021-2025 LST Change (°C)", "coolwarm", vmin=-5, vmax=5)
    make_map(delta_ndvi, dst_transform, RESULTS_DIR / f"{DATE_TAG}_map_NDVI_change.png", "2021-2025 NDVI Change", "RdYlGn", vmin=-0.5, vmax=0.5)
    make_map(uhi_class.astype(np.float32), dst_transform, RESULTS_DIR / f"{DATE_TAG}_map_UHI_class_2025.png", "2025 UHI Intensity Classes", "YlOrRd", vmin=0, vmax=3, discrete=True, labels={0: "None", 1: "Weak", 2: "Moderate", 3: "Strong"})

    validation_csv = RESULTS_DIR / f"{DATE_TAG}_validation_points.csv"
    validation_df = sample_validation_points_from_mosaic(delta_ndvi, delta_lst, valid_mask, dst_transform, 100)
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")
    confusion = compute_confusion_if_labeled(validation_csv)

    strong_uhi_area = float(np.nansum(area[uhi_class == 3]))
    total_area = float(np.nansum(area[valid_mask]))
    summary_lines = [
        "# 技术摘要",
        "",
        "## 数据源",
        "- Landsat 8 Collection 2 Level-2 Science Product (L2SP)，每年使用 124/046 与 125/047 两景配对处理。",
        "",
        "## 方法",
        "- 逐 pair 解析 MTL、应用 QA 掩膜、执行同轨道跨年配准、计算 LST/NDVI/EVI/SAVI/UHI/MAD。",
        "- 再将 pair 结果重投影并拼接到 EPSG:4326 输出网格。",
        "",
        "## 关键发现",
        f"- 2021 年 LST 平均值 {np.nanmean(lst_2021[valid_mask]):.2f} °C，2025 年为 {np.nanmean(lst_2025[valid_mask]):.2f} °C，平均变化 {np.nanmean(delta_lst[valid_mask]):.2f} °C。",
        f"- NDVI 改善面积 {np.nansum(area[valid_mask & (delta_ndvi > 0.05)]):.2f} ha，退化面积 {np.nansum(area[valid_mask & (delta_ndvi < -0.05)]):.2f} ha。",
        f"- 2025 年强热岛面积 {strong_uhi_area:.2f} ha，占分析区 {100.0 * strong_uhi_area / total_area:.2f}%。",
        f"- MAD 变化像元面积 {np.nansum(area[mad_mask == 1]):.2f} ha。",
    ]
    if confusion is None:
        summary_lines.append("- 已生成 100 个稳定样本点，需人工在 Google Earth 历史影像中填写标签后计算 OA/Kappa。")
    else:
        _, oa, kappa = confusion
        summary_lines.append(f"- 目视验证结果：OA={oa * 100:.2f}%，Kappa={kappa:.3f}。")
    (RESULTS_DIR / f"{DATE_TAG}_technical_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    readme = """# Landsat Pair Change Analysis

## 运行环境
- Python 3.10+
- rasterio 1.5.0
- numpy 2.4.4
- scipy 1.17.1
- scikit-image 0.26.0
- scikit-learn 1.8.0
- pandas 3.0.2
- matplotlib 3.10.8

## 一键执行
```bash
E:\\Latex\\.venv\\Scripts\\python.exe E:\\Latex\\code\\landsat_pair_change_analysis.py
```

## 输出说明
- `workspace/results/{date}_metadata_summary.csv`
- `workspace/results/{date}_statistics.csv`
- `workspace/results/{date}_technical_summary.md`
- `workspace/results/{date}_map_LST_change.png`
- `workspace/results/{date}_map_NDVI_change.png`
- `workspace/results/{date}_map_UHI_class_2025.png`
- `workspace/results/{date}_validation_points.csv`

## 验证说明
- `validation_points.csv` 里的 `ge_label` 需人工在 Google Earth 历史影像中填写。
- 填写后重新运行脚本，即可自动计算 OA 与 Kappa。

## 方法备注
- 采用 pair 级配准与分析，可避免整岛一次性镶嵌造成的内存压力。
- 输入为两年单时相影像，像元级严格配对 t 检验不可行；此处采用 5x5 邻域 Welch 显著性替代。
"""
    (RESULTS_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "results_dir": str(RESULTS_DIR),
                "metadata_csv": str(metadata_csv),
                "stats_csv": str(stats_csv),
                "validation_csv": str(validation_csv),
                "pairs": {
                    item["pair"]: {
                        "shift_row_px": item["shift_row_px"],
                        "shift_col_px": item["shift_col_px"],
                        "registration_error_metric": item["reg_error"],
                        "analysis_area_ha": item["analysis_area_ha"],
                    }
                    for item in pair_outputs
                },
                "lst_mean_2021": float(np.nanmean(lst_2021[valid_mask])),
                "lst_mean_2025": float(np.nanmean(lst_2025[valid_mask])),
                "lst_delta_mean": float(np.nanmean(delta_lst[valid_mask])),
                "ndvi_improved_ha": float(np.nansum(area[valid_mask & (delta_ndvi > 0.05)])),
                "ndvi_degraded_ha": float(np.nansum(area[valid_mask & (delta_ndvi < -0.05)])),
                "strong_uhi_area_ha": strong_uhi_area,
                "analysis_area_ha": total_area,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
