import json
import math
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
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


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATE_TAG = datetime.now().strftime("%Y%m%d")
ROOT = Path(r"e:\Latex")
DATA_DIR = ROOT / "workspace" / "data"
RESULTS_DIR = ROOT / "workspace" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENES = {
    2021: [
        DATA_DIR / "LC08_L2SP_124046_20210101_20210308_02_T1",
        DATA_DIR / "LC08_L2SP_125047_20210124_20210305_02_T1",
    ],
    2025: [
        DATA_DIR / "LC08_L2SP_124046_20250112_20250122_02_T1",
        DATA_DIR / "LC08_L2SP_125047_20250103_20250111_02_T1",
    ],
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


def decode_quality_masks(scene_dir: Path):
    qa_pixel_path = scene_band_path(scene_dir, "QA_PIXEL")
    qa_radsat_path = scene_band_path(scene_dir, "QA_RADSAT")
    with rasterio.open(qa_pixel_path) as src:
        qa_pixel = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
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
    return good.astype(np.uint8), land.astype(np.uint8), src_transform, src_crs


def reproject_to_grid(src_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs, resampling):
    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def reproject_mask(src_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    dst = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=src_arr.astype(np.uint8),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst.astype(bool)


def build_target_grid(scene_dirs_by_year):
    year_bounds = {}
    for year, scene_dirs in scene_dirs_by_year.items():
        bounds_list = []
        for scene_dir in scene_dirs:
            b4_path = scene_band_path(scene_dir, "SR_B4")
            with rasterio.open(b4_path) as src:
                bounds_list.append(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
        xmin = min(b[0] for b in bounds_list)
        ymin = min(b[1] for b in bounds_list)
        xmax = max(b[2] for b in bounds_list)
        ymax = max(b[3] for b in bounds_list)
        year_bounds[year] = (xmin, ymin, xmax, ymax)

    common_bounds = (
        max(year_bounds[2021][0], year_bounds[2025][0]),
        max(year_bounds[2021][1], year_bounds[2025][1]),
        min(year_bounds[2021][2], year_bounds[2025][2]),
        min(year_bounds[2021][3], year_bounds[2025][3]),
    )
    width = int(math.ceil((common_bounds[2] - common_bounds[0]) / RES_DEG))
    height = int(math.ceil((common_bounds[3] - common_bounds[1]) / RES_DEG))
    transform = from_origin(common_bounds[0], common_bounds[3], RES_DEG, RES_DEG)
    return common_bounds, transform, height, width


def mosaic_year(scene_dirs, target_transform, target_shape, target_crs="EPSG:4326"):
    accum = {band: np.zeros(target_shape, dtype=np.float32) for band in BANDS_SR + ["ST_B10"]}
    count = {band: np.zeros(target_shape, dtype=np.uint16) for band in BANDS_SR + ["ST_B10"]}
    qa_good_accum = np.zeros(target_shape, dtype=np.uint16)
    land_accum = np.zeros(target_shape, dtype=np.uint16)
    metadata_rows = []

    for scene_dir in scene_dirs:
        mtl = parse_mtl(next(scene_dir.glob("*_MTL.txt")))
        metadata_rows.append(
            {
                "scene": scene_dir.name,
                "acquisition_date": mtl.get("DATE_ACQUIRED"),
                "sun_elevation": float(mtl.get("SUN_ELEVATION")),
                "sun_azimuth": float(mtl.get("SUN_AZIMUTH")),
                "sr_mult": float(mtl.get("REFLECTANCE_MULT_BAND_4")),
                "sr_add": float(mtl.get("REFLECTANCE_ADD_BAND_4")),
                "st_mult": float(mtl.get("TEMPERATURE_MULT_BAND_ST_B10")),
                "st_add": float(mtl.get("TEMPERATURE_ADD_BAND_ST_B10")),
                "utm_zone": mtl.get("UTM_ZONE"),
                "cloud_cover": float(mtl.get("CLOUD_COVER")),
                "cloud_cover_land": float(mtl.get("CLOUD_COVER_LAND")),
                "geom_rmse_model_m": float(mtl.get("GEOMETRIC_RMSE_MODEL", "nan")),
            }
        )

        good_raw, land_raw, qa_transform, qa_crs = decode_quality_masks(scene_dir)
        good_mask = reproject_mask(good_raw, qa_transform, qa_crs, target_shape, target_transform, target_crs)
        land_mask = reproject_mask(land_raw, qa_transform, qa_crs, target_shape, target_transform, target_crs)
        qa_good_accum += good_mask.astype(np.uint16)
        land_accum += land_mask.astype(np.uint16)

        for band in BANDS_SR + ["ST_B10"]:
            suffix = f"SR_{band}" if band != "ST_B10" else "ST_B10"
            band_path = scene_band_path(scene_dir, suffix)
            with rasterio.open(band_path) as src:
                arr = src.read(1).astype(np.float32)
                src_transform = src.transform
                src_crs = src.crs
            arr[arr == 0] = np.nan
            if band.startswith("B"):
                band_num = band.replace("B", "")
                arr = arr * float(mtl[f"REFLECTANCE_MULT_BAND_{band_num}"]) + float(mtl[f"REFLECTANCE_ADD_BAND_{band_num}"])
            else:
                arr = arr * float(mtl["TEMPERATURE_MULT_BAND_ST_B10"]) + float(mtl["TEMPERATURE_ADD_BAND_ST_B10"])

            arr_proj = reproject_to_grid(arr, src_transform, src_crs, target_shape, target_transform, target_crs, Resampling.bilinear)
            arr_proj[~good_mask] = np.nan
            valid = np.isfinite(arr_proj)
            accum[band][valid] += arr_proj[valid]
            count[band][valid] += 1

    mosaic = {}
    for band in accum:
        out = np.full(target_shape, np.nan, dtype=np.float32)
        valid = count[band] > 0
        out[valid] = accum[band][valid] / count[band][valid]
        mosaic[band] = out

    mosaic["qa_good_count"] = qa_good_accum.astype(np.float32)
    mosaic["land_presence"] = land_accum > 0
    return mosaic, pd.DataFrame(metadata_rows)


def save_raster(path: Path, arr, transform, crs="EPSG:4326", nodata=np.nan, dtype="float32"):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "nodata": nodata if not (isinstance(nodata, float) and np.isnan(nodata)) else None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def estimate_shift(base, moving, mask):
    if np.isfinite(base[mask]).sum() < 1000 or np.isfinite(moving[mask]).sum() < 1000:
        return (0.0, 0.0), np.nan
    a = np.where(mask, base, np.nan)
    b = np.where(mask, moving, np.nan)
    a_fill = np.where(np.isfinite(a), a, np.nanmedian(a))
    b_fill = np.where(np.isfinite(b), b, np.nanmedian(b))
    shift, error, _ = phase_cross_correlation(a_fill, b_fill, upsample_factor=10)
    return (float(shift[0]), float(shift[1])), float(error)


def apply_shift(arr, shift):
    arr_filled = np.where(np.isfinite(arr), arr, 0.0)
    shifted = ndimage.shift(arr_filled, shift=shift, order=1, mode="nearest")
    valid = ndimage.shift(np.isfinite(arr).astype(np.float32), shift=shift, order=0, mode="nearest") > 0.5
    shifted[~valid] = np.nan
    return shifted.astype(np.float32)


def compute_indices(mosaic):
    blue = mosaic["B2"]
    red = mosaic["B4"]
    nir = mosaic["B5"]
    swir1 = mosaic["B6"]
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1.0)
    savi = 1.5 * (nir - red) / (nir + red + 0.5)
    ndbi = (swir1 - nir) / (swir1 + nir)
    lst_c = mosaic["ST_B10"] - 273.15
    return {
        "NDVI": ndvi.astype(np.float32),
        "EVI": evi.astype(np.float32),
        "SAVI": savi.astype(np.float32),
        "NDBI": ndbi.astype(np.float32),
        "LST_C": lst_c.astype(np.float32),
    }


def pixel_area_ha(transform, shape):
    _, _, _, ymax = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    rows = np.arange(shape[0])
    lats = np.abs(ymax - (rows + 0.5) * abs(transform.e))
    dy_m = abs(transform.e) * 111320.0
    dx_m = abs(transform.a) * 111320.0 * np.cos(np.deg2rad(lats))
    area_row = (dx_m * dy_m) / 10000.0
    return np.repeat(area_row[:, None], shape[1], axis=1).astype(np.float32)


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
    return tstat.astype(np.float32), p.astype(np.float32)


def compute_mad(mosaic2021, mosaic2025, valid_mask):
    x_stack = np.stack([mosaic2021[b] for b in BANDS_SR], axis=-1)
    y_stack = np.stack([mosaic2025[b] for b in BANDS_SR], axis=-1)
    valid_idx = np.where(valid_mask.ravel())[0]
    x = x_stack.reshape(-1, len(BANDS_SR))[valid_idx]
    y = y_stack.reshape(-1, len(BANDS_SR))[valid_idx]
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[finite]
    y = y[finite]
    valid_idx = valid_idx[finite]

    xs = StandardScaler().fit_transform(x)
    ys = StandardScaler().fit_transform(y)
    cca = CCA(n_components=len(BANDS_SR), max_iter=1000)
    u, v = cca.fit_transform(xs, ys)
    mad = u - v
    mad_std = np.nanstd(mad, axis=0)
    mad_std[mad_std == 0] = 1.0
    intensity = np.sqrt(np.sum((mad / mad_std) ** 2, axis=1)).astype(np.float32)

    out = np.full(valid_mask.size, np.nan, dtype=np.float32)
    out[valid_idx] = intensity
    out = out.reshape(valid_mask.shape)
    thr = float(np.nanmean(intensity) + 2 * np.nanstd(intensity))
    change = np.where(out > thr, 1, 0).astype(np.int16)
    change[~valid_mask] = -9999
    return out, change, thr


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
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - (ymax - ymin) * 0.08),
        arrowprops=dict(facecolor="black", width=3, headwidth=10),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )


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


def sample_validation_points(valid_mask, stable_mask, transform, n=100, seed=42):
    rng = np.random.default_rng(seed)
    candidate = np.argwhere(valid_mask & stable_mask)
    if len(candidate) == 0:
        return pd.DataFrame(columns=["point_id", "lon", "lat", "predicted_class", "ge_label"])
    chosen = candidate[rng.choice(len(candidate), size=min(n, len(candidate)), replace=False)]
    rows = []
    for i, (r, c) in enumerate(chosen, start=1):
        x, y = rasterio.transform.xy(transform, r, c, offset="center")
        rows.append({"point_id": i, "lon": x, "lat": y, "predicted_class": "stable", "ge_label": ""})
    return pd.DataFrame(rows)


def compute_confusion_if_labeled(points_csv: Path):
    if not points_csv.exists():
        return None
    df = pd.read_csv(points_csv)
    if "ge_label" not in df.columns or df["ge_label"].astype(str).str.strip().eq("").all():
        return None
    y_true = df["ge_label"].astype(str)
    y_pred = df["predicted_class"].astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    cm = pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, labels), dropna=False)
    oa = np.trace(cm.values) / cm.values.sum() if cm.values.sum() else np.nan
    row_sum = cm.sum(axis=1).values
    col_sum = cm.sum(axis=0).values
    total = cm.values.sum()
    pe = np.sum(row_sum * col_sum) / (total * total) if total else np.nan
    kappa = (oa - pe) / (1 - pe) if total and (1 - pe) != 0 else np.nan
    return cm, oa, kappa


def main():
    common_bounds, transform, height, width = build_target_grid(SCENES)
    shape = (height, width)

    mosaics = {}
    metadata_frames = []
    for year, scene_dirs in SCENES.items():
        mosaic, meta_df = mosaic_year(scene_dirs, transform, shape)
        mosaics[year] = mosaic
        meta_df.insert(0, "year", year)
        metadata_frames.append(meta_df)

    metadata_df = pd.concat(metadata_frames, ignore_index=True)
    metadata_csv = RESULTS_DIR / f"{DATE_TAG}_metadata_summary.csv"
    metadata_df.to_csv(metadata_csv, index=False, encoding="utf-8-sig")

    common_mask = np.ones(shape, dtype=bool)
    for year in [2021, 2025]:
        for band in BANDS_SR + ["ST_B10"]:
            common_mask &= np.isfinite(mosaics[year][band])
        common_mask &= mosaics[year]["land_presence"]

    shift, reg_error = estimate_shift(mosaics[2021]["B4"], mosaics[2025]["B4"], common_mask)
    for band in BANDS_SR + ["ST_B10"]:
        mosaics[2025][band] = apply_shift(mosaics[2025][band], shift)

    common_mask = np.ones(shape, dtype=bool)
    for year in [2021, 2025]:
        for band in BANDS_SR + ["ST_B10"]:
            common_mask &= np.isfinite(mosaics[year][band])
        common_mask &= mosaics[year]["land_presence"]

    idx2021 = compute_indices(mosaics[2021])
    idx2025 = compute_indices(mosaics[2025])
    delta_lst = idx2025["LST_C"] - idx2021["LST_C"]
    delta_ndvi = idx2025["NDVI"] - idx2021["NDVI"]
    delta_evi = idx2025["EVI"] - idx2021["EVI"]
    delta_savi = idx2025["SAVI"] - idx2021["SAVI"]

    _, pvals = local_ttest(idx2021["LST_C"], idx2025["LST_C"], common_mask, size=5)
    signif_change = np.full(shape, -9999, dtype=np.int16)
    signif_change[(pvals < 0.05) & (delta_lst > 0)] = 1
    signif_change[(pvals < 0.05) & (delta_lst < 0)] = -1
    signif_change[(pvals >= 0.05) & common_mask] = 0

    rural_mask = common_mask & (idx2025["NDVI"] > 0.5) & (idx2025["NDBI"] < 0)
    rural_mean_lst = float(np.nanmean(idx2025["LST_C"][rural_mask])) if np.any(rural_mask) else float(np.nanmean(idx2025["LST_C"][common_mask]))
    uhi_intensity = idx2025["LST_C"] - rural_mean_lst
    uhi_class = np.full(shape, -9999, dtype=np.int16)
    uhi_class[(uhi_intensity <= 0) & common_mask] = 0
    uhi_class[(uhi_intensity > 0) & (uhi_intensity <= 2)] = 1
    uhi_class[(uhi_intensity > 2) & (uhi_intensity <= 4)] = 2
    uhi_class[uhi_intensity > 4] = 3

    mad_intensity, mad_change, mad_thr = compute_mad(mosaics[2021], mosaics[2025], common_mask)

    pixel_area = pixel_area_ha(transform, shape)
    imp_mask = common_mask & (delta_ndvi > 0.05)
    deg_mask = common_mask & (delta_ndvi < -0.05)

    stats_rows = []

    def add_basic_stats(region_name, metric, arr2021, arr2025, delta):
        area_ha = float(np.nansum(pixel_area[common_mask]))
        stats_rows.append({"region_name": region_name, "metric": metric, "year": 2021, "area_ha": area_ha, "mean": float(np.nanmean(arr2021[common_mask])), "std": float(np.nanstd(arr2021[common_mask]))})
        stats_rows.append({"region_name": region_name, "metric": metric, "year": 2025, "area_ha": area_ha, "mean": float(np.nanmean(arr2025[common_mask])), "std": float(np.nanstd(arr2025[common_mask]))})
        stats_rows.append({"region_name": region_name, "metric": f"delta_{metric}", "year": "2021_2025", "area_ha": area_ha, "mean": float(np.nanmean(delta[common_mask])), "std": float(np.nanstd(delta[common_mask]))})

    add_basic_stats("CommonOverlap", "LST_C", idx2021["LST_C"], idx2025["LST_C"], delta_lst)
    add_basic_stats("CommonOverlap", "NDVI", idx2021["NDVI"], idx2025["NDVI"], delta_ndvi)
    add_basic_stats("CommonOverlap", "EVI", idx2021["EVI"], idx2025["EVI"], delta_evi)
    add_basic_stats("CommonOverlap", "SAVI", idx2021["SAVI"], idx2025["SAVI"], delta_savi)

    for code, label in [(0, "NoUHI"), (1, "WeakUHI"), (2, "ModerateUHI"), (3, "StrongUHI")]:
        mask = uhi_class == code
        if np.any(mask):
            stats_rows.append({"region_name": label, "metric": "UHI_2025", "year": 2025, "area_ha": float(np.nansum(pixel_area[mask])), "mean": float(np.nanmean(uhi_intensity[mask])), "std": float(np.nanstd(uhi_intensity[mask]))})

    for label, mask in [("NDVI_Improved", imp_mask), ("NDVI_Degraded", deg_mask), ("LST_Warming_Significant", signif_change == 1), ("LST_Cooling_Significant", signif_change == -1), ("MAD_Changed", mad_change == 1)]:
        if np.any(mask):
            stats_rows.append({"region_name": label, "metric": "AreaSummary", "year": "2021_2025", "area_ha": float(np.nansum(pixel_area[mask])), "mean": np.nan, "std": np.nan})

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = RESULTS_DIR / f"{DATE_TAG}_statistics.csv"
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")

    raster_outputs = {
        "lst_2021": idx2021["LST_C"],
        "lst_2025": idx2025["LST_C"],
        "delta_lst": delta_lst,
        "ndvi_2021": idx2021["NDVI"],
        "ndvi_2025": idx2025["NDVI"],
        "delta_ndvi": delta_ndvi,
        "evi_2021": idx2021["EVI"],
        "evi_2025": idx2025["EVI"],
        "savi_2021": idx2021["SAVI"],
        "savi_2025": idx2025["SAVI"],
        "uhi_2025": uhi_intensity,
        "mad_intensity": mad_intensity,
        "qa_good_count_2021": mosaics[2021]["qa_good_count"],
        "qa_good_count_2025": mosaics[2025]["qa_good_count"],
    }
    for name, arr in raster_outputs.items():
        save_raster(RESULTS_DIR / f"{DATE_TAG}_{name}.tif", np.where(common_mask | name.startswith("qa_"), arr, np.nan), transform)

    save_raster(RESULTS_DIR / f"{DATE_TAG}_uhi_class_2025.tif", uhi_class.astype(np.int16), transform, nodata=-9999, dtype="int16")
    save_raster(RESULTS_DIR / f"{DATE_TAG}_significant_lst_change.tif", signif_change.astype(np.int16), transform, nodata=-9999, dtype="int16")
    save_raster(RESULTS_DIR / f"{DATE_TAG}_mad_change_mask.tif", mad_change.astype(np.int16), transform, nodata=-9999, dtype="int16")

    make_map(delta_lst, transform, RESULTS_DIR / f"{DATE_TAG}_map_LST_change.png", "2021-2025 LST Change (°C)", "coolwarm", vmin=-5, vmax=5)
    make_map(delta_ndvi, transform, RESULTS_DIR / f"{DATE_TAG}_map_NDVI_change.png", "2021-2025 NDVI Change", "RdYlGn", vmin=-0.5, vmax=0.5)
    make_map(uhi_class.astype(np.float32), transform, RESULTS_DIR / f"{DATE_TAG}_map_UHI_class_2025.png", "2025 UHI Intensity Classes", "YlOrRd", vmin=0, vmax=3, discrete=True, labels={0: "None", 1: "Weak", 2: "Moderate", 3: "Strong"})

    validation_csv = RESULTS_DIR / f"{DATE_TAG}_validation_points.csv"
    validation_df = sample_validation_points(common_mask, (np.abs(delta_ndvi) < 0.05) & (np.abs(delta_lst) < 1.0), transform, 100)
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8-sig")
    confusion = compute_confusion_if_labeled(validation_csv)

    strong_uhi_area = float(np.nansum(pixel_area[uhi_class == 3]))
    total_area = float(np.nansum(pixel_area[common_mask]))
    summary_lines = [
        "# 技术摘要",
        "",
        "## 数据源",
        "- Landsat 8 Collection 2 Level-2 Science Product (L2SP)，包含 Surface Reflectance、Surface Temperature、QA_PIXEL、QA_RADSAT。",
        "- 时相：2021 年 1 月与 2025 年 1 月，两年各用 124/046 与 125/047 两景拼接成年度镶嵌。",
        "",
        "## 方法",
        "- 解析 MTL 提取 acquisition date、太阳高度角、缩放参数与几何 RMSE。",
        "- 通过 QA_PIXEL 与 QA_RADSAT 构建高质量像元掩膜，剔除填充值、扩张云、卷云、云、云影、雪和辐射饱和像元。",
        "- 使用官方 L2SP 缩放参数将 SR 波段转为物理反射率、ST_B10 转为地表温度（K），统一到 EPSG:4326 的 30m 等效网格。",
        "- 通过相位相关估计 2025 相对 2021 的亚像素位移，在公共区域计算 LST、NDVI、EVI、SAVI、UHI 与 MAD 变化强度。",
        "",
        "## 关键发现",
        f"- 估计配准平移量：row={shift[0]:.3f} px，col={shift[1]:.3f} px；相位相关误差指标={reg_error:.4f}。",
        f"- 2021 年 LST 平均值 {np.nanmean(idx2021['LST_C'][common_mask]):.2f} °C，标准差 {np.nanstd(idx2021['LST_C'][common_mask]):.2f} °C。",
        f"- 2025 年 LST 平均值 {np.nanmean(idx2025['LST_C'][common_mask]):.2f} °C，标准差 {np.nanstd(idx2025['LST_C'][common_mask]):.2f} °C。",
        f"- 2021→2025 平均温度变化 {np.nanmean(delta_lst[common_mask]):.2f} °C。",
        f"- NDVI 改善面积 {np.nansum(pixel_area[imp_mask]):.2f} ha，退化面积 {np.nansum(pixel_area[deg_mask]):.2f} ha。",
        f"- 2025 年强热岛面积 {strong_uhi_area:.2f} ha，占分析区 {100.0 * strong_uhi_area / total_area:.2f}%。",
        f"- MAD 变化强度阈值（均值+2σ）为 {mad_thr:.2f}，变化像元面积 {np.nansum(pixel_area[mad_change == 1]):.2f} ha。",
    ]
    if confusion is None:
        summary_lines.append("- 已生成 100 个稳定样本点，但 Google Earth 历史影像人工标签尚未填写，因此 OA/Kappa 需在人工标注后再计算。")
    else:
        _, oa, kappa = confusion
        summary_lines.append(f"- 目视验证结果：OA={oa * 100:.2f}%，Kappa={kappa:.3f}。")

    (RESULTS_DIR / f"{DATE_TAG}_technical_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    readme = f"""# Landsat Change Analysis Results

## 运行环境
- Python 3.10+
- 关键依赖：rasterio 1.5.0, numpy 2.4.4, scipy 1.17.1, scikit-image 0.26.0, scikit-learn 1.8.0, geopandas 1.1.3, pandas 3.0.2, matplotlib 3.10.8
- 说明：脚本使用 rasterio（GDAL backend）完成栅格 IO 与重投影，无需单独调用 osgeo.gdal。

## 一键执行
```bash
E:\\Latex\\.venv\\Scripts\\python.exe E:\\Latex\\code\\landsat_change_analysis.py
```

## 输入数据
- 2021: LC08_L2SP_124046_20210101_20210308_02_T1, LC08_L2SP_125047_20210124_20210305_02_T1
- 2025: LC08_L2SP_124046_20250112_20250122_02_T1, LC08_L2SP_125047_20250103_20250111_02_T1

## 输出内容
- GeoTIFF: LST/NDVI/EVI/SAVI/UHI/MAD 结果及变化图
- CSV: 元数据摘要、统计表、验证点
- PNG: 300 dpi 专题地图
- Markdown: 技术摘要

## 方法说明
1. 解析 MTL 并读取官方缩放参数。
2. 通过 QA_PIXEL 与 QA_RADSAT 构建高质量像元掩膜。
3. 将四景数据重投影到统一的 EPSG:4326 30m 等效网格并按年份镶嵌。
4. 用相位相关估计 2025 相对 2021 的亚像素位移，并对 2025 年影像做平移校正。
5. 计算 LST、NDVI、EVI、SAVI、UHI 与 MAD 变化强度。
6. 输出 GeoTIFF、CSV 与专题图。

## 验证说明
- `*_validation_points.csv` 已提供 100 个稳定样本点，请在 Google Earth 历史影像中填写 `ge_label` 列。
- 填写后重新运行脚本即可自动计算混淆矩阵、OA 和 Kappa。

## 备注
- 输入为两年单时相影像，像元级严格“配对 t 检验”不具备重复观测条件；脚本采用 5x5 邻域 Welch t 显著性图替代，用于识别显著升温/降温空间簇。
- 本次反射率使用 L2SP 官方 Surface Reflectance 缩放参数；若需 TOA 反射率，应下载对应 L1 影像并重新定标。
"""
    (RESULTS_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "results_dir": str(RESULTS_DIR),
                "metadata_csv": str(metadata_csv),
                "stats_csv": str(stats_csv),
                "validation_csv": str(validation_csv),
                "shift_row_px": shift[0],
                "shift_col_px": shift[1],
                "registration_error_metric": reg_error,
                "lst_mean_2021": float(np.nanmean(idx2021["LST_C"][common_mask])),
                "lst_mean_2025": float(np.nanmean(idx2025["LST_C"][common_mask])),
                "lst_delta_mean": float(np.nanmean(delta_lst[common_mask])),
                "ndvi_improved_ha": float(np.nansum(pixel_area[imp_mask])),
                "ndvi_degraded_ha": float(np.nansum(pixel_area[deg_mask])),
                "strong_uhi_area_ha": strong_uhi_area,
                "analysis_area_ha": total_area,
                "mad_threshold": mad_thr,
                "common_bounds": common_bounds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
