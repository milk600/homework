from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject


ROOT = Path(r"e:\Latex")
RESULTS_DIR = ROOT / "workspace" / "results"
ROI_PATH = ROOT / "workspace" / "data" / "Hainan_Mangrove_ROI" / "Hainan_Mangrove_ROI.shp"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_TAG = "20260421"
PIXEL_AREA_HA = (30.0 * 30.0) / 10000.0


def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, profile, transform, crs


def save_raster(path: Path, arr, profile, dtype="float32", nodata=np.nan):
    out_arr = arr.copy()
    if np.issubdtype(np.dtype(dtype), np.integer):
        fill_value = nodata if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)) else -9999
        out_arr = np.where(np.isfinite(out_arr), out_arr, fill_value)
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype=dtype,
        compress="lzw",
        nodata=nodata if not (isinstance(nodata, float) and np.isnan(nodata)) else None,
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out_arr.astype(dtype), 1)


def build_roi_mask(shape, transform, crs):
    gdf = gpd.read_file(ROI_PATH)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return geometry_mask(geoms, transform=transform, invert=True, out_shape=shape)


def mosaic_to_reference(files, ref_shape, ref_transform, ref_crs, discrete=False):
    if discrete:
        out = np.full(ref_shape, np.nan, dtype=np.float32)
    else:
        sum_arr = np.zeros(ref_shape, dtype=np.float32)
        count_arr = np.zeros(ref_shape, dtype=np.uint16)

    for path in files:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            dst = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(
                source=arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.nearest if discrete else Resampling.bilinear,
                num_threads=8,
            )
            if discrete:
                fill = np.isnan(out) & np.isfinite(dst)
                out[fill] = dst[fill]
            else:
                valid = np.isfinite(dst)
                sum_arr[valid] += dst[valid]
                count_arr[valid] += 1

    if discrete:
        return out

    out = np.full(ref_shape, np.nan, dtype=np.float32)
    valid = count_arr > 0
    out[valid] = sum_arr[valid] / count_arr[valid]
    return out


def crop_to_valid_extent(arr, pad=20):
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    rows, cols = np.where(valid)
    r0 = max(int(rows.min()) - pad, 0)
    r1 = min(int(rows.max()) + pad + 1, arr.shape[0])
    c0 = max(int(cols.min()) - pad, 0)
    c1 = min(int(cols.max()) + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def downsample_for_display(arr, max_side=1600):
    step = max(1, int(np.ceil(max(arr.shape) / max_side)))
    return arr[::step, ::step]


def prepare_sparse_plot(arr, max_points=60000):
    arr = crop_to_valid_extent(arr)
    rows, cols = np.where(np.isfinite(arr))
    values = arr[rows, cols]
    if values.size == 0:
        return arr, None, None, None
    if values.size > max_points:
        step = int(np.ceil(values.size / max_points))
        rows = rows[::step]
        cols = cols[::step]
        values = values[::step]
    return arr, rows, cols, values


def make_triptych(arr1, arr2, delta, out_png: Path, titles, cmaps, vmins=None, vmaxs=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    arrays = [arr1, arr2, delta]
    for i, ax in enumerate(axes):
        cropped, rows, cols, values = prepare_sparse_plot(arrays[i])
        ax.set_facecolor("white")
        if values is None:
            im = ax.imshow(np.full((10, 10), np.nan), cmap=cmaps[i], vmin=None if vmins is None else vmins[i], vmax=None if vmaxs is None else vmaxs[i])
        else:
            im = ax.scatter(
                cols,
                rows,
                c=values,
                cmap=cmaps[i],
                vmin=None if vmins is None else vmins[i],
                vmax=None if vmaxs is None else vmaxs[i],
                s=10,
                marker="s",
                linewidths=0,
            )
            ax.set_xlim(-5, cropped.shape[1] + 5)
            ax.set_ylim(cropped.shape[0] + 5, -5)
            ax.set_aspect("equal")
        ax.set_title(titles[i], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def make_single_map(arr, out_png: Path, title, cmap, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    cropped, rows, cols, values = prepare_sparse_plot(arr)
    ax.set_facecolor("white")
    if values is None:
        im = ax.imshow(np.full((10, 10), np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.scatter(cols, rows, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=10, marker="s", linewidths=0)
        ax.set_xlim(-5, cropped.shape[1] + 5)
        ax.set_ylim(cropped.shape[0] + 5, -5)
        ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def make_summary_bars(summary_df: pd.DataFrame, out_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)

    metric_means = summary_df[summary_df["metric"].isin(["LST_C", "NDVI", "EVI", "SAVI"])]
    pivot = metric_means.pivot(index="metric", columns="year", values="mean").reindex(["LST_C", "NDVI", "EVI", "SAVI"])
    x = np.arange(len(pivot.index))
    width = 0.35
    axes[0].bar(x - width / 2, pivot[2021], width=width, label="2021")
    axes[0].bar(x + width / 2, pivot[2025], width=width, label="2025")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pivot.index)
    axes[0].set_title("Indicators Mean Comparison")
    axes[0].legend()

    areas = summary_df[summary_df["region_name"].isin(["NDVI_Improved_Mangrove", "NDVI_Degraded_Mangrove", "MAD_Changed_Mangrove"])]
    axes[1].bar(areas["region_name"], areas["area_ha"], color=["green", "firebrick", "slateblue"])
    axes[1].set_title("Area Statistics (ha)")
    axes[1].tick_params(axis="x", rotation=20)

    uhi = summary_df[summary_df["metric"] == "UHI_2025"].copy()
    axes[2].bar(uhi["region_name"].str.replace("_Mangrove", "", regex=False), uhi["area_ha"], color=["#9ecae1", "#fdd870", "#fdae61", "#d73027"])
    axes[2].set_title("UHI Class Area (ha)")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    delta_lst_path = RESULTS_DIR / f"{DATE_TAG}_delta_lst.tif"
    delta_ndvi_path = RESULTS_DIR / f"{DATE_TAG}_delta_ndvi.tif"
    uhi_class_path = RESULTS_DIR / f"{DATE_TAG}_uhi_class_2025.tif"
    signif_path = RESULTS_DIR / f"{DATE_TAG}_significant_lst_change.tif"
    mad_intensity_path = RESULTS_DIR / f"{DATE_TAG}_mad_intensity.tif"
    mad_mask_path = RESULTS_DIR / f"{DATE_TAG}_mad_change_mask.tif"

    delta_lst, ref_profile, ref_transform, ref_crs = read_raster(delta_lst_path)
    delta_ndvi, _, _, _ = read_raster(delta_ndvi_path)
    uhi_class, _, _, _ = read_raster(uhi_class_path)
    signif_lst, _, _, _ = read_raster(signif_path)
    mad_intensity, _, _, _ = read_raster(mad_intensity_path)
    mad_mask, _, _, _ = read_raster(mad_mask_path)

    lst_2021_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_lst_2021.tif"))
    lst_2025_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_lst_2025.tif"))
    ndvi_2021_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_ndvi_2021.tif"))
    ndvi_2025_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_ndvi_2025.tif"))
    evi_2021_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_evi_2021.tif"))
    evi_2025_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_evi_2025.tif"))
    savi_2021_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_savi_2021.tif"))
    savi_2025_files = sorted(RESULTS_DIR.glob(f"{DATE_TAG}_*/{DATE_TAG}_*_savi_2025.tif"))
    if not all([lst_2021_files, lst_2025_files, ndvi_2021_files, ndvi_2025_files, evi_2021_files, evi_2025_files, savi_2021_files, savi_2025_files]):
        raise FileNotFoundError("Missing pair-level indicator outputs for secondary review.")

    lst_2021 = mosaic_to_reference(lst_2021_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    lst_2025 = mosaic_to_reference(lst_2025_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    ndvi_2021 = mosaic_to_reference(ndvi_2021_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    ndvi_2025 = mosaic_to_reference(ndvi_2025_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    evi_2021 = mosaic_to_reference(evi_2021_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    evi_2025 = mosaic_to_reference(evi_2025_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    savi_2021 = mosaic_to_reference(savi_2021_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    savi_2025 = mosaic_to_reference(savi_2025_files, delta_lst.shape, ref_transform, ref_crs, discrete=False)
    delta_evi = evi_2025 - evi_2021
    delta_savi = savi_2025 - savi_2021

    roi_mask = build_roi_mask(delta_lst.shape, ref_transform, ref_crs)
    valid_mask = (
        roi_mask
        & np.isfinite(delta_lst)
        & np.isfinite(delta_ndvi)
        & np.isfinite(lst_2021)
        & np.isfinite(lst_2025)
        & np.isfinite(ndvi_2021)
        & np.isfinite(ndvi_2025)
        & np.isfinite(evi_2021)
        & np.isfinite(evi_2025)
        & np.isfinite(savi_2021)
        & np.isfinite(savi_2025)
    )
    if not np.any(valid_mask):
        raise RuntimeError("No valid pixels remain inside the mangrove ROI.")

    area = np.full(delta_lst.shape, PIXEL_AREA_HA, dtype=np.float32)

    clipped_dir = OUTPUT_DIR / "mangrove_secondary_review"
    clipped_dir.mkdir(parents=True, exist_ok=True)
    clip_profile = ref_profile.copy()

    def clip_arr(arr, fill_value=np.nan):
        clipped = np.full(arr.shape, fill_value, dtype=np.float32)
        clipped[roi_mask] = arr[roi_mask]
        return clipped

    save_raster(clipped_dir / f"{DATE_TAG}_lst_2021_mangrove_clip.tif", clip_arr(lst_2021), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_lst_2025_mangrove_clip.tif", clip_arr(lst_2025), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_delta_lst_mangrove_clip.tif", clip_arr(delta_lst), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_ndvi_2021_mangrove_clip.tif", clip_arr(ndvi_2021), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_ndvi_2025_mangrove_clip.tif", clip_arr(ndvi_2025), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_delta_ndvi_mangrove_clip.tif", clip_arr(delta_ndvi), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_evi_2021_mangrove_clip.tif", clip_arr(evi_2021), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_evi_2025_mangrove_clip.tif", clip_arr(evi_2025), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_delta_evi_mangrove_clip.tif", clip_arr(delta_evi), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_savi_2021_mangrove_clip.tif", clip_arr(savi_2021), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_savi_2025_mangrove_clip.tif", clip_arr(savi_2025), clip_profile)
    save_raster(clipped_dir / f"{DATE_TAG}_delta_savi_mangrove_clip.tif", clip_arr(delta_savi), clip_profile)
    save_raster(
        clipped_dir / f"{DATE_TAG}_uhi_class_2025_mangrove_clip.tif",
        np.where(roi_mask, uhi_class, -9999),
        clip_profile,
        dtype="int16",
        nodata=-9999,
    )
    save_raster(
        clipped_dir / f"{DATE_TAG}_significant_lst_change_mangrove_clip.tif",
        np.where(roi_mask, signif_lst, -9999),
        clip_profile,
        dtype="int16",
        nodata=-9999,
    )
    save_raster(clipped_dir / f"{DATE_TAG}_mad_intensity_mangrove_clip.tif", clip_arr(mad_intensity), clip_profile)
    save_raster(
        clipped_dir / f"{DATE_TAG}_mad_change_mask_mangrove_clip.tif",
        np.where(roi_mask, mad_mask, -9999),
        clip_profile,
        dtype="int16",
        nodata=-9999,
    )

    rows = [
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "LST_C",
            "year": 2021,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(lst_2021[valid_mask])),
            "std": float(np.nanstd(lst_2021[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "LST_C",
            "year": 2025,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(lst_2025[valid_mask])),
            "std": float(np.nanstd(lst_2025[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "delta_LST_C",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(delta_lst[valid_mask])),
            "std": float(np.nanstd(delta_lst[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "NDVI",
            "year": 2021,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(ndvi_2021[valid_mask])),
            "std": float(np.nanstd(ndvi_2021[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "NDVI",
            "year": 2025,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(ndvi_2025[valid_mask])),
            "std": float(np.nanstd(ndvi_2025[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "delta_NDVI",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(delta_ndvi[valid_mask])),
            "std": float(np.nanstd(delta_ndvi[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "EVI",
            "year": 2021,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(evi_2021[valid_mask])),
            "std": float(np.nanstd(evi_2021[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "EVI",
            "year": 2025,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(evi_2025[valid_mask])),
            "std": float(np.nanstd(evi_2025[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "delta_EVI",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(delta_evi[valid_mask])),
            "std": float(np.nanstd(delta_evi[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "SAVI",
            "year": 2021,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(savi_2021[valid_mask])),
            "std": float(np.nanstd(savi_2021[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "SAVI",
            "year": 2025,
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(savi_2025[valid_mask])),
            "std": float(np.nanstd(savi_2025[valid_mask])),
        },
        {
            "region_name": "Hainan_Mangrove_ROI",
            "metric": "delta_SAVI",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[valid_mask])),
            "mean": float(np.nanmean(delta_savi[valid_mask])),
            "std": float(np.nanstd(delta_savi[valid_mask])),
        },
        {
            "region_name": "NDVI_Improved_Mangrove",
            "metric": "AreaSummary",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[valid_mask & (delta_ndvi > 0.05)])),
            "mean": np.nan,
            "std": np.nan,
        },
        {
            "region_name": "NDVI_Degraded_Mangrove",
            "metric": "AreaSummary",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[valid_mask & (delta_ndvi < -0.05)])),
            "mean": np.nan,
            "std": np.nan,
        },
        {
            "region_name": "LST_Warming_Significant_Mangrove",
            "metric": "AreaSummary",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[(signif_lst == 1) & roi_mask])),
            "mean": np.nan,
            "std": np.nan,
        },
        {
            "region_name": "LST_Cooling_Significant_Mangrove",
            "metric": "AreaSummary",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[(signif_lst == -1) & roi_mask])),
            "mean": np.nan,
            "std": np.nan,
        },
        {
            "region_name": "MAD_Changed_Mangrove",
            "metric": "AreaSummary",
            "year": "2021_2025",
            "area_ha": float(np.nansum(area[(mad_mask == 1) & roi_mask])),
            "mean": np.nan,
            "std": np.nan,
        },
    ]

    for code, label in [(0, "NoUHI"), (1, "WeakUHI"), (2, "ModerateUHI"), (3, "StrongUHI")]:
        mask = (uhi_class == code) & roi_mask
        rows.append(
            {
                "region_name": f"{label}_Mangrove",
                "metric": "UHI_2025",
                "year": 2025,
                "area_ha": float(np.nansum(area[mask])),
                "mean": float(np.nanmean(np.where(mask, uhi_class, np.nan))),
                "std": float(np.nanstd(np.where(mask, uhi_class, np.nan))),
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_path = OUTPUT_DIR / "海南岛红树林二次汇总复核统计.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8")

    make_triptych(
        clip_arr(lst_2021),
        clip_arr(lst_2025),
        clip_arr(delta_lst),
        clipped_dir / f"{DATE_TAG}_fig_lst_triptych.png",
        ["LST 2021", "LST 2025", "Delta LST"],
        ["YlOrRd", "YlOrRd", "coolwarm"],
        [18, 18, -2],
        [24, 24, 4],
    )
    make_triptych(
        clip_arr(ndvi_2021),
        clip_arr(ndvi_2025),
        clip_arr(delta_ndvi),
        clipped_dir / f"{DATE_TAG}_fig_ndvi_triptych.png",
        ["NDVI 2021", "NDVI 2025", "Delta NDVI"],
        ["YlGn", "YlGn", "RdYlGn"],
        [0.1, 0.1, -0.2],
        [0.6, 0.6, 0.2],
    )
    make_triptych(
        clip_arr(evi_2021),
        clip_arr(evi_2025),
        clip_arr(delta_evi),
        clipped_dir / f"{DATE_TAG}_fig_evi_triptych.png",
        ["EVI 2021", "EVI 2025", "Delta EVI"],
        ["YlGn", "YlGn", "RdYlGn"],
        [0.05, 0.05, -0.15],
        [0.5, 0.5, 0.15],
    )
    make_triptych(
        clip_arr(savi_2021),
        clip_arr(savi_2025),
        clip_arr(delta_savi),
        clipped_dir / f"{DATE_TAG}_fig_savi_triptych.png",
        ["SAVI 2021", "SAVI 2025", "Delta SAVI"],
        ["YlGn", "YlGn", "RdYlGn"],
        [0.05, 0.05, -0.15],
        [0.5, 0.5, 0.15],
    )
    make_single_map(
        np.where(roi_mask, uhi_class, np.nan),
        clipped_dir / f"{DATE_TAG}_fig_uhi_class.png",
        "UHI Class 2025",
        "YlOrRd",
        0,
        3,
    )
    make_single_map(
        clip_arr(mad_intensity),
        clipped_dir / f"{DATE_TAG}_fig_mad_intensity.png",
        "MAD Intensity",
        "magma",
    )
    make_summary_bars(stats_df, clipped_dir / f"{DATE_TAG}_fig_summary_bars.png")

    summary_lines = [
        "# 海南岛红树林二次汇总复核说明",
        "",
        "## 是否有必要",
        "- 有必要。二次汇总复核的作用是把最终统计结果重新限制到红树林掩膜范围内，避免把掩膜外公共覆盖区域一并计入总量。",
        "- 对论文写作而言，这一步可以显著提高“研究区严格限定为海南岛红树林生态系统”这一表述的严谨性。",
        "",
        "## 本次复核方式",
        "- 以 `Hainan_Mangrove_ROI.shp` 作为唯一研究边界。",
        "- 对已生成的顶层 GeoTIFF 结果重新构建掩膜内统计，并输出掩膜内裁切栅格与复核统计表。",
        "",
        "## 掩膜内复核结果",
        f"- 红树林掩膜内 LST 2021 均值：{np.nanmean(lst_2021[valid_mask]):.2f} °C。",
        f"- 红树林掩膜内 LST 2025 均值：{np.nanmean(lst_2025[valid_mask]):.2f} °C。",
        f"- 红树林掩膜内平均温度变化：{np.nanmean(delta_lst[valid_mask]):.2f} °C。",
        f"- 红树林掩膜内平均 ΔNDVI：{np.nanmean(delta_ndvi[valid_mask]):.4f}。",
        f"- 红树林掩膜内 EVI 2021/2025 均值：{np.nanmean(evi_2021[valid_mask]):.4f} / {np.nanmean(evi_2025[valid_mask]):.4f}。",
        f"- 红树林掩膜内 SAVI 2021/2025 均值：{np.nanmean(savi_2021[valid_mask]):.4f} / {np.nanmean(savi_2025[valid_mask]):.4f}。",
        f"- 红树林掩膜内 NDVI 改善面积：{np.nansum(area[valid_mask & (delta_ndvi > 0.05)]):.2f} ha。",
        f"- 红树林掩膜内 NDVI 退化面积：{np.nansum(area[valid_mask & (delta_ndvi < -0.05)]):.2f} ha。",
        f"- 红树林掩膜内显著升温面积：{np.nansum(area[(signif_lst == 1) & roi_mask]):.2f} ha。",
        f"- 红树林掩膜内强热岛面积：{np.nansum(area[(uhi_class == 3) & roi_mask]):.2f} ha。",
        f"- 红树林掩膜内 MAD 变化面积：{np.nansum(area[(mad_mask == 1) & roi_mask]):.2f} ha。",
        "",
        "## 输出文件",
        f"- 复核统计表：`{stats_path}`",
        f"- 掩膜裁切结果目录：`{clipped_dir}`",
        f"- 论文图表目录：`{clipped_dir}`",
    ]
    summary_path = OUTPUT_DIR / "海南岛红树林二次汇总复核说明.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(
        {
            "stats_csv": str(stats_path),
            "summary_md": str(summary_path),
            "clip_dir": str(clipped_dir),
            "analysis_area_ha": float(np.nansum(area[valid_mask])),
            "lst_mean_2021": float(np.nanmean(lst_2021[valid_mask])),
            "lst_mean_2025": float(np.nanmean(lst_2025[valid_mask])),
            "delta_lst_mean": float(np.nanmean(delta_lst[valid_mask])),
            "delta_ndvi_mean": float(np.nanmean(delta_ndvi[valid_mask])),
            "evi_mean_2021": float(np.nanmean(evi_2021[valid_mask])),
            "evi_mean_2025": float(np.nanmean(evi_2025[valid_mask])),
            "savi_mean_2021": float(np.nanmean(savi_2021[valid_mask])),
            "savi_mean_2025": float(np.nanmean(savi_2025[valid_mask])),
        }
    )


if __name__ == "__main__":
    main()
