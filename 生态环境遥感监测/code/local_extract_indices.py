import os
import glob
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd

def calculate_indices_local(landsat_dir, shp_path, output_dir, year):
    """
    在本地离线计算 RSEI 的四个分量 (绿度、湿度、干度、热度)
    
    Args:
        landsat_dir: 存放某一年 Landsat 8/9 Level-2 影像解压后文件的目录
        shp_path: 裁剪用的掩膜 shapefile 路径
        output_dir: 输出目录
        year: 年份标识
    """
    print(f"\n正在处理 {year} 年的数据...")
    
    # 1. 自动寻找各波段的 tif 文件 (Landsat 8/9 C02 T1_L2 命名规则)
    # B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2, B10=Thermal
    bands = {
        'B2': glob.glob(os.path.join(landsat_dir, '*_SR_B2.TIF')),
        'B3': glob.glob(os.path.join(landsat_dir, '*_SR_B3.TIF')),
        'B4': glob.glob(os.path.join(landsat_dir, '*_SR_B4.TIF')),
        'B5': glob.glob(os.path.join(landsat_dir, '*_SR_B5.TIF')),
        'B6': glob.glob(os.path.join(landsat_dir, '*_SR_B6.TIF')),
        'B7': glob.glob(os.path.join(landsat_dir, '*_SR_B7.TIF')),
        'B10': glob.glob(os.path.join(landsat_dir, '*_ST_B10.TIF'))
    }
    
    # 检查是否所有必需波段都存在
    for band_name, files in bands.items():
        if not files:
            print(f"错误: 在目录 {landsat_dir} 中找不到波段 {band_name}")
            return
    
    print("已找到所有必需波段文件。加载掩膜矢量...")
    gdf = gpd.read_file(shp_path)
    geom = [shapes for shapes in gdf.geometry]
    
    def read_and_clip(file_path):
        """读取并使用矢量裁剪栅格"""
        with rasterio.open(file_path) as src:
            # 检查坐标系是否一致，如果不一致需要先转换
            if gdf.crs != src.crs:
                gdf_proj = gdf.to_crs(src.crs)
                geoms = [shapes for shapes in gdf_proj.geometry]
            else:
                geoms = geom
                
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
            
            # 更新元数据
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # 转换为浮点型以避免计算溢出
            data = out_image[0].astype('float32')
            
            # Landsat 8/9 Collection 2 Level 2 的缩放因子 (Scale Factor)
            # 光学波段: multiply 0.0000275 add -0.2
            # 热红外波段: multiply 0.00341802 add 149.0
            if 'SR_B' in file_path:
                data = data * 0.0000275 - 0.2
            elif 'ST_B10' in file_path:
                data = data * 0.00341802 + 149.0
                
            # 将无效值(nodata)设为 np.nan
            data[data <= -0.2] = np.nan # 简单过滤背景值
            return data, out_meta

    print("正在裁剪并应用缩放因子...")
    b2, meta = read_and_clip(bands['B2'][0])
    b3, _ = read_and_clip(bands['B3'][0])
    b4, _ = read_and_clip(bands['B4'][0])
    b5, _ = read_and_clip(bands['B5'][0])
    b6, _ = read_and_clip(bands['B6'][0])
    b7, _ = read_and_clip(bands['B7'][0])
    b10, _ = read_and_clip(bands['B10'][0])
    
    print("正在计算生态指标...")
    # 忽略除以 0 的警告
    np.seterr(divide='ignore', invalid='ignore')
    
    # 1. 绿度 (NDVI)
    ndvi = (b5 - b4) / (b5 + b4)
    
    # 2. 湿度 (Wetness - TCW)
    wetness = (0.1511*b2 + 0.1973*b3 + 0.3283*b4 + 0.3407*b5 - 0.7117*b6 - 0.4559*b7)
    
    # 3. 干度 (NDBSI = (SI + IBI) / 2)
    # SI
    si = ((b6 + b4) - (b5 + b2)) / ((b6 + b4) + (b5 + b2))
    # IBI
    ibi = (2 * b6 / (b6 + b5) - (b5 / (b5 + b4) + b3 / (b3 + b6))) / \
          (2 * b6 / (b6 + b5) + (b5 / (b5 + b4) + b3 / (b3 + b6)))
    ndbsi = (si + ibi) / 2.0
    
    # 4. 热度 (LST - 转换为摄氏度)
    lst = b10 - 273.15
    
    print("保存计算结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 将四个波段堆叠为一个多波段 TIFF
    meta.update(count=4, dtype='float32')
    out_path = os.path.join(output_dir, f'Mangrove_Indices_{year}.tif')
    
    with rasterio.open(out_path, 'w', **meta) as dest:
        dest.write(ndvi, 1)     # Band 1: NDVI
        dest.write(wetness, 2)  # Band 2: Wetness
        dest.write(ndbsi, 3)    # Band 3: Dryness
        dest.write(lst, 4)      # Band 4: LST
        
    print(f"{year} 年数据处理完成！保存在: {out_path}")
    print("波段顺序: 1-绿度(NDVI), 2-湿度(Wetness), 3-干度(NDBSI), 4-热度(LST)")

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 裁剪用的红树林矢量边界路径
    SHP_PATH = 'e:/Latex/workspace/data/Hainan_Mangrove_ROI/Hainan_Mangrove_ROI.shp'
    
    # 存放最终 TIFF 结果的文件夹
    OUTPUT_DIR = 'e:/Latex/workspace/output/Indices'
    
    # 【注意】
    # 由于您放弃了 GEE，您必须去地理空间数据云或 USGS 官网手动下载 Landsat 8/9 数据。
    # 将下载好的压缩包解压到以下对应的文件夹中。
    # 文件夹内应包含形如 LC08_..._SR_B2.TIF 的文件。
    
    # 这里以 2021 年为例（请将路径修改为您实际解压影像的路径）
    LANDSAT_2021_DIR = 'e:/Latex/workspace/data/Landsat_2021' 
    
    # ============================================
    
    if os.path.exists(LANDSAT_2021_DIR):
        calculate_indices_local(LANDSAT_2021_DIR, SHP_PATH, OUTPUT_DIR, 2021)
    else:
        print("提示：请先下载 Landsat 8/9 数据并解压到指定目录，然后修改脚本中的文件夹路径。")
