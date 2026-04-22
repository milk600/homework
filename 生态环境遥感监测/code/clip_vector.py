import geopandas as gpd
from shapely.geometry import box
import os

def clip_shapefile(input_shp, output_shp, bbox):
    """
    使用 bounding box 裁剪 shapefile
    
    Args:
        input_shp: 输入的原始 shapefile 路径
        output_shp: 输出的裁剪后 shapefile 路径
        bbox: 裁剪范围 (minx, miny, maxx, maxy)
    """
    print(f"正在加载原始矢量数据: {input_shp} ...")
    # 加载原始数据
    gdf = gpd.read_file(input_shp)
    
    print(f"原始数据包含 {len(gdf)} 个要素。")
    print(f"使用范围 {bbox} 进行裁剪...")
    
    # 创建裁剪多边形
    bbox_polygon = box(*bbox)
    
    # 执行裁剪 (使用 cx 可以基于边界框快速过滤，然后用 clip 精确裁剪)
    # 先用 cx 粗筛，提高效率
    gdf_filtered = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
    # 再用 clip 精确裁剪 (如果多边形跨越边界，会被切断)
    # 如果只想保留完全在 bbox 内或者与 bbox 相交的完整多边形，可以不用 clip，直接保存 gdf_filtered
    # 这里我们使用精确裁剪，只保留在框内的部分
    clipped_gdf = gpd.clip(gdf_filtered, bbox_polygon)
    
    print(f"裁剪后包含 {len(clipped_gdf)} 个要素。")
    
    if len(clipped_gdf) > 0:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_shp), exist_ok=True)
        
        print(f"正在保存裁剪后的矢量数据到: {output_shp} ...")
        clipped_gdf.to_file(output_shp)
        print("裁剪完成！")
    else:
        print("警告: 裁剪范围内没有发现任何要素，未生成输出文件。")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 替换为您下载并解压后的 GMW shapefile 路径
    INPUT_SHP = 'e:/Latex/workspace/data/gmw_mng_2020_v4019_vec/gmw_mng_2020_v4019.shp' 
    
    # 2. 裁剪后保存的路径
    OUTPUT_SHP = 'e:/Latex/workspace/output/Hainan_Mangrove_ROI/Hainan_Mangrove_ROI.shp'
    
    # 3. 设置海南岛（或特定保护区，如东寨港）的经纬度范围 (min_lon, min_lat, max_lon, max_lat)
    # 东寨港大概范围: (110.50, 19.85, 110.70, 20.05)
    # 海南全岛大概范围: (108.50, 18.10, 111.10, 20.20)
    BBOX = (108.50, 18.10, 111.10, 20.20) 
    
    # ============================================
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_SHP):
        print(f"错误: 找不到输入文件 {INPUT_SHP}")
        print("请确保您已经下载解压了数据，并将 INPUT_SHP 修改为实际的路径。")
    else:
        clip_shapefile(INPUT_SHP, OUTPUT_SHP, BBOX)