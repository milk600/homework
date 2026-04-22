import geopandas as gpd
import matplotlib.pyplot as plt
import os

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def plot_clipped_shapefile(shp_path, output_png):
    """
    加载并绘制 shapefile，保存为 PNG 图片
    """
    print(f"正在加载裁剪后的矢量数据: {shp_path} ...")
    
    if not os.path.exists(shp_path):
        print(f"错误: 找不到文件 {shp_path}")
        return
        
    # 加载数据
    gdf = gpd.read_file(shp_path)
    
    if len(gdf) == 0:
        print("警告: shapefile 中没有要素！")
        return
        
    print(f"数据加载成功，共 {len(gdf)} 个要素。正在生成预览图...")
    
    # 创建绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制多边形
    # facecolor: 填充颜色，这里用绿色表示红树林
    # edgecolor: 边框颜色
    # alpha: 透明度
    gdf.plot(ax=ax, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # 设置标题和坐标轴标签
    ax.set_title("海南省近岸红树林裁剪结果预览", fontsize=16)
    ax.set_xlabel("经度 (Longitude)", fontsize=12)
    ax.set_ylabel("纬度 (Latitude)", fontsize=12)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"预览图已成功保存至: {output_png}")
    
    # 打印一些基本统计信息
    print("\n--- 裁剪数据统计信息 ---")
    print(f"坐标系 (CRS): {gdf.crs}")
    bounds = gdf.total_bounds
    print(f"地理范围 (min_lon, min_lat, max_lon, max_lat):")
    print(f"({bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f})")
    
    # 如果有面积字段，计算总面积 (注意：通常经纬度坐标系下的面积单位是平方度，需要投影转换才能得到准确的平方米/公顷)
    # 这里只是简单展示
    
if __name__ == "__main__":
    # 裁剪后的 shapefile 路径
    SHP_PATH = 'e:/Latex/workspace/data/Hainan_Mangrove_ROI/Hainan_Mangrove_ROI.shp'
    
    # 输出预览图的路径
    OUTPUT_PNG = 'e:/Latex/workspace/data/Hainan_Mangrove_ROI/preview.png'
    
    plot_clipped_shapefile(SHP_PATH, OUTPUT_PNG)
