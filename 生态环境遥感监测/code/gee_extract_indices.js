// 海南省近岸红树林生态指标(RSEI/FVC)云端自动化提取脚本
// 平台: Google Earth Engine (GEE)
// 语言: JavaScript

// ================== 1. 导入数据 ==================
// 请将下面这行代码替换为您在 GEE Assets 中上传的红树林边界 (Hainan_Mangrove_ROI.shp)
// 例如: var roi = ee.FeatureCollection('users/yourname/Hainan_Mangrove_ROI');
var roi = ee.FeatureCollection('users/yourname/Hainan_Mangrove_ROI'); 

// ================== 2. 定义函数 ==================

// 去云函数 (基于 Landsat 8/9 C02 T1_L2 QA_PIXEL 波段)
function maskL89sr(image) {
  var qa = image.select('QA_PIXEL');
  // 云(bit 3) 和 云影(bit 4) 的掩码
  var cloudShadowBitMask = (1 << 4);
  var cloudsBitMask = (1 << 3);
  // 获取高质量像素
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
    .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  
  // 辐射定标：将缩放的反射率和温度转换为真实物理量
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true)
              .updateMask(mask)
              .clip(roi); // 裁剪到研究区
}

// 计算生态指标函数
function calculateIndices(image) {
  // 提取波段
  var b = image.select('SR_B2'); // Blue
  var g = image.select('SR_B3'); // Green
  var r = image.select('SR_B4'); // Red
  var nir = image.select('SR_B5'); // NIR
  var swir1 = image.select('SR_B6'); // SWIR1
  var swir2 = image.select('SR_B7'); // SWIR2
  
  // 1. 绿度 (Greenness): NDVI
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  
  // 2. 湿度 (Wetness): TCW (基于 OLI 传感器的系数)
  var wetness = image.expression(
    '0.1511*B + 0.1973*G + 0.3283*R + 0.3407*NIR - 0.7117*SWIR1 - 0.4559*SWIR2', {
      'B': b, 'G': g, 'R': r, 'NIR': nir, 'SWIR1': swir1, 'SWIR2': swir2
    }).rename('Wetness');
    
  // 3. 干度 (Dryness): NDBSI = (SI + IBI) / 2
  // SI (裸土指数)
  var si = image.expression(
    '((SWIR1 + R) - (NIR + B)) / ((SWIR1 + R) + (NIR + B))', {
      'SWIR1': swir1, 'R': r, 'NIR': nir, 'B': b
    });
  // IBI (建筑指数)
  var ibi = image.expression(
    '(2 * SWIR1 / (SWIR1 + NIR) - (NIR / (NIR + R) + G / (G + SWIR1))) / (2 * SWIR1 / (SWIR1 + NIR) + (NIR / (NIR + R) + G / (G + SWIR1)))', {
      'SWIR1': swir1, 'NIR': nir, 'R': r, 'G': g
    });
  var ndbsi = si.add(ibi).divide(2).rename('Dryness');
  
  // 4. 热度 (Heat): LST (这里直接使用经过缩放处理的表面温度波段)
  // 将开尔文(K)转换为摄氏度(°C)
  var lst = image.select('ST_B10').subtract(273.15).rename('LST');

  // 将所有计算好的指数合并回原影像，并剔除水体 (如果需要，可以通过MNDWI剔除)
  return image.addBands([ndvi, wetness, ndbsi, lst]);
}

// ================== 3. 主程序：按年度处理并导出 ==================

// 设定年份列表 (2021 - 2025)
var years = [2021, 2022, 2023, 2024, 2025];

// 遍历每一年
years.forEach(function(year) {
  var startDate = year + '-01-01';
  var endDate = year + '-12-31';
  
  // 获取 Landsat 8 和 9 的集合，去云并合并
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
             .filterBounds(roi)
             .filterDate(startDate, endDate)
             .map(maskL89sr);
             
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
             .filterBounds(roi)
             .filterDate(startDate, endDate)
             .map(maskL89sr);
             
  var collection = l8.merge(l9);
  
  // 取年度中位数合成，并计算指标
  var annualComposite = collection.median();
  var finalImage = calculateIndices(annualComposite);
  
  // 提取我们需要导出的四个波段
  var exportImage = finalImage.select(['NDVI', 'Wetness', 'Dryness', 'LST']);
  
  // 导出到 Google Drive
  Export.image.toDrive({
    image: exportImage,
    description: 'Mangrove_Indices_' + year,
    folder: 'GEE_Mangrove_Export', // 您 Drive 里的文件夹名称
    fileNamePrefix: 'Mangrove_Indices_' + year,
    region: roi.geometry().bounds(),
    scale: 30, // Landsat 分辨率
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});

print('任务已提交到 Tasks 面板！请点击 "Run" 按钮开始导出。');
