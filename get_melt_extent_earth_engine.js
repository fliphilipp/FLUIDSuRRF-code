// simplified script based on Pete Tuckett Amery melt paper

// input parameters, change as needed
var assetPath = 'projects/ee-philipparndt/assets/B-C'; // shapefile of basin, imported into EE as an asset
var startDate = '2022-10-01'; // yyyy-10-01 for Antarctic melt season, yyyy-04-01 for Greenland melt season
var endDate = '2023-03-31'; // yyyy-03-31 for Antarctic melt season, yyyy-10-15 for Greenland melt season

var sunElev = 20; 
var cloudCover = 50;
var ndwi_thresh = 0.19;

var pathlist = assetPath.split('/');
var name_region = pathlist[pathlist.length - 1];
var start_year = startDate.split('-')[0];
var end_year = endDate.split('-')[0];
var descriptor = name_region + '_extent_' + start_year + '_' + end_year.slice(2);
print(descriptor);

var L7T1 = ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA");
var L7T2 = ee.ImageCollection("LANDSAT/LE07/C02/T2_TOA");
var L8T1 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA');
var L8T2 = ee.ImageCollection('LANDSAT/LC08/C02/T2_TOA');
var L9T1 = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA');
var L9T2 = ee.ImageCollection('LANDSAT/LC09/C02/T2_TOA');

var L8bands = ['B2','B3','B4','B6', 'B10'];
var L8bandnames = ['blue','green','red','swir','tir'];
var roi = ee.FeatureCollection(assetPath);

var L8_coll = L7T1.merge(L7T2).merge(L8T1).merge(L8T2).merge(L9T2).merge(L9T2)
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .sort('system:time_start')
  .filterMetadata('SUN_ELEVATION', 'greater_than', sunElev)
  .filter(ee.Filter.lt('CLOUD_COVER', cloudCover))
  .select(L8bands)
  .map(function(image){
    return image.rename(L8bandnames).clip(roi); 
  });
  
var addRockMask = function(image){
  var diff = image.select('tir').divide(image.select('blue'));
  var blue = image.select('blue');
  var RockSeawater = diff.gt(650).and(blue.lt(0.35));
  var RockSeaMask = RockSeawater.neq(1);
  var RockSeaMaskv2 = RockSeaMask.updateMask(RockSeaMask).rename('RockSeaMask');
  var RockSeaArea = RockSeawater.eq(1);
  var RockSeaAreav2 = RockSeaArea.updateMask(RockSeaArea).rename('RockSeaArea');

  var SWIR = image.select('swir');
  var NDSI = image.normalizedDifference(['green', 'swir']);
  var Cloud = NDSI.lt(0.8).and(SWIR.gt(0.1));
  var CloudMask = Cloud.neq(1);
  var CloudMaskv2 = CloudMask.updateMask(CloudMask).rename('VisibleAreas');
  var CloudArea = Cloud.eq(1);
  var CloudAreav2 = CloudArea.updateMask(CloudArea).rename('CloudArea');
  
  return image.updateMask(RockSeaMaskv2).updateMask(CloudMaskv2)
    .addBands(RockSeaAreav2).addBands(CloudMaskv2).addBands(CloudAreav2);
};

var addLakeMask = function(image){
  var BLUE = image.select('blue');
  var GREEN = image.select('green');
  var RED = image.select('red');
  var NDWI = image.normalizedDifference(['blue', 'red']).rename('ndwi');
  var Lakes = NDWI.gt(ndwi_thresh).and((GREEN.subtract(RED)).gt(0.10)).and((BLUE.subtract(GREEN)).gt(0.11)); // 0.10, 0.11
  var LakeMask = Lakes.updateMask(Lakes).rename('LakeMask');
  
  return image.addBands(NDWI).addBands(LakeMask);
};


L8_coll = L8_coll.map(addRockMask).map(addLakeMask);
var watermask = L8_coll.select('LakeMask').max();
print(L8_coll, 'L8_collection');
print(watermask, 'Water Mask');

// Reduce to vectors
var min_area = 0.009; // km^2  (10 pixels)
var reduce_scale = ee.Number(min_area).multiply(1e6).sqrt().multiply(0.8).round();
var water_vectors = watermask.reduceToVectors({scale: reduce_scale, geometry: roi, maxPixels:1e13}); 
var err = ee.ErrorMargin(reduce_scale);

// Get lake areas
var addArea = function(feature) {
  var lakeArea = ee.Number(feature.geometry(err).area(err).divide(1e6));
  return feature.set({areaKm2: lakeArea});
};

water_vectors = water_vectors.map(addArea);
// print('total lake patches',water_vectors.size());

var lakes_filtered = water_vectors.filter(ee.Filter.gt('areaKm2', min_area));
// print('filtered lake patches',lakes_filtered.size());
// print(lakes_filtered, 'lake collection');

var empty = ee.Image().byte();
var filledOutlines = empty.paint(lakes_filtered).paint(lakes_filtered, 0, 1).clip(roi);

var roi_viz = {color: 'black', fillColor: '00000000'};
Map.addLayer(watermask, {palette: ['red']}, 'filtered lakes mask');
// Map.addLayer(filledOutlines, {palette: ['blue']}, 'filtered lakes vectors');

Map.addLayer(roi.style(roi_viz),null,'Region Of Interest');
Map.centerObject(roi, 6);

Export.table.toDrive({
  collection: lakes_filtered,
  description:descriptor,
  folder:"Continent_wide_GEOJSONs", // If wanting to save GeoJson directly into this folder
  fileFormat: 'GEOJSON'
});

