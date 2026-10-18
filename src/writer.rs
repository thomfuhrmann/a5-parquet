use std::{error::Error, fs::File, io::BufWriter, path::Path, sync::Arc};

use a5::{LonLat, line_string_to_cells, lonlat_to_cell, polygon_to_cells, union};
use arrow::{
    array::{
        ArrayRef, Float64Array, ListArray, ListBuilder, RecordBatch, StructArray, UInt32Array,
        UInt64Builder,
    },
    datatypes::{DataType, Field, Fields},
};
use geojson::{GeoJson, Geometry, Position};
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};

/// Bounding box of a feature
#[derive(Debug, Clone)]
pub struct BBox {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
}

impl BBox {
    /// Create a new bounding box from coordinates
    pub fn new(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Self {
        BBox {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }

    /// Create a new empty bounding box
    pub fn empty() -> Self {
        BBox {
            xmin: f64::MAX,
            ymin: f64::MAX,
            xmax: f64::MIN,
            ymax: f64::MIN,
        }
    }

    /// Extend a bounding box by the given coordinates
    pub fn extend(&mut self, x: f64, y: f64) {
        self.xmin = self.xmin.min(x);
        self.ymin = self.ymin.min(y);
        self.xmax = self.xmax.max(x);
        self.ymax = self.ymax.max(y);
    }

    /// Owning version of extend
    pub fn into_extended(mut self, x: f64, y: f64) -> Self {
        self.extend(x, y);
        self
    }

    /// Extend a bounding box by a `GeoJson::Position`
    pub fn extend_from_pos(&mut self, pos: &Position) {
        self.extend(pos[0], pos[1]);
    }

    /// Owning version
    pub fn into_extended_from_pos(mut self, pos: &Position) -> Self {
        self.extend(pos[0], pos[1]);
        self
    }

    /// Extend with another bounding box
    pub fn extend_from_bbox(&mut self, other: &BBox) {
        self.xmin = self.xmin.min(other.xmin);
        self.ymin = self.ymin.min(other.ymin);
        self.xmax = self.xmax.max(other.xmax);
        self.ymax = self.ymax.max(other.ymax);
    }

    /// Create bounding box from a single point
    pub fn from_point(x: f64, y: f64) -> Self {
        BBox {
            xmin: x,
            ymin: y,
            xmax: x,
            ymax: y,
        }
    }

    /// Calculate the center of the bounding box
    pub fn center(&self) -> (f64, f64) {
        ((self.xmax - self.xmin) / 2.0, (self.ymax - self.ymin) / 2.0)
    }
}

/// Convert the geometry to its bounding box
pub fn geom_to_bbox(geom: &Geometry) -> BBox {
    match &geom.value {
        geojson::GeometryValue::Point { coordinates } => {
            BBox::from_point(coordinates[0], coordinates[1])
        }
        geojson::GeometryValue::MultiPoint { coordinates } => coordinates
            .iter()
            .fold(BBox::empty(), |bbox, pos| bbox.into_extended_from_pos(pos)),
        geojson::GeometryValue::LineString { coordinates } => coordinates
            .iter()
            .fold(BBox::empty(), |bbox, pos| bbox.into_extended_from_pos(pos)),
        geojson::GeometryValue::MultiLineString { coordinates } => {
            coordinates
                .iter()
                .fold(BBox::empty(), |outer_bbox, positions| {
                    positions
                        .iter()
                        .fold(outer_bbox, |bbox, pos| bbox.into_extended_from_pos(pos))
                })
        }
        geojson::GeometryValue::Polygon { coordinates } => {
            coordinates.iter().fold(BBox::empty(), |outer_bbox, ring| {
                ring.iter()
                    .fold(outer_bbox, |bbox, pos| bbox.into_extended_from_pos(pos))
            })
        }
        geojson::GeometryValue::MultiPolygon { coordinates } => {
            coordinates.iter().fold(BBox::empty(), |outer_bbox, poly| {
                poly.iter().fold(outer_bbox, |poly_bbox, ring| {
                    ring.iter().fold(poly_bbox, |ring_bbox, pos| {
                        ring_bbox.into_extended_from_pos(pos)
                    })
                })
            })
        }
        geojson::GeometryValue::GeometryCollection { geometries } => {
            let mut bbox = BBox::empty();
            for geom in geometries {
                let geom_bbox = geom_to_bbox(geom);
                bbox.extend_from_bbox(&geom_bbox);
            }
            bbox
        }
    }
}

/// Convert the geometry to its covering cells
/// Takes the union of multipart geometries
/// Returns the compacted form of the covering
pub fn geom_to_cells(geom: &Geometry, resolution: i32) -> Result<Vec<u64>, Box<dyn Error>> {
    let cells = match &geom.value {
        geojson::GeometryValue::Point { coordinates } => {
            let cell = lonlat_to_cell(LonLat::new(coordinates[0], coordinates[1]), resolution)?;
            vec![cell]
        }
        geojson::GeometryValue::MultiPoint { coordinates } => coordinates
            .iter()
            .flat_map(|coord| lonlat_to_cell(LonLat::new(coord[0], coord[1]), resolution))
            .try_fold(Vec::new(), |acc, cell| union(&acc, &[cell]))?,
        geojson::GeometryValue::LineString { coordinates } => {
            let coords = coordinates
                .iter()
                .map(|coord| LonLat::new(coord[0], coord[1]))
                .collect::<Vec<_>>();
            line_string_to_cells(&coords, resolution)?
        }
        geojson::GeometryValue::MultiLineString { coordinates } => coordinates
            .iter()
            .flat_map(|line_string| {
                let coords = line_string
                    .iter()
                    .map(|coord| LonLat::new(coord[0], coord[1]))
                    .collect::<Vec<_>>();
                line_string_to_cells(&coords, resolution)
            })
            .try_fold(Vec::new(), |acc, cells| union(&acc, &cells))?,
        geojson::GeometryValue::Polygon { coordinates } => {
            let coords = coordinates
                .iter()
                .map(|ring| {
                    ring.iter()
                        .map(|coord| LonLat::new(coord[0], coord[1]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            polygon_to_cells(&coords, resolution, None)?
        }
        geojson::GeometryValue::MultiPolygon { coordinates } => coordinates
            .iter()
            .flat_map(|poly| {
                let coords = poly
                    .iter()
                    .map(|ring| {
                        ring.iter()
                            .map(|coord| LonLat::new(coord[0], coord[1]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                polygon_to_cells(&coords, resolution, None)
            })
            .try_fold(Vec::new(), |acc, cells| union(&acc, &cells))?,
        geojson::GeometryValue::GeometryCollection { geometries } => {
            geometries.iter().try_fold(Vec::new(), |acc, geom| {
                geom_to_cells(geom, resolution)
                    .and_then(|cells| union(&acc, &cells).map_err(|e| e.into()))
            })?
        }
    };

    Ok(cells)
}

/// Calculate the Hilbert index of a point
fn xy_to_hilbert(mut x: u32, mut y: u32) -> u64 {
    let mut n = 1_u32 << 15; // Start at 32768 for 16-bit coordinates (0 to 65535)
    let mut id: u64 = 0;

    while n > 0 {
        let rx = if (x & n) != 0 { 1 } else { 0 };
        let ry = if (y & n) != 0 { 1 } else { 0 };

        id += ((3 * rx) ^ ry) as u64 * ((n as u64) * (n as u64));

        // Rotate/flip quadrant if necessary
        if ry == 0 {
            if rx == 1 {
                // Isolate local coordinates within the current block of size n
                let cx = x & (n - 1);
                let cy = y & (n - 1);

                // Reflect local coordinates safely without underflow
                x = (n - 1) - cx;
                y = (n - 1) - cy;
            }
            std::mem::swap(&mut x, &mut y);
        }

        n >>= 1;
    }
    id
}

/// Sort features by Hilbert index
pub fn sort_geojson(geojson: GeoJson) -> GeoJson {
    match geojson {
        GeoJson::FeatureCollection(mut fc) => {
            let mut bbox = BBox::empty();
            for feat in &fc {
                let feat_bbox = if let Some(geom) = &feat.geometry {
                    geom_to_bbox(geom)
                } else {
                    BBox::empty()
                };
                bbox.extend_from_bbox(&feat_bbox);
            }
            let span_x = if bbox.xmin == bbox.xmax {
                1.0
            } else {
                bbox.xmax - bbox.xmin
            };
            let span_y = if bbox.ymin == bbox.ymax {
                1.0
            } else {
                bbox.ymax - bbox.ymin
            };
            let grid_max = 65535.0; // 16-bit grid space

            fc.features.sort_by_key(|feat| {
                // Normalize coordinates to [0, 65535] u32 grid
                let center = if let Some(geom) = &feat.geometry {
                    geom_to_bbox(geom).center()
                } else {
                    BBox::empty().center()
                };
                let norm_x =
                    (((center.0 - bbox.xmin) / span_x) * grid_max).clamp(0.0, grid_max) as u32;
                let norm_y =
                    (((center.1 - bbox.ymin) / span_y) * grid_max).clamp(0.0, grid_max) as u32;

                xy_to_hilbert(norm_x, norm_y)
            });
            GeoJson::FeatureCollection(fc)
        }
        _ => geojson,
    }
}

/// Convert `GeoJson` to `Parquet`
/// Geometries are converted to A5 coverings
/// Appends one column for bounding boxes
/// Appends one column for the coverings as lists of `u64`s
pub fn geojson_to_parquet<P>(
    path: P,
    geojson: GeoJson,
    resolution: i32,
) -> Result<(), Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let geojson = sort_geojson(geojson);

    let feat_cells = match &geojson {
        GeoJson::Geometry(geom) => vec![geom_to_cells(geom, resolution)?],
        GeoJson::Feature(feat) => {
            if let Some(geom) = &feat.geometry {
                vec![geom_to_cells(geom, resolution)?]
            } else {
                vec![vec![]]
            }
        }
        GeoJson::FeatureCollection(fc) => {
            let mut all_cells = Vec::new();
            for feat in fc {
                let cells = if let Some(geom) = &feat.geometry {
                    geom_to_cells(geom, resolution)?
                } else {
                    Vec::new()
                };
                all_cells.push(cells);
            }
            all_cells
        }
    };

    let feat_bboxes = match &geojson {
        GeoJson::Geometry(geom) => vec![geom_to_bbox(geom)],
        GeoJson::Feature(feat) => {
            if let Some(geom) = &feat.geometry {
                vec![geom_to_bbox(geom)]
            } else {
                vec![BBox::empty()]
            }
        }
        GeoJson::FeatureCollection(fc) => {
            let mut bboxes = vec![];
            for feat in fc {
                let bbox = if let Some(geom) = &feat.geometry {
                    geom_to_bbox(geom)
                } else {
                    BBox::empty()
                };
                bboxes.push(bbox);
            }
            bboxes
        }
    };

    let ids: UInt32Array = feat_cells
        .iter()
        .enumerate()
        .map(|(id, _)| id as u32)
        .collect();

    let inner_builder = UInt64Builder::new();
    let mut builder = ListBuilder::new(inner_builder);
    for cells in feat_cells {
        let values_builder = builder.values();
        for cell in cells {
            values_builder.append_value(cell);
        }
        builder.append(true);
    }

    let vals: ListArray = builder.finish();

    let mut xmin_vals = Vec::with_capacity(feat_bboxes.len());
    let mut ymin_vals = Vec::with_capacity(feat_bboxes.len());
    let mut xmax_vals = Vec::with_capacity(feat_bboxes.len());
    let mut ymax_vals = Vec::with_capacity(feat_bboxes.len());

    for bbox in feat_bboxes {
        xmin_vals.push(bbox.xmin);
        ymin_vals.push(bbox.ymin);
        xmax_vals.push(bbox.xmax);
        ymax_vals.push(bbox.ymax);
    }

    // Convert vectors into trait object array references
    let xmin_array: ArrayRef = Arc::new(Float64Array::from(xmin_vals));
    let ymin_array: ArrayRef = Arc::new(Float64Array::from(ymin_vals));
    let xmax_array: ArrayRef = Arc::new(Float64Array::from(xmax_vals));
    let ymax_array: ArrayRef = Arc::new(Float64Array::from(ymax_vals));

    // Define the schema fields for the struct
    let fields = Fields::from(vec![
        Field::new("xmin", DataType::Float64, false),
        Field::new("ymin", DataType::Float64, false),
        Field::new("xmax", DataType::Float64, false),
        Field::new("ymax", DataType::Float64, false),
    ]);

    // Assemble the StructArray (third parameter is an optional null bitmap buffer)
    let struct_array = StructArray::try_new(
        fields,
        vec![xmin_array, ymin_array, xmax_array, ymax_array],
        None,
    )?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("bbox", Arc::new(struct_array) as ArrayRef),
        ("cells", Arc::new(vals) as ArrayRef),
    ])?;

    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);

    // WriterProperties can be used to set Parquet file options
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(49152))
        .build();

    let mut writer = ArrowWriter::try_new(buf_writer, batch.schema(), Some(props))?;

    writer.write(&batch)?;

    // writer must be closed to write footer
    writer.close().unwrap();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use super::*;

    #[test]
    fn test_write_parquet() {
        let file = File::open("tests/fixtures/countries.geojson").expect("should open");
        let reader = BufReader::new(file);
        let geojson: GeoJson = serde_json::from_reader(reader).expect("failed to parse GeoJSON");

        let _ = geojson_to_parquet("tests/output/countries.parquet", geojson, 10)
            .expect("should write");
    }
}
