//! Clustering algorithms.
//!
//! This crate provides generic implementations of clustering
//! algorithms, allowing them to work with any back-end "point
//! database" that implements the required operations, e.g. one might
//! be happy with using the naive collection `BruteScan` from this
//! crate, or go all out and implement a specialised R*-tree for
//! optimised performance.
//!
//! # Installation
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! cogset = "0.1"
//! ```

mod dbscan;
pub use dbscan::Dbscan;

mod point;
pub use point::{Point, RegionQuery, Points, ListPoints, BruteScan, BruteScanNeighbours, Euclid};
