/// A point in some (metric) space.
pub trait Point {
    /// Accurate compute the distance from `self` to `other`.
    ///
    /// This should be real and non-negative, or else algorithms may
    /// return unexpected results.
    fn dist(&self, other: &Self) -> f64;

    /// Compute an estimate of the distance from `self` to `other`.
    ///
    /// This should be less than or equal to `self.dist(other)` or
    /// else algorithms may return unexpected results.
    #[allow(unused_variables)]
    fn dist_lower_bound(&self, other: &Self) -> f64 {
        ::std::f64::NEG_INFINITY
    }
}

/// A data structure that contains points of some sort.
pub trait Points {
    /// The representation of a point.
    ///
    /// It is expected that this should be stable for the given
    /// instance of a `Points`, but doesn't need to be stable across
    /// instances, e.g. if some hypothetical structure `Foo` stores
    /// points of type `struct Bar(f64, f64)` and two different `Foo`s
    /// store the point `Bar(0.0, 1.0)`, there's no requirement for
    /// the `Point` representing those `Bar`s to be identical. In
    /// particular, this allows one to use an index into a vector of
    /// points stored inside the `Self` type.
    type Point;
}
/// Collections of points that can be queried to find nearby points.
pub trait RegionQuery: Points {
    /// An iterator over the nearby points of a given one.
    type Neighbours: Iterator<Item = Self::Point>;

    /// Return an iterator over points in `self` with distance from
    /// `point` less than or equal to `epsilon`.
    ///
    /// It is expected that this includes `point` itself if `epsilon
    /// >= 0`.
    ///
    /// The behaviour is unspecified if `point` is not in `self`, and
    /// if `epsilon < 0` (although it is strongly recommended that the
    /// iterator is empty in this case).
    fn neighbours(&self, point: &Self::Point, epsilon: f64) -> Self::Neighbours;
}

/// Collections of points that can list everything they contain.
pub trait ListPoints: Points {
    /// An iterator over all the points in an instance of `Self`
    type AllPoints: Iterator<Item = Self::Point>;

    /// Return an iterator over all points in `self`.
    ///
    /// It is expected that this iterator isn't invalidated by calling
    /// other methods like `RegionQuery::neighbours` (if the type also
    /// implements that).
    fn all_points(&self) -> Self::AllPoints;
}

use std::ops::Range;
use std::slice::Iter;
use std::iter::Enumerate;

/// A point collection where queries are answered via brute-force
/// scans over the whole list.
///
/// Points are represented via their indices into the list passed to
/// `new`.
pub struct BruteScan<'a, P: Point + 'a> {
    points: &'a [P]
}

impl<'a, P: Point> BruteScan<'a, P> {
    /// Create a new `BruteScan`.
    pub fn new(p: &'a [P]) -> BruteScan<'a, P> {
        BruteScan { points: p }
    }
}
impl<'a,P: Point> Points for BruteScan<'a, P> {
    type Point = usize;
}
impl<'a,P: Point> ListPoints for BruteScan<'a, P> {
    type AllPoints = Range<usize>;
    fn all_points(&self) -> Range<usize> {
        0..self.points.len()
    }
}

impl<'a, P: Point> RegionQuery for BruteScan<'a, P> {
    type Neighbours = BruteScanNeighbours<'a, P>;

    fn neighbours(&self, point: &usize, eps: f64) -> BruteScanNeighbours<'a, P> {
        BruteScanNeighbours {
            points: self.points.iter().enumerate(),
            point: &self.points[*point],
            eps: eps
        }
    }
}

/// An iterator over the neighbours of a point in a `BruteScan`.
pub struct BruteScanNeighbours<'a, P: Point + 'a> {
    points: Enumerate<Iter<'a, P>>,
    point: &'a P,
    eps: f64,
}

impl<'a,P: Point> Iterator for BruteScanNeighbours<'a, P> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let BruteScanNeighbours { ref mut points, ref point, eps } = *self;

        points.find(|&(_, p)| {
            point.dist_lower_bound(p) <= eps
                && point.dist(p) <= eps
        }).map(|(i, _)| i)
    }
}

/// Points in ℝ<sup><i>n</i></sup> with the <i>L</i><sup>2</sup> norm.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct Euclid<T>(pub T);

macro_rules! euclidean_points {
    ($($e: expr),*) => {
        $(
            impl Point for Euclid<[f64; $e]> {
                fn dist(&self, other: &Euclid<[f64; $e]>) -> f64 {
                    self.0.iter().zip(other.0.iter())
                        .map(|(a, b)| {
                            let d = *a - *b;
                            d * d
                        })
                        .fold(0.0, |a, b| a + b)
                        .sqrt()
                }

                fn dist_lower_bound(&self, other: &Euclid<[f64; $e]>) -> f64 {
                    (self.0[0] - other.0[0]).abs()
                }
            }
            )*
    }
}
euclidean_points!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclid() {
        let p1 = Euclid([0.0, 1.0]);
        let p2 = Euclid([1.0, 0.0]);
        assert!((p1.dist(&p2) - 2f64.sqrt()).abs() < 1e-10);
    }
    #[test]
    fn naive_neigbours() {
        let points = [Euclid([0.0]), Euclid([10.0]), Euclid([5.0]), Euclid([2.5])];
        let points = BruteScan::new(&points);

        assert_eq!(points.neighbours(&0, 1.0).collect::<Vec<_>>(),
                   [0]);
        assert_eq!(points.neighbours(&0, 3.0).collect::<Vec<_>>(),
                   [0, 3]);
        assert_eq!(points.neighbours(&1, 3.0).collect::<Vec<_>>(),
                   [1]);
        assert_eq!(points.neighbours(&1, 10.0).collect::<Vec<_>>(),
                   [0, 1, 2, 3]);
        assert_eq!(points.neighbours(&3, 3.0).collect::<Vec<_>>(),
                   [0, 2, 3]);

    }
}
