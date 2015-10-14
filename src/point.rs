/// A point in some (metric) space.
///
/// # Example
///
/// ```rust
/// use cogset::Point;
///
/// struct TwoD {
///     x: f64, y: f64
/// }
///
/// impl Point for TwoD {
///     fn dist(&self, other: &TwoD) -> f64 {
///         self.dist_monotonic(other).sqrt()
///     }
///
///     // these three methods together are optional, but can provide an
///     // optimisation for some algorithms (if one is implemented the
///     // others must be too)
///     fn dist_monotonic(&self, other: &TwoD) -> f64 {
///         let dx = self.x - other.x;
///         let dy = self.y - other.y;
///         dx * dx + dy * dy
///     }
///     fn monotonic_transform(x: f64) -> f64 {
///         x * x
///     }
///     fn monotonic_inverse(x: f64) -> f64 {
///         x.sqrt()
///     }
///
///     // another optimisation for some algorithms (if `dist` is
///     // cheap to compute this can just be left off entirely)
///     fn dist_lower_bound(&self, other: &TwoD) -> f64 {
///         (self.x - other.x).abs()
///     }
/// }
/// ```
pub trait Point {
    /// Accurately compute the distance from `self` to `other`.
    ///
    /// This should be real and non-negative, or else algorithms may
    /// return unexpected results.
    fn dist(&self, other: &Self) -> f64;

    /// Accurately compute some monotonic function of the distance
    /// from `self` to `other`.
    ///
    /// This should satisfy `a.dist_monotonic(b) ==
    /// Self::monotonic_transform(a.dist(b))` where
    /// `Self::monotonic_transform` has been implemented to be
    /// increasing and non-negative.
    ///
    /// This can be used to optimize algorithms that care only about
    /// the ordering of distances, not their magnitude, since it may
    /// be implemented more efficiently than `a.dist(b)` itself. For
    /// example, for the 2-D Euclidean distance between
    /// (<i>x</i><sub>0</sub>, <i>y</i><sub>0</sub>) and
    /// (<i>x</i><sub>1</sub>, <i>y</i><sub>1</sub>), `dist_monotonic`
    /// could be Δ = (<i>x</i><sub>0</sub> -
    /// <i>x</i><sub>1</sub>)<sup>2</sup> + (<i>y</i><sub>0</sub> -
    /// <i>y</i><sub>1</sub>)<sup>2</sup> (i.e. `monotonic_transform`
    /// is <i>x</i> ↦ <i>x</i><sup>2</sup>) instead of sqrt(Δ) as
    /// `dist` requires.
    ///
    /// **Warning**: changes to this should be reflected by changes to
    /// `monotonic_transform` and `monotonic_inverse`.
    fn dist_monotonic(&self, other: &Self) -> f64 {
        self.dist(other)
    }

    /// Perform the increasing and non-negative transformation that
    /// `dist_monotonic` applies to `dist`.
    ///
    /// It should satisfy:
    ///
    /// - `a.dist_monotonic(b) ==
    ///    Self::monotonic_transform(a.dist(b))` for all points `a` and
    ///    `b`,
    ///
    /// - `Self::monotonic_transform(x) <
    ///    Self::monotonic_transform(y)` if and only if `x < y`, for
    ///    `x >= 0`, `y >= 0`.
    ///
    /// **Warning**: changes to this should be reflected by changes to
    /// `dist_monotonic` and `monotonic_inverse`.
    fn monotonic_transform(x: f64) -> f64 {
        x
    }

    /// Perform the inverse of `monotonic_transform` to `x`.
    ///
    /// This should satisfy
    /// `Self::monotonic_transform(Self::monotonic_inverse(x)) == x`
    /// and similarly with the application order reversed.
    ///
    /// **Warning**: changes to this should be reflected by changes to
    /// `dist_monotonic` and `monotonic_transform`.
    fn monotonic_inverse(x: f64) -> f64 {
        x
    }

    /// Compute an estimate of the distance from `self` to `other`.
    ///
    /// This should be less than or equal to `self.dist(other)` or
    /// else algorithms may return unexpected results.
    #[allow(unused_variables)]
    fn dist_lower_bound(&self, other: &Self) -> f64 {
        ::std::f64::NEG_INFINITY
    }
}

impl<'a, P: Point + ?Sized> Point for &'a P {
    fn dist(&self, other: &Self) -> f64 {
        (**self).dist(other)
    }
    fn dist_lower_bound(&self, other: &Self) -> f64 {
        (**self).dist_lower_bound(other)
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
    /// An iterator over the nearby points and their distances of a given one.
    type Neighbours: Iterator<Item = (f64, Self::Point)>;

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
    type Item = (f64, usize);

    fn next(&mut self) -> Option<(f64,usize)> {
        let BruteScanNeighbours { ref mut points, point, eps } = *self;

        points.filter_map(|(i, p)| {
            if point.dist_lower_bound(p) <= eps {
                let d = point.dist(p);
                if d <= eps {
                    return Some((d, i))
                }
            };
            None
        }).next()
    }
}

/// Points in ℝ<sup><i>n</i></sup> with the <i>L</i><sup>2</sup> norm.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Euclid<T>(pub T);

pub trait Euclidean {
    fn zero() -> Self;
    fn add(&mut self, &Self);
    fn scale(&mut self, f64);
}

macro_rules! euclidean_points {
    ($($e: expr),*) => {
        $(
            impl Point for Euclid<[f64; $e]> {
                #[inline]
                fn dist(&self, other: &Euclid<[f64; $e]>) -> f64 {
                    self.dist_monotonic(other).sqrt()
                }

                #[inline]
                fn dist_monotonic(&self, other: &Euclid<[f64; $e]>) -> f64 {
                    self.0.iter().zip(other.0.iter())
                        .map(|(a, b)| {
                            let d = *a - *b;
                            d * d
                        })
                        .fold(0.0, |a, b| a + b)
                }

                #[inline]
                fn monotonic_transform(x: f64) -> f64 {
                    x * x
                }

                #[inline]
                fn monotonic_inverse(x: f64) -> f64 {
                    x.sqrt()
                }

                #[inline]
                fn dist_lower_bound(&self, other: &Euclid<[f64; $e]>) -> f64 {
                    (self.0[0] - other.0[0]).abs()
                }
            }
            impl Euclidean for Euclid<[f64; $e]> {
                fn zero() -> Self {
                    Euclid([0.0; $e])
                }
                fn add(&mut self, other: &Self) {
                    for (place, val) in self.0.iter_mut().zip(other.0.iter()) {
                        *place += *val
                    }
                }
                fn scale(&mut self, factor: f64) {
                    for place in &mut self.0 {
                        *place *= factor
                    }
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

        type V = Vec<(f64, usize)>;

        assert_eq!(points.neighbours(&0, 1.0).collect::<V>(),
                   [(0.0, 0)]);
        assert_eq!(points.neighbours(&0, 3.0).collect::<V>(),
                   [(0.0, 0), (2.5, 3)]);
        assert_eq!(points.neighbours(&1, 3.0).collect::<V>(),
                   [(0.0, 1)]);
        assert_eq!(points.neighbours(&1, 10.0).collect::<V>(),
                   [(10.0, 0), (0.0, 1), (5.0, 2), (7.5, 3)]);
        assert_eq!(points.neighbours(&3, 3.0).collect::<V>(),
                   [(2.5, 0), (2.5, 2), (0.0, 3)]);
    }
}
