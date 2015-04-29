use std::collections::HashSet;
use std::hash::Hash;

use {Point, RegionQuery, ListPoints};

/// Clustering via the DBSCAN algorithm[1].
///
/// DBSCAN is a density-based clustering algorithm: given a set of
/// points in some space, it groups together points that are closely
/// packed together (points with many nearby neighbors), marking as
/// outliers points that lie alone in low-density regions (whose
/// nearest neighbors are too far away).<sup><a
/// href="https://en.wikipedia.org/wiki/DBSCAN">wikipedia</a></sup>
///
/// An instance of `Dbscan` is an iterator over clusters of
/// `P`. Points classified as noise once all clusters are found are
/// available via `noise_points`.
///
/// This uses the `P::Point` yielded by the iterators provided by
/// `ListPoints` and `RegionQuery` as a unique identifier for each
/// point. The algorithm will behave strangely if the identifier is
/// not unique or not stable within a given execution of DBSCAN. The
/// identifier is cloned several times in the course of execution, so
/// it should be cheap to duplicate (e.g. a `usize` index, or a `&T`
/// reference).
///
/// [1]: Ester, Martin; Kriegel, Hans-Peter; Sander, Jörg; Xu, Xiaowei
/// (1996). Simoudis, Evangelos; Han, Jiawei; Fayyad, Usama M., eds. *A
/// density-based algorithm for discovering clusters in large spatial
/// databases with noise.* Proceedings of the Second International
/// Conference on Knowledge Discovery and Data Mining (KDD-96). AAAI
/// Press. pp. 226–231.
///
/// # Examples
///
/// A basic example:
///
/// ```rust
/// use cogset::{Dbscan, BruteScan, Euclid};
///
/// let points = [Euclid([0.1]), Euclid([0.2]), Euclid([1.0])];
///
/// let scanner = BruteScan::new(&points);
/// let mut dbscan = Dbscan::new(scanner, 0.2, 2);
///
/// // get the clusters themselves
/// let clusters = dbscan.by_ref().collect::<Vec<_>>();
/// // the first two points are the only cluster
/// assert_eq!(clusters, &[&[0, 1]]);
///
/// // now the noise
/// let noise = dbscan.noise_points();
/// // which is just the last point
/// assert_eq!(noise.iter().cloned().collect::<Vec<_>>(),
///            &[2]);
/// ```
///
/// A more complicated example that renders the output nicely:
///
/// ```rust
/// use std::str;
/// use cogset::{Dbscan, BruteScan, Euclid};
///
/// fn write_points<I>(output: &mut [u8; 76], byte: u8, it: I)
///     where I: Iterator<Item = Euclid<[f64; 1]>>
/// {
///     for p in it { output[(p.0[0] * 30.0) as usize] = byte; }
/// }
///
/// // the points we're going to cluster, considered as points in ℝ
/// // with the conventional distance.
/// let points = [Euclid([0.25]), Euclid([0.9]), Euclid([2.0]), Euclid([1.2]),
///               Euclid([1.9]), Euclid([1.1]),  Euclid([1.35]), Euclid([1.85]),
///               Euclid([1.05]), Euclid([0.1]), Euclid([2.5]), Euclid([0.05]),
///               Euclid([0.6]), Euclid([0.55]), Euclid([1.6])];
///
/// // print the points before clustering
/// let mut original = [b' '; 76];
/// write_points(&mut original, b'x', points.iter().cloned());
/// println!("{}", str::from_utf8(&original).unwrap());
///
/// // set-up the data structure that will manage the queries that
/// // Dbscan needs to do.
/// let scanner = BruteScan::new(&points);
///
/// // create the clusterer: we need 3 points to consider a group a
/// // cluster, and we're only looking at points 0.2 units apart.
/// let min_points = 3;
/// let epsilon = 0.2;
/// let mut dbscan = Dbscan::new(scanner, epsilon, min_points);
///
/// let mut clustered = [b' '; 76];
///
/// // run over all the clusters, writing each to the output
/// for (i, cluster) in dbscan.by_ref().enumerate() {
///     // since we used `BruteScan`, `cluster` is a vector of indices
///     // into `points`, not the points themselves, so lets map back
///     // to the points.
///     let actual_points = cluster.iter().map(|idx| points[*idx]);
///
///     write_points(&mut clustered, b'0' + i as u8,
///                  actual_points)
/// }
/// // now run over the noise points, i.e. points that aren't close
/// // enough to others to be in a cluster.
/// let noise = dbscan.noise_points();
/// write_points(&mut clustered, b'.',
///              noise.iter().map(|idx| points[*idx]));
///
/// // print the numbered clusters
/// println!("{}", str::from_utf8(&clustered).unwrap());
/// ```
///
/// Output:
///
/// ```txt
///  x x   x        x x        x   x x  x   x       x      x x  x              x
///  0 0   0        . .        2   2 2  2   2       .      1 1  1              .
/// ```
pub struct Dbscan<P: RegionQuery + ListPoints> where P::Point: Hash + Eq + Clone {
    visited: HashSet<P::Point>,
    in_cluster: HashSet<P::Point>,
    unclustered: HashSet<P::Point>,
    points: P,
    all_points: P::AllPoints,
    eps: f64,
    min_points: usize,
}

impl<P: RegionQuery + ListPoints> Dbscan<P> where P::Point: Hash + Eq + Clone {
    /// Create a new DBSCAN instance, with the given `eps` and
    /// `min_points`.
    ///
    /// `eps` is the maximum distance between points when creating
    /// neighbours to construct clusters. `min_points` is the minimum
    /// of points for a cluster.
    pub fn new(points: P, eps: f64, min_points: usize) -> Dbscan<P> {
        Dbscan {
            all_points: points.all_points(),
            points: points,
            eps: eps,
            min_points: min_points,
            visited: HashSet::new(),
            in_cluster: HashSet::new(),
            unclustered: HashSet::new(),
        }
    }

    /// Points that have been classified as noise once the algorithm
    /// finishes.
    ///
    /// This only makes sense to call once the iterator is exhausted,
    /// and will give unspecified nonsense if called earlier.
    pub fn noise_points(&self) -> &HashSet<P::Point> {
        &self.unclustered
    }
}

impl<P: RegionQuery + ListPoints> Iterator for Dbscan<P> where P::Point: Hash + Eq + Clone {
    type Item = Vec<P::Point>;
    fn next(&mut self) -> Option<Vec<P::Point>> {
        let mut nbrs;
        loop {
            match self.all_points.next() {
                Some(p) => {
                    if self.visited.insert(p.clone()) {
                        let n = self.points.neighbours(&p, self.eps).collect::<Vec<_>>();
                        if n.len() >= self.min_points {
                            nbrs = n;
                            break
                        } else {
                            self.unclustered.insert(p);
                        }
                    }
                }
                None => return None
            }
        }

        let mut cluster = vec![];

        for idx in 0.. {
            if idx >= nbrs.len() { break }
            let (_, p2) = nbrs[idx].clone();
            if self.visited.insert(p2.clone()) {
                let old_len = nbrs.len();
                nbrs.extend(self.points.neighbours(&p2, self.eps));
                if nbrs.len() - old_len < self.min_points {
                    // undo: the new point doesn't have enough close
                    nbrs.truncate(old_len)
                }
            }
            if self.in_cluster.insert(p2.clone()) {
                self.unclustered.remove(&p2);
                cluster.push(p2);
            }
        }
        Some(cluster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use {Point, BruteScan};

    struct Linear(f64);
    impl Point for Linear {
        fn dist(&self, other: &Linear) -> f64 {
            (self.0 - other.0).abs()
        }
        fn dist_lower_bound(&self, other: &Linear) -> f64 {
            self.dist(other)
        }
    }

    #[test]
    fn smoke() {
        // 0 ...        .         .... 10 (not to scale)
        let points = [Linear(0.0), Linear(10.0), Linear(9.5), Linear(0.5), Linear(0.6),
                      Linear(9.1), Linear(9.9), Linear(5.0)];
        let points = BruteScan::new(&points);
        let mut dbscan = Dbscan::new(points, 0.5, 3);
        let mut clusters = dbscan.by_ref().collect::<Vec<_>>();

        // normalise:
        for x in &mut clusters { x.sort() }
        clusters.sort();


        assert_eq!(clusters, &[&[0usize, 3, 4] as &[_], &[1usize, 2, 5, 6] as &_]);
        assert_eq!(dbscan.noise_points().iter().cloned().collect::<Vec<_>>(),
                   &[7]);
    }
}

make_benches!(|p, e, mp| super::Dbscan::new(p, e, mp).count());
