use std::collections::{HashMap, HashSet, BinaryHeap};
use std::f64;
use std::hash::Hash;
use std::slice;

use order_stat;

use {Point, RegionQuery, ListPoints, Points};

/// Clustering via the OPTICS algorithm[1].
///
/// > [OPTICS] is an algorithm for finding density-based clusters in
/// spatial data. [...] Its basic idea is similar to DBSCAN, but it
/// addresses one of DBSCAN's major weaknesses: the problem of
/// detecting meaningful clusters in data of varying density.<sup><a
/// href="https://en.wikipedia.org/wiki/OPTICS_algorithm">wikipedia</a></sup>
///
/// An instance of `Optics` represents the dendrogram that OPTICS
/// computes for a data set. Once computed, this dendrogram can then
/// be queried for clustering structure, for example, a clustering
/// similar to the one that would be computed by DBSCAN can be
/// retrieved with `dbscan_clustering`.
///
/// This uses the `P::Point` yielded by the iterators provided by
/// `ListPoints` and `RegionQuery` as a unique identifier for each
/// point. The algorithm will behave strangely if the identifier is
/// not unique or not stable within a given execution of OPTICS. The
/// identifier is cloned several times in the course of execution, so
/// it should be cheap to duplicate (e.g. a `usize` index, or a `&T`
/// reference).
///
/// [1]: Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, Jörg
/// Sander (1999). *OPTICS: Ordering Points To Identify the Clustering
/// Structure.* ACM SIGMOD international conference on Management of
/// data. ACM Press. pp. 49–60.
///
/// # Examples
///
/// A basic example:
///
/// ```rust
/// use cogset::{Optics, BruteScan, Euclid};
///
/// let points = [Euclid([0.1]), Euclid([0.2]), Euclid([1.0])];
///
/// let scanner = BruteScan::new(&points);
/// let optics = Optics::new(scanner, 0.2, 2);
///
/// // use the same epsilon that OPTICS used
/// let mut clustering = optics.dbscan_clustering(0.2);
///
/// // get the clusters themselves
/// let mut clusters = clustering.by_ref().collect::<Vec<_>>();
/// // the first two points are the only cluster
/// assert_eq!(clusters, &[&[0, 1]]);
/// // now the noise, which is just the last point
/// assert_eq!(clustering.noise_points(), &[2]);
///
///
/// // cluster again, with a much smaller epsilon
/// let mut clustering = optics.dbscan_clustering(0.05);
///
/// // get the clusters themselves
/// let mut clusters = clustering.by_ref().collect::<Vec<_>>();
/// // no clusters (its less than the smallest distance between points)
/// assert!(clusters.is_empty());
/// // everything is noise
/// assert_eq!(clustering.noise_points(), &[0, 1, 2]);
/// ```
pub struct Optics<P: Points> where P::Point: Hash + Eq + Clone {
    computed_eps: f64,
    min_pts: usize,
    #[allow(dead_code)] points: P,
    order: Vec<P::Point>,
    core_dist: HashMap<P::Point, f64>,
    reachability: HashMap<P::Point, f64>,
}

impl<P: RegionQuery + ListPoints> Optics<P>
    where P::Point: Hash + Eq + Clone
{
    /// Run the OPTICS algorithm on the index `points`, with the `eps`
    /// and `min_pts` parameters.
    ///
    /// The return value can be queried for the actual clustering
    /// structure using, for example, `dbscan_clustering`. The
    /// parameter `eps` is used as a performance enhancement, and
    /// should be made as small as possible for the use-case.
    ///
    /// NB. this computes the clustering dendrogram immediately,
    /// unlike `Dbscan`'s laziness.
    pub fn new(points: P, eps: f64, min_pts: usize) -> Optics<P> {
        let mut processed = HashSet::new();
        let mut order = vec![];
        let mut reachability = HashMap::new();
        let mut core_dist = HashMap::new();
        let mut seeds = BinaryHeap::new();
        for p in points.all_points() {
            seeds.clear();
            seeds.push(Dist { dist: 0.0, point: p });
            while let Some(q) = seeds.pop() {
                if !processed.insert(q.point.clone()) {
                    continue
                }

                let mut neighbours = points.neighbours(&q.point, eps)
                                           .map(|t| Dist { dist: t.0, point: t.1 })
                                           .collect::<Vec<_>>();
                order.push(q.point.clone());
                if let Some(cd) = compute_core_dist(&mut neighbours, min_pts) {
                    core_dist.insert(q.point.clone(), cd);
                    update(&neighbours, cd, &processed, &mut seeds, &mut reachability)
                }
            }
        }
        Optics {
            points: points,
            min_pts: min_pts,
            computed_eps: eps,
            order: order,
            core_dist: core_dist,
            reachability: reachability,
        }
    }

    /// Extract a clustering like one that DBSCAN would give.
    ///
    /// The returned type is similar to the `Dbscan` type: an iterator
    /// over the clusters (as vectors of points), along with the
    /// `noise_points` method to retrieve unclustered points.
    ///
    /// # Panics
    ///
    /// `eps` must be less than the `eps` passed to `new`.
    pub fn dbscan_clustering<'a>(&'a self, eps: f64) -> OpticsDbscanClustering<'a, P> {
        assert!(eps <= self.computed_eps);
        OpticsDbscanClustering {
            noise: vec![],
            order: self.order.iter(),
            optics: self,
            next: None,
            eps: eps,
        }
    }
}

/// An iterator over clusters generated by OPTICS using a DBSCAN-like
/// criterion for clustering.
///
/// This type offers essentially the same interface as `Dbscan`.
pub struct OpticsDbscanClustering<'a, P: 'a + Points>
    where P::Point: 'a + Eq + Hash + Clone
{
    noise: Vec<P::Point>,
    order: slice::Iter<'a, P::Point>,
    optics: &'a Optics<P>,
    next: Option<P::Point>,
    eps: f64,
}

impl<'a, P: Points> OpticsDbscanClustering<'a, P>
    where P::Point: 'a + Eq + Hash + Clone
{
    pub fn noise_points(&self) -> &[P::Point] {
        &self.noise
    }
}
impl<'a, P: RegionQuery + ListPoints> Iterator for OpticsDbscanClustering<'a, P>
    where P::Point: 'a + Eq + Hash + Clone + ::std::fmt::Debug
{
    type Item = Vec<P::Point>;
    #[inline(never)]
    fn next(&mut self) -> Option<Vec<P::Point>> {
        let mut current = Vec::with_capacity(self.optics.min_pts);
        if let Some(x) = self.next.take() {
            current.push(x)
        }

        for p in &mut self.order {
            if *self.optics.reachability.get(p).unwrap_or(&f64::INFINITY) > self.eps {
                if *self.optics.core_dist.get(p).unwrap_or(&f64::INFINITY) <= self.eps {
                    if current.len() > 0 {
                        self.next = Some(p.clone());
                        return Some(current)
                    }
                } else {
                    self.noise.push(p.clone());
                    continue
                }
            }
            current.push(p.clone())
        }
        if current.len() > 0 {
            Some(current)
        } else {
            None
        }
    }
}

#[inline(never)]
fn update<P>(neighbours: &[Dist<P>],
             core_dist: f64,
             processed: &HashSet<P>,
             seeds: &mut BinaryHeap<Dist<P>>,
             reachability: &mut HashMap<P, f64>)
    where P: Hash + Eq + Clone
{
    for n in neighbours {
        if processed.contains(&n.point) {
            continue
        }

        let new_reach_dist = core_dist.max(n.dist);
        let entry = reachability.entry(n.point.clone()).or_insert(f64::INFINITY);
        if new_reach_dist < *entry {
            *entry = new_reach_dist;
            // BinaryHeap is a max-heap, but we need a min-heap
            seeds.push(Dist { dist: -new_reach_dist, point: n.point.clone() })
        }
    }
}

#[derive(Clone)]
struct Dist<P> {
    dist: f64,
    point: P
}
impl<P> PartialEq for Dist<P> {
    fn eq(&self, other: &Dist<P>) -> bool {
        self.dist == other.dist
    }
}
impl<P> Eq for Dist<P> {}
use std::cmp::Ordering;
impl<P> PartialOrd for Dist<P> {
    fn partial_cmp(&self, other: &Dist<P>) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl<P> Ord for Dist<P> {
    fn cmp(&self, other: &Dist<P>) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn compute_core_dist<P>(x: &mut [Dist<P>], n: usize) -> Option<f64> {
    if x.len() >= n {
        Some(order_stat::kth(x, n - 1).dist)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use {Point, BruteScan};
    #[derive(Copy, Clone)]
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
        let optics = Optics::new(points, 0.8, 3);
        let mut clustering = optics.dbscan_clustering(0.8);
        println!("{:?}", optics.reachability);
        let mut clusters = clustering.by_ref().collect::<Vec<_>>();

        // normalise:
        for x in &mut clusters { x.sort() }
        clusters.sort();


        assert_eq!(clusters, &[&[0usize, 3, 4] as &[_], &[1usize, 2, 5, 6] as &_]);
        assert_eq!(clustering.noise_points().iter().cloned().collect::<Vec<_>>(),
                   &[7]);
    }

    #[test]
    fn reachability_restricted() {
        use std::f64::INFINITY as INF;
        macro_rules! l {
            ($($e: expr),*) => {
                [$(Linear($e),)*]
            }
        }
        let points = l![0.0, 0.01, 10.0, 9.5, 0.6, 0.5, 9.1, 9.9, 5.0, 5.3];
        let scanner = BruteScan::new(&points);
        let optics = Optics::new(scanner, 0.5, 3);

        let expected = [(0.0, INF),
                        (0.01, 0.5),
                        (0.5, 0.49),
                        (0.6, 0.49),
                        (10.0, INF),
                        (9.9, 0.5),
                        (9.5, 0.4),
                        (9.1, 0.4),
                        (5.0, INF),
                        (5.3, INF)];
        assert_eq!(optics.order.len(), points.len());
        for (&idx, &(point, reachability)) in optics.order.iter().zip(&expected) {
            let idx_point = points[idx];
            assert_eq!(idx_point.0, point);

            let computed_r = optics.reachability.get(&idx).map_or(INF, |&f| f);
            assert!((reachability == computed_r) || (reachability - computed_r).abs() < 1e-5,
                    "difference in reachability for {} ({}): true {}, computed {}", idx, point,
                    reachability, computed_r);
        }
    }
    #[test]
    fn reachability_unrestricted() {
        use std::f64::INFINITY as INF;
        macro_rules! l {
            ($($e: expr),*) => {
                [$(Linear($e),)*]
            }
        }
        let points = l![0.0, 0.01, 10.0, 9.5, 0.6, 0.5, 9.1, 9.9, 5.0, 5.3];
        let scanner = BruteScan::new(&points);
        let optics = Optics::new(scanner, 1e10, 3);

        let expected = [(0.0, INF),
                        (0.01, 0.5),
                        (0.5, 0.49),
                        (0.6, 0.49),
                        (5.0, 4.4),
                        (5.3, 4.1),
                        (9.1, 3.8),
                        (9.5, 0.8),
                        (9.9, 0.4),
                        (10.0, 0.4)];

        assert_eq!(optics.order.len(), points.len());
        for (&idx, &(point, reachability)) in optics.order.iter().zip(&expected) {
            let idx_point = points[idx];
            assert_eq!(idx_point.0, point);

            let computed_r = optics.reachability.get(&idx).map_or(INF, |&f| f);
            assert!((reachability == computed_r) || (reachability - computed_r).abs() < 1e-5,
                    "difference in reachability for {} ({}): true {}, computed {}", idx, point,
                    reachability, computed_r);
        }
    }
}

make_benches!(|p, e, mp| super::Optics::new(p, e, mp).dbscan_clustering(e).count());
