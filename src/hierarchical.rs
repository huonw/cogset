use std::cmp::{min, max};
use std::f64::INFINITY;
use std::mem::replace;

use {Point};


/// Agglomerative clustering, i.e., hierarchical bottum-up clustering.
///
/// How the clustering actually behaves depends on the linkage criterion, which is
/// mainly defined by the implementator and distance of two points.
pub trait AgglomerativeClustering {
    /// Clusters the given list of points.
    ///
    /// Starting with each element in its own cluster, iteratively merges the two most similar
    /// clusters until a certain distance `threshold` is reached. The remaining
    /// clusters are returned. The clusters are represented by vectors of indices into
    /// the original data.
    ///
    /// If you need to cluster certain points repeatidly with different thresholds, it might
    /// makes sense to use `compute_dendrogram()` to build a full dendrogram and use its
    /// `cut_at` method to extract clusters afterwards.
    fn cluster<P: Point>(&self, points: &[P], threshold: f64) -> Vec<Vec<usize>>;

    /// Computes the dendrogram describing the merging process exhaustively.
    ///
    /// The leafs of the returned dendrogram contain the indices into the original set
    /// of points.
    ///
    /// Despite one being interested in the dendrogram itself, it might make sense to use
    /// the returned `Dendrogram` to find various clusters given different thresholds using
    /// the `cut_at` method.
    fn compute_dendrogram<P: Point>(&self, points: &[P]) -> Box<Dendrogram<usize>>;
}


/// A linkage cirterion used for agglomerative clustering.
///
/// Implementing this trait means to define a linkage criterion, which is defined by its
/// method to compute the inter-cluster distance.
pub trait LinkageCriterion {
    /// Compute the distance between cluster `a` and cluster `b`.
    ///
    /// The supplied function `dist` can be used to retrieve the distance between two points given
    /// their indices.
    fn distance<F: Fn(usize, usize) -> f64>(&self, dist: F, a: &[usize], b: &[usize]) -> f64;
}


/// Minimum or single-linkage clustering.
///
/// **Criterion:** `min { d(a, b) | a ∈ A, b ∈ B }`
///
/// For an optimal clustering algorithm using this criterion have a look at `SLINK`.
/// It produces the same results as the naive algorithm given this criterion, but has
/// much better performance characteristics.
pub struct SingleLinkage;

impl LinkageCriterion for SingleLinkage {
    fn distance<F: Fn(usize, usize) -> f64>(&self, dist: F, a: &[usize], b: &[usize]) -> f64 {
        let mut min_d = INFINITY;

        for &i in a {
            for &j in b {
                let d = dist(i, j);

                if d < min_d {
                    min_d = d;
                }
            }
        }

        return min_d;
    }
}


/// Maximum or complete-linkage clustering.
///
/// **Criterion:** `max { d(a, b) | a ? A, b ? B }`
///
/// Use `CLINK` if you need a better performing algorithm, but beware, `CLINK`
/// might not produce the same result as the naive algorithm given this criterion.
pub struct CompleteLinkage;

impl LinkageCriterion for CompleteLinkage {
    fn distance<F: Fn(usize, usize) -> f64>(&self, dist: F, a: &[usize], b: &[usize]) -> f64 {
        let mut max_d = 0.0;

        for &i in a {
            for &j in b {
                let d = dist(i, j);

                if d > max_d {
                    max_d = d;
                }
            }
        }

        return max_d;
    }
}


struct Cluster {
    max_index: usize,
    elements: Vec<usize>,
    dendrogram: Box<Dendrogram<usize>>
}


/// Generic but naive agglomerative (bottom-up) clustering.
///
/// A naive implementation of agglomerative clustering generic over any linkage
/// criterion. It can be used with any linkage criterion, thus it cannot  apply certain
/// optimizations. Check the implementations of `AgglomerativeClustering` to see if there is
/// an optimized implementation for a certain criterion.
///
/// The general case has a O( n^3 ) time complexity and O( n`2 ) space complexity. Beware,
/// they might be even worse depending on the linkage criterion.
pub struct NaiveBottomUp<L: LinkageCriterion> {
    linkage: L,
}

impl<L: LinkageCriterion> NaiveBottomUp<L> {
    pub fn new(linkage: L) -> Self {
        NaiveBottomUp {
            linkage: linkage
        }
    }

    fn cluster_naively<P: Point>(&self, points: &[P], threshold: f64) -> Vec<Cluster> {
        let point_distances: Vec<Vec<f64>> = (0..points.len()).map(|i| {
            (0..i).map(|j| points[i].dist(&points[j])).collect()
        }).collect();
        let dist = |i: usize, j: usize| point_distances[max(i,j)][min(i,j)];

        let mut cluster_distances = point_distances.clone();

        let mut clusters: Vec<_> = (0..points.len()).map(|i| Cluster {
            max_index: i,
            elements: vec![i],
            dendrogram: Box::new(Dendrogram::Leaf(i))
        }).collect();

        while clusters.len() > 1 {
            let mut min_dist = INFINITY;
            let mut merge = (0, 0);

            for i in 1..clusters.len() {
                for j in 0..i {
                    let a = clusters[i].max_index;
                    let b = clusters[j].max_index;

                    let d = cluster_distances[max(a, b)][min(a, b)];

                    if d < min_dist {
                        min_dist = d;
                        merge = (i, j);
                    }
                }
            }

            if min_dist > threshold {
                break;
            }

            let a = clusters.swap_remove(merge.0);
            let b = clusters.swap_remove(merge.1);

            let mut c = Cluster {
                max_index: max(a.max_index, b.max_index),
                elements: a.elements,
                dendrogram: Box::new(Dendrogram::Branch(min_dist, a.dendrogram, b.dendrogram))
            };
            c.elements.extend(b.elements);

            for i in &clusters {
                cluster_distances[max(c.max_index, i.max_index)][min(c.max_index, i.max_index)] =
                    self.linkage.distance(&dist, &*c.elements, &i.elements);
            }

            clusters.push(c);
        }

        clusters
    }
}

impl<L: LinkageCriterion> AgglomerativeClustering for NaiveBottomUp<L> {
    fn cluster<P: Point>(&self, points: &[P], threshold: f64) -> Vec<Vec<usize>> {
        self.cluster_naively(points, threshold).into_iter().map(|c| c.elements).collect()
    }

    fn compute_dendrogram<P: Point>(&self, points: &[P]) -> Box<Dendrogram<usize>> {
        self.cluster_naively(points, INFINITY).pop().unwrap().dendrogram
    }
}


/// Optimal single-linkage clustering using the SLINK algorithm.
///
/// Uses an implementation of the optimal SLINK [[1]](#slink) algorithm with
/// O( n^2 ) time and O( n ) space complexity. Produces the same result as
/// naive bottom-up clustering given the single-linkage criterion.
///
/// # References
///
/// * <a name="slink">[1]</a>: Sibson, R. (1973). *SLINK: an optimally efficient algorithm for
///   the single-link cluster method.* The Computer Journal, 16(1), 30-34.
pub struct SLINK;

impl AgglomerativeClustering for SLINK {
    fn cluster<P: Point>(&self, points: &[P], threshold: f64) -> Vec<Vec<usize>> {
        slink(points, threshold).into_iter().map(|c| c.elements).collect()
    }

    fn compute_dendrogram<P: Point>(&self, points: &[P]) -> Box<Dendrogram<usize>> {
        slink(points, INFINITY).pop().unwrap().dendrogram
    }
}


/// Optimal complete-linkage clustering using the CLINK algorithm.
///
/// Uses an implementation of the optimal CLINK [[1]](#clink) algorithm with
/// O( n^2 ) time and O( n ) space complexity.
///
/// **Beware:** It depends on the order of the elements and doesn't always return the best
/// dendrogram. Thus it _might not_ return the same result as the naive bottom-up approach
/// given the complete-linkage criterion.
///
/// # References
///
/// * <a name="clink">[1]</a>: Defays, D. (1977). *An efficient algorithm for a complete link
///   method. The Computer Journal*, 20(4), 364-366.
pub struct CLINK;

impl AgglomerativeClustering for CLINK {
    fn cluster<P: Point>(&self, points: &[P], threshold: f64) -> Vec<Vec<usize>> {
        clink(points, threshold).into_iter().map(|c| c.elements).collect()
    }

    fn compute_dendrogram<P: Point>(&self, points: &[P]) -> Box<Dendrogram<usize>> {
        clink(points, INFINITY).pop().unwrap().dendrogram
    }
}


// the SLINK algorithm to perform optimal single linkage clustering
fn slink<P: Point>(points: &[P], threshold: f64) -> Vec<Cluster> {
    using_pointer_representation(points, threshold, |n, pi, lambda, em| {
        for i in 0..n {
            if lambda[i] >= em[i] {
                em[pi[i]] = em[pi[i]].min(lambda[i]);
                lambda[i] = em[i];
                pi[i] = n;
            } else {
                em[pi[i]] = em[pi[i]].min(em[i]);
            }
        }

        for i in 0..n {
            if lambda[i] >= lambda[pi[i]] {
                pi[i] = n;
            }
        }
    })
}

// the CLINK algorithm to perform optimal complete linkage clustering
fn clink<P: Point>(points: &[P], threshold: f64) -> Vec<Cluster> {
    using_pointer_representation(points, threshold, |n, pi, lambda, em| {
        for i in 0..n {
            if lambda[i] < em[i] {
                em[pi[i]] = em[pi[i]].max(em[i]);
                em[i] = INFINITY;
            }
        }

        let mut a = n - 1;

        for i in (0..n).rev() {
            if lambda[i] >= em[pi[i]] {
                if em[i] < em[a] {
                    a = i;
                }
            } else {
                em[i] = INFINITY;
            }
        }

        let mut b = replace(&mut pi[a], n);
        let mut c = replace(&mut lambda[a], em[a]);

        if a < n - 1 {
            while  b < n - 1 {
                b = replace(&mut pi[b], n);
                c = replace(&mut lambda[b], c);
            }

            if b == n - 1 {
                pi[b] = n;
                lambda[b] = c;
            }
        }

        for i in 0..n {
            if pi[pi[i]] == n && lambda[i] >= lambda[pi[i]] {
                pi[i] = n;
            }
        }
    })
}

fn using_pointer_representation<P, F>(points: &[P], threshold: f64, f: F) -> Vec<Cluster>
    where P: Point, F: Fn(usize, &mut Vec<usize>, &mut Vec<f64>, &mut Vec<f64>)
{
    let mut pi = vec![0; points.len()];
    let mut lambda = vec![0.0; points.len()];
    let mut em = vec![0.0; points.len()];

    pi[0] = 0;
    lambda[0] = INFINITY;

    for n in 1..points.len() {
        pi[n] = n;
        lambda[n] = INFINITY;

        for i in 0..n {
            em[i] = points[i].dist(&points[n]);
        }

        f(n, &mut pi, &mut lambda, &mut em);
    }

    // convert pointer representation to dendrogram
    let mut dendrograms: Vec<_> = (0..points.len()).map(|i| Some(Cluster {
        max_index: i,
        elements: vec![i],
        dendrogram: Box::new(Dendrogram::Leaf(i))
    })).collect();

    let mut idx: Vec<_> = (0..points.len()).collect();
    idx.sort_by(|&a, &b| lambda[a].partial_cmp(&lambda[b]).unwrap());

    for i in 0..points.len()-1 {
        let leaf = idx[i];

        if lambda[leaf] > threshold {
            break;
        }

        let merge = pi[leaf];

        let a = replace(&mut dendrograms[leaf], None).unwrap();
        let b = replace(&mut dendrograms[merge], None).unwrap();

        let mut c = Cluster {
            max_index: max(a.max_index, b.max_index),
            elements: a.elements,
            dendrogram: Box::new(if leaf < merge {
                Dendrogram::Branch(lambda[leaf], a.dendrogram, b.dendrogram)
            } else {
                Dendrogram::Branch(lambda[leaf], b.dendrogram, a.dendrogram)
            })
        };
        c.elements.extend(b.elements);

        dendrograms[max(leaf, merge)] = Some(c);
    }

    dendrograms.into_iter().filter_map(|x| x).collect()
}


/// A hierarchical tree structure describing the merge operations.
#[derive(Clone, Debug)]
pub enum Dendrogram<T> {
    /// The union of two sub-trees (clusters) and their distance.
    Branch(f64, Box<Dendrogram<T>>, Box<Dendrogram<T>>),

    /// The leaf node contains an element.
    Leaf(T)
}

impl<T: Clone> Dendrogram<T> {
    /// Extract clusters given a certain threshold `threshold`.
    pub fn cut_at(&self, threshold: f64) -> Vec<Vec<T>> {
        let mut dendrograms = vec![self];
        let mut clusters = vec![];

        while let Some(dendrogram) = dendrograms.pop() {
            match dendrogram {
                &Dendrogram::Leaf(ref i) => clusters.push(vec![i.clone()]),
                &Dendrogram::Branch(d, ref a, ref b) => {
                    if d < threshold {
                        clusters.push(dendrogram.into_iter().map(|i| i.clone()).collect());
                    } else {
                        dendrograms.push(a);
                        dendrograms.push(b);
                    }
                }
            }
        }

        return clusters;
    }
}

impl<T: PartialEq> PartialEq for Dendrogram<T> {
    fn eq(&self, other: &Self) -> bool {
        if let (&Dendrogram::Leaf(ref a), &Dendrogram::Leaf(ref b)) = (self, other) {
            a == b
        } else if let (&Dendrogram::Branch(d1, ref a1, ref b1),
            &Dendrogram::Branch(d2, ref a2, ref b2)) = (self, other)
        {
            d1 == d2 && ((a1 == a2 && b1 == b2) || (a1 == b2 && a2 == b1))
        } else {
            false
        }
    }
}


/// An `Iterator` over all elements contained in an `Dendrogram`.
#[derive(Clone, Debug)]
pub struct Elements<'a, T: 'a> {
    items: Vec<&'a Dendrogram<T>>
}

impl<'a, T> Iterator for Elements<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop().and_then(|item| {
            match item {
                &Dendrogram::Branch(_, ref a, ref b) => {
                    self.items.push(a);
                    self.items.push(b);

                    self.next()
                },
                &Dendrogram::Leaf(ref p) => Some(p)
            }
        })
    }
}

impl<'a, T> IntoIterator for &'a Dendrogram<T> {
    type Item = &'a T;
    type IntoIter = Elements<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Elements {
            items: vec![self]
        }
    }
}


#[cfg(test)]
mod tests {
    use permutohedron::Heap;
    use quickcheck::{Arbitrary, Gen, TestResult, quickcheck};

    use Euclid;
    use super::{AgglomerativeClustering, NaiveBottomUp, SingleLinkage,
        CompleteLinkage, SLINK, CLINK};

    #[test]
    fn single_dendrogram() {
        fn prop(points: Vec<Euclid<[f64; 2]>>) -> TestResult {
            if points.len() == 0 {
                return TestResult::discard();
            }

            let optimal = SLINK.compute_dendrogram(&*points);
            let naive = NaiveBottomUp::new(SingleLinkage).compute_dendrogram(&*points);

            TestResult::from_bool(optimal == naive)
        }
        quickcheck(prop as fn(Vec<Euclid<[f64; 2]>>) -> TestResult);
    }

    #[test]
    fn single_cluster() {
        fn prop(points: Vec<Euclid<[f64; 2]>>, threshold: f64) -> TestResult {
            if points.len() == 0 || threshold <= 0.0 {
                return TestResult::discard();
            }

            let optimal = SLINK.cluster(&*points, threshold);
            let naive = NaiveBottomUp::new(SingleLinkage).cluster(&*points, threshold);

            TestResult::from_bool(eq_clusters(&optimal, &naive))
        }
        quickcheck(prop as fn(Vec<Euclid<[f64; 2]>>, f64) -> TestResult);
    }

    #[test]
    fn dendrogram_cut_at() {
        fn prop(points: Vec<Euclid<[f64; 2]>>, threshold: f64) -> TestResult {
            if points.len() == 0 || threshold <= 0.0 {
                return TestResult::discard();
            }

            let algo = NaiveBottomUp::new(SingleLinkage);

            let one = algo.cluster(&*points, threshold);
            let two = algo.compute_dendrogram(&*points).cut_at(threshold);

            TestResult::from_bool(eq_clusters(&one, &two))
        }
        quickcheck(prop as fn(Vec<Euclid<[f64; 2]>>, f64) -> TestResult);
    }

    fn eq_clusters(a: &Vec<Vec<usize>>, b: &Vec<Vec<usize>>) -> bool {
        for a in a {
            let mut matched = true;

            for b in b {
                matched = true;

                for i in a {
                    if !b.contains(&i) {
                        matched = false;
                        break;
                    }
                }

                if matched {
                    break;
                }
            }

            if !matched {
                return false;
            }
        }

        return true;
    }

    /*
    #[test]
    fn complete() {
        // CLINK is not equal to complete linnkage, it depends on the order of the elements
        // thus we perform an exhaustive comparison to the native algorithm on all possible
        // permutations using only small sets of elements
        fn prop(points: Vec<Euclid<[f64; 2]>>) -> TestResult {
            if points.len() == 0 || points.len() > 6 || points.len() < 4 {
                return TestResult::discard();
            }

            let naive = NaiveBottomUp::new(CompleteLinkage).compute_dendrogram(&*points);

            let mut new = points.clone();
            for points in Heap::new(&mut *new) {
                let optimal = CLINK.compute_dendrogram(&*points);

                if naive == optimal {
                    //panic!("got it");
                    return TestResult::from_bool(true);
                }
            }

            TestResult::from_bool(false)
        }
        quickcheck(prop as fn(Vec<Euclid<[f64; 2]>>) -> TestResult);
    }
    */

    impl Arbitrary for Euclid<[f64; 2]> {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            Euclid([Arbitrary::arbitrary(g), Arbitrary::arbitrary(g)])
        }
    }
}


#[cfg(all(test, feature = "unstable"))]
mod benches {
    use rand::{XorShiftRng, Rng};
    use test::Bencher;

    use Euclid;
    use super::{AgglomerativeClustering, NaiveBottomUp, SingleLinkage, SLINK,
        CompleteLinkage, CLINK};

    macro_rules! benches {
        ($($name: ident, $l: expr, $d: expr, $n: expr;)*) => {
            $(
                #[bench]
                fn $name(b: &mut Bencher) {
                    let mut rng = XorShiftRng::new_unseeded();
                    let points = (0..$n)
                        .map(|_| Euclid(rng.gen::<[f64; $d]>()))
                        .collect::<Vec<_>>();

                    b.iter(|| $l.compute_dendrogram(&*points))
                }
            )*
        }
    }

    benches! {
        slink_d1_n0010, SLINK, 1,   10;
        slink_d1_n0100, SLINK, 1,  100;
        slink_d1_n1000, SLINK, 1, 1000;

        naive_single_d1_n0010, NaiveBottomUp::new(SingleLinkage), 1,   10;
        naive_single_d1_n0100, NaiveBottomUp::new(SingleLinkage), 1,  100;
        //naive_single_d1_n1000, NaiveBottomUp::new(SingleLinkage), 1, 1000;

        clink_d1_n0010, CLINK, 1,   10;
        clink_d1_n0100, CLINK, 1,  100;
        clink_d1_n1000, CLINK, 1, 1000;

        naive_complete_d1_n0010, NaiveBottomUp::new(CompleteLinkage), 1,   10;
        naive_complete_d1_n0100, NaiveBottomUp::new(CompleteLinkage), 1,  100;
        //naive_complete_d1_n1000, NaiveBottomUp::new(CompleteLinkage), 1, 1000;
    }
}
