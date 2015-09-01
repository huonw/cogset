use std::borrow::Borrow;
use std::cmp::{Ordering, min, max};
use std::f64::INFINITY;
use std::mem::replace;

use {Point};


/// A function to compute the inter-cluster distance between set `A` and `B`.
pub type LinkageFunction = Fn(&Fn(usize, usize) -> f64, &[usize], &[usize]) -> f64;


/// Hierarchical linkage criteria.
///
/// The linkage decides how the distance between clusters is computed.
///
/// <script src="https://is.gd/BFouBe"></script>
#[derive(Copy, Clone)]
pub enum LinkageCriterion<'a> {
    /// Minimum or single-linkage clustering.
    ///
    /// <p>
    /// $$\min_{a \in A,\, b \in B} \, d(a, b)$$
    /// </p>
    Single,

    /// Maximum or complete-linkage clustering.
    ///
    /// <p>
    /// $$\max_{a \in A,\, b \in B} \, d(a, b)$$
    /// </p>
    Complete,

    /// An optimal version of complete-linkage clustering known as CLINK.
    ///
    /// **Beware:** It depends on the order of the elements and doesn't always return the best
    /// dendrogram.
    CLINK,

    /// Custom likage criterion.
    ///
    /// By providing a custom `LinkageFunction` it's possible to define
    /// a custom linkage criterion.
    Custom(&'a LinkageFunction)
}


/// Agglomerative clustering, i.e., hierarchical bottum-up clustering.
///
/// Starting with each element in its own cluster, iteratively merge the two most similar
/// clusters until a certain distance threshold is reached. The similarity between two
/// clusters is mainly defined by the linkage criterion and the `Point`'s distance
/// metric.
///
/// * **Single-linkage**: Implementation of the optimal SLINK [1] algorithm with O( n^2 ) time
///   and O( n ) space complexity.
/// * **Complete-linkage**: Currently a naive implementation with O( n^3 ) time
///   and O( n^2 ) space complexity.
/// * **CLINK**: Implementation of the optimal CLINK [2] algorithm with O( n^2 ) time
///   and O( n ) space complexity.
/// * **Custom**: _At least_ O( n^3 ) time and O( n^2 ) space complexity. The exact
///   complexity depends on the linkage criterion.
///
/// # Examples
///
/// ```rust
/// use cogset::{Agglomerative, Euclid, LinkageCriterion};
///
/// let data = [Euclid([0.0, 0.0]),
///             Euclid([1.0, 0.5]),
///             Euclid([0.2, 0.2]),
///             Euclid([0.3, 0.8]),
///             Euclid([0.0, 1.0])];
///
/// let agglomerative = Agglomerative::new(&data, LinkageCriterion::Single);
///
/// println!("Dendogram: {:#?}", agglomerative.dendrogram());
/// ```
///
/// # References
///
/// [1]: Sibson, R. (1973). SLINK: an optimally efficient algorithm for the single-link
///      cluster method. The Computer Journal, 16(1), 30-34.
/// [2]: Defays, D. (1977). An efficient algorithm for a complete link method. The Computer
///     Journal, 20(4), 364-366.
#[derive(Clone, Debug)]
pub struct Agglomerative<'a, P: 'a + Point> {
    dendrogram: Box<Dendrogram<&'a P>>
}

impl<'a, P: 'a + Point> Agglomerative<'a, P> {
    /// Computes the `Dendrogram` for the supplied `data` using `linkage` as criterion.
    pub fn new<T>(data: T, linkage: LinkageCriterion) -> Agglomerative<'a, P>
        where T: IntoIterator<Item=&'a P>
    {
        let points:Vec<_> = data.into_iter().collect();

        let dendrogram = match linkage {
            LinkageCriterion::Single => slink(&points),
            LinkageCriterion::Complete => naive(&points, &maximal_distance),
            LinkageCriterion::CLINK => clink(&points),
            LinkageCriterion::Custom(f) => naive(&points, f)
        };

        Agglomerative {
            dendrogram: dendrogram
        }
    }

    /// Retrieve the clusters from the dendrogram by cutting all branches with a distance
    /// greater than `threshold`.
    pub fn clusters(&'a self, threshold: f64) -> Vec<Vec<&'a P>> {
        let mut clusters = vec![&self.dendrogram];

        let mut i = 0;
        while i < clusters.len() {
            if let &Dendrogram::Branch(d, ref a, ref b) = clusters[i].borrow() {
                if d > threshold {
                    clusters.swap_remove(i);
                    clusters.push(a);
                    clusters.push(b);

                    continue;
                }
            }

            i += 1;
        }

        clusters.iter().map(|c| c.into_iter().map(|p| *p).collect()).collect()
    }

    /// Return the calculate `Dendrogram`.
    pub fn dendrogram(&'a self) -> &'a Box<Dendrogram<&'a P>> {
        &self.dendrogram
    }
}


fn naive<'a, P: Point>(points: &Vec<&'a P>, linkage: &LinkageFunction)
    -> Box<Dendrogram<&'a P>>
{
    let point_distances: Vec<Vec<f64>> = (0..points.len()).map(|i| {
        (0..i).map(|j| points[i].dist(points[j])).collect()
    }).collect();

    let mut cluster_distances = point_distances.clone();

    let mut clusters: Vec<_> = (0..points.len()).map(|i| Cluster {
        max_index: i,
        dendrogram: Box::new(Dendrogram::Leaf(points[i])),
        elements: vec![i]
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

        let a = clusters.swap_remove(merge.0);
        let b = clusters.swap_remove(merge.1);

        let mut c = Cluster {
            max_index: max(a.max_index, b.max_index),
            dendrogram: Box::new(Dendrogram::Branch(min_dist, a.dendrogram, b.dendrogram)),
            elements: a.elements
        };
        c.elements.extend(b.elements);

        for i in &clusters {
            cluster_distances[max(c.max_index, i.max_index)][min(c.max_index, i.max_index)] =
                linkage(&|i: usize,j: usize| point_distances[max(i,j)][min(i,j)],
                    &*c.elements, &i.elements);
        }

        clusters.push(c);
    }

    clusters.pop().unwrap().dendrogram
}

struct Cluster<'a, P: 'a + Point> {
    max_index: usize,
    dendrogram: Box<Dendrogram<&'a P>>,
    elements: Vec<usize>
}


// the SLINK algorithm to perform optimal single linkage clustering
fn slink<'a, P: Point>(points: &Vec<&'a P>) -> Box<Dendrogram<&'a P>> {
    using_pointer_representation(points, |n, pi, lambda, em| {
        for i in (0..n) {
            if lambda[i] >= em[i] {
                em[pi[i]] = em[pi[i]].min(lambda[i]);
                lambda[i] = em[i];
                pi[i] = n;
            } else {
                em[pi[i]] = em[pi[i]].min(em[i]);
            }
        }

        for i in (0..n) {
            if lambda[i] >= lambda[pi[i]] {
                pi[i] = n;
            }
        }
    })
}


// the CLINK algorithm to perform optimal complete linkage clustering
fn clink<'a, P: Point>(points: &Vec<&'a P>) -> Box<Dendrogram<&'a P>> {
    using_pointer_representation(points, |n, pi, lambda, em| {
        for i in (0..n) {
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

        for i in (0..n) {
            if pi[pi[i]] == n && lambda[i] >= lambda[pi[i]] {
                pi[i] = n;
            }
        }
    })
}


fn using_pointer_representation<'a, P, F>(points: &Vec<&'a P>, f: F) -> Box<Dendrogram<&'a P>>
    where P: Point, F: Fn(usize, &mut Vec<usize>, &mut Vec<f64>, &mut Vec<f64>)
{
    let mut pi = vec![0; points.len()];
    let mut lambda = vec![0.0; points.len()];
    let mut em = vec![0.0; points.len()];

    pi[0] = 0;
    lambda[0] = INFINITY;

    for n in (1..points.len()) {
        pi[n] = n;
        lambda[n] = INFINITY;

        for i in (0..n) {
            em[i] = points[i].dist(points[n]);
        }

        f(n, &mut pi, &mut lambda, &mut em);
    }

    // convert pointer representation to dendrogram
    let mut dendrograms: Vec<_> = (0..points.len()).map(|i|
        Some(Box::new(Dendrogram::Leaf(points[i])))).collect();

    let mut idx: Vec<_> = (0..points.len()).collect();
    idx.sort_by(|&a, &b| lambda[a].partial_cmp(&lambda[b]).unwrap());

    for i in (0..points.len()-1) {
        let leaf = idx[i];
        let merge = pi[leaf];

        let a = replace(&mut dendrograms[leaf], None).unwrap();
        let b = replace(&mut dendrograms[merge], None).unwrap();

        if leaf < merge {
            dendrograms[merge] = Some(Box::new(Dendrogram::Branch(lambda[leaf], a, b)));
        } else {
            dendrograms[leaf] = Some(Box::new(Dendrogram::Branch(lambda[leaf], b, a)));
        }
    }

    dendrograms.swap_remove(points.len() - 1).unwrap()
}


#[cfg(test)]
#[allow(non_snake_case)]
fn minimal_distance(d: &Fn(usize, usize) -> f64, A: &[usize], B: &[usize]) -> f64 {
    A.iter().flat_map(|&a|
        B.iter().map(move |&b| Dist(d(a,b)))
    ).min().unwrap().0
}

#[allow(non_snake_case)]
fn maximal_distance(d: &Fn(usize, usize) -> f64, A: &[usize], B: &[usize]) -> f64 {
    A.iter().flat_map(|&a|
        B.iter().map(move |&b| Dist(d(a,b)))
    ).max().unwrap().0
}

#[derive(PartialEq, PartialOrd)]
struct Dist(f64);
impl Eq for Dist { }
impl Ord for Dist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}


/// A hierarchical tree structure describing the merge operations.
#[derive(Clone, Debug)]
pub enum Dendrogram<T> {
    /// The union of two sub-trees (clusters) and their distance.
    Branch(f64, Box<Dendrogram<T>>, Box<Dendrogram<T>>),

    /// The leaf node contains an element.
    Leaf(T)
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
    use super::{Agglomerative, LinkageCriterion, minimal_distance};

    #[test]
    fn single() {
        fn prop(points: Vec<Euclid<[f64; 2]>>) -> TestResult {
            if points.len() == 0 {
                return TestResult::discard();
            }

            let optimal = Agglomerative::new(&points, LinkageCriterion::Single);
            let naive = Agglomerative::new(&points, LinkageCriterion::Custom(&minimal_distance));

            TestResult::from_bool(optimal.dendrogram() == naive.dendrogram())
        }
        quickcheck(prop as fn(Vec<Euclid<[f64; 2]>>) -> TestResult);
    }

    #[test]
    fn complete() {
        // CLINK is not equal to complete linnkage, it depends on the order of the elements
        // thus we perform an exhaustive comparison to the native algorithm on all possible
        // permutations using only small sets of elements
        fn prop(points: Vec<Euclid<[f64; 2]>>) -> TestResult {
            if points.len() == 0 || points.len() > 6 {
                return TestResult::discard();
            }

            let naive = Agglomerative::new(&points, LinkageCriterion::Complete);

            let mut new = points.clone();
            for points in Heap::new(&mut *new) {
                let optimal = Agglomerative::new(&points, LinkageCriterion::CLINK);

                if naive.dendrogram() == optimal.dendrogram() {
                    return TestResult::from_bool(true);
                }
            }

            TestResult::from_bool(false)
        }
        quickcheck(prop as fn(Vec<Euclid<[f64; 2]>>) -> TestResult);
    }

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
    use super::{Agglomerative, LinkageCriterion, minimal_distance};

    macro_rules! benches {
        ($($name: ident, $l: expr, $d: expr, $n: expr;)*) => {
            $(
                #[bench]
                fn $name(b: &mut Bencher) {
                    let mut rng = XorShiftRng::new_unseeded();
                    let points = (0..$n)
                        .map(|_| Euclid(rng.gen::<[f64; $d]>()))
                        .collect::<Vec<_>>();

                    b.iter(|| Agglomerative::new(&points, $l))
                }
            )*
        }
    }

    benches! {
        single_slink_d1_n0010, LinkageCriterion::Single, 1,   10;
        single_slink_d1_n0100, LinkageCriterion::Single, 1,  100;
        single_slink_d1_n1000, LinkageCriterion::Single, 1, 1000;

        single_naive_d1_n0010, LinkageCriterion::Custom(&minimal_distance), 1,   10;
        single_naive_d1_n0100, LinkageCriterion::Custom(&minimal_distance), 1,  100;
        //single_naive_d1_n1000, LinkageCriterion::Custom(&minimal_distance), 1, 1000;

        complete_d1_n0010, LinkageCriterion::Complete, 1,   10;
        complete_d1_n0100, LinkageCriterion::Complete, 1,  100;
        //complete_d1_n1000, LinkageCriterion::Complete, 1, 1000;

        clink_d1_n0010, LinkageCriterion::CLINK, 1,   10;
        clink_d1_n0100, LinkageCriterion::CLINK, 1,  100;
        clink_d1_n1000, LinkageCriterion::CLINK, 1, 1000;
    }
}
