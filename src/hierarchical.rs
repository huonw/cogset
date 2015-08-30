use std::borrow::Borrow;
use std::cmp::{min, max};
use std::f64::{INFINITY, NEG_INFINITY};
use std::mem::replace;

use {Point};

// TODO: More efficient `Elements` Iterator
// TODO: Implement `CLINK` as alternative complete-linkage criterion


/// A lookup function for point distances, i.e., `d(i,j)`.
pub type DistanceFunction = Fn(usize, usize) -> f64;

/// A function to compute the inter-cluster distance between set `A` and `B`.
pub type LinkageFunction = Fn(&DistanceFunction, &Dendrogram<usize>, &Dendrogram<usize>) -> f64;


/// Hierarchical linkage criteria.
///
/// The linkage decides how the distance between clusters is computed.
///
/// <script src="https://is.gd/BFouBe"></script>
#[derive(Copy, Clone)]
pub enum LinkageCriterion<'a> {
    /// Maximum or complete-linkage clustering.
    ///
    /// <p>
    /// $$\max_{a \in A,\, b \in B} \, d(a, b)$$
    /// </p>
    Complete,

    /// Minimum or single-linkage clustering.
    ///
    /// <p>
    /// $$\min_{a \in A,\, b \in B} \, d(a, b)$$
    /// </p>
    Single,

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
/// * **Complete-linkage**: Currently a naive implementation with O( n^3 ) time
///   and O( n^2 ) space complexity.
/// * **Single-linkage**: Implementation of the optimal SLINK [1] algorithm with O( n^2 ) time
///   and O( n ) space complexity.
/// * **Custom**: _At least_ O( n^3 ) time and O( n^2 ) space complexity. The exact
///   complexity depends on the linkage criterion.
///
/// [1]: Sibson, R. (1973). SLINK: an optimally efficient algorithm for the single-link
///      cluster method. The Computer Journal, 16(1), 30-34.
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
    let n = points.len();

    // pre-compute distances, using as few space as possible,
    // we assume that d(i,j) = d(j,i) and d(i,i) = 0
    let distances: Vec<Vec<f64>> = (0..n).map(|i| {
        (i+1..n).map(|j| points[i].dist(points[j])).collect()
    }).collect();

    // TODO: Why is the move required? How could this closure outlive the currenct function?
    let d = move |i: usize, j: usize| {
        if i == j {
            0.0
        } else {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            distances[i][j - i - 1]
        }
    };

    let mut clusters: Vec<_> = (0..n).map(|i| Box::new(Dendrogram::Leaf(i))).collect();

    // there must be two clusters to merge them
    while clusters.len() > 1 {
        let mut min_dist = INFINITY;
        let mut merge = (0, 0);

        // find the next two clusters to merge
        for i in (0..clusters.len()) {
            for j in (i+1..clusters.len()) {
                let distance = linkage(&d, &clusters[i], &clusters[j]);

                if distance < min_dist {
                    min_dist = distance;
                    merge = (i, j);
                }
            }
        }

        // remove first the one with the higher index
        let a = clusters.swap_remove(max(merge.0, merge.1));
        let b = clusters.swap_remove(min(merge.0, merge.1));

        clusters.push(Box::new(Dendrogram::Branch(min_dist, b, a)));
    }

    fn convert<'a, P: Point>(points: &Vec<&'a P>, root: &Dendrogram<usize>) -> Box<Dendrogram<&'a P>> {
        Box::new(match root {
            &Dendrogram::Branch(d, ref a, ref b) =>
                Dendrogram::Branch(d, convert(points, a), convert(points, b)),
            &Dendrogram::Leaf(i) => Dendrogram::Leaf(points[i])
        })
    }

    convert(points, &clusters.pop().unwrap())
}


fn slink<'a, P: Point>(points: &Vec<&'a P>) -> Box<Dendrogram<&'a P>> {
    // clustering while creating the proper pointer representation
    let mut pi = vec![0; points.len()];
    let mut lambda = vec![0.0; points.len()];
    let mut em = vec![0.0; points.len()];

    for n in (0..points.len()) {
        pi[n] = n;
        lambda[n] = INFINITY;

        for i in (0..n) {
            em[i] = points[i].dist(points[n]);
        }

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
fn minimal_distance(d: &DistanceFunction, A: &Dendrogram<usize>, B: &Dendrogram<usize>)
    -> f64
{
    let mut minimal_distance = INFINITY;

    for &a in A {
        for &b in B {
            let distance = d(a, b);

            if distance < minimal_distance {
                minimal_distance = distance;
            }
        }
    }

    minimal_distance
}

#[allow(non_snake_case)]
fn maximal_distance(d: &DistanceFunction, A: &Dendrogram<usize>, B: &Dendrogram<usize>)
    -> f64
{
    let mut maximal_distance = NEG_INFINITY;

    for &a in A {
        for &b in B {
            let distance = d(a, b);

            if distance > maximal_distance {
                maximal_distance = distance;
            }
        }
    }

    maximal_distance
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
    use quickcheck::{Arbitrary, Gen, TestResult, quickcheck};

    use Euclid;
    use super::{Agglomerative, LinkageCriterion, minimal_distance};

    #[test]
    fn test_slink() {
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

    macro_rules! gen {
        ($r: expr, 1) => {
            [$r.gen::<f64>()]
        };
        ($r: expr, 2) => {
            [$r.gen::<f64>(), $r.gen::<f64>()]
        };
        ($r: expr, 3) => {
            [$r.gen::<f64>(), $r.gen::<f64>(), $r.gen::<f64>()]
        };
        ($r: expr, 9) => {
            [$r.gen::<f64>(), $r.gen::<f64>(), $r.gen::<f64>(),
             $r.gen::<f64>(), $r.gen::<f64>(), $r.gen::<f64>(),
             $r.gen::<f64>(), $r.gen::<f64>(), $r.gen::<f64>()]
        };
    }

    macro_rules! benches {
        ($($name: ident, $l: expr, $d: tt, $n: expr;)*) => {
            $(
                #[bench]
                fn $name(b: &mut Bencher) {
                    let mut rng = XorShiftRng::new_unseeded();
                    let points = (0..$n)
                        .map(|_| Euclid(gen!(rng, $d)))
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
    }
}
