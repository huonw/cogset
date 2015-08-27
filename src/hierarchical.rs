use std::cmp::{min, max};
use std::f64::{INFINITY, NEG_INFINITY};

use {Point};

// TODO: Add tests
// TODO: More efficient `Elements` Iterator
// TODO: Implement `SLINK` as single-linkage criterion
// TODO: Implement `CLINK` as alternative complete-linkage criterion


/// Hierarchical linkage criteria.
///
/// The linkage decides how the distance between clusters is computed.
///
/// <script src="https://is.gd/BFouBe"></script>
#[derive(Copy, Clone, Debug)]
pub enum LinkageCriterion {
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
    Single
}


/// Agglomerative clustering, i.e., hierarchical bottum-up clustering.
///
/// Starting with each element in its own cluster, iteratively merge the two most similar
/// clusters until a certain distance threshold is reached. The similarity between two
/// clusters is mainly defined by the linkage criterion and the `Point`'s distance
/// metric.
///
/// The inter-point distances are pre-computed and cached in a distance matrix.
///
/// * **Complete-linkage**: Currently a naive implementation with O( n^3 ) time
///   and O( n^2 ) space complexity.
/// * **Single-linkage**: Currently a naive implementation with O( n^3 ) time
///   and O( n^2 ) space complexity.
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
/// let agglomerative = Agglomerative::new(&data);
///
/// println!("{:?}", agglomerative.clusters(LinkageCriterion::Complete, 0.4));
/// println!("{:?}", agglomerative.clusters(LinkageCriterion::Complete, 0.6));
///
/// println!("{:?}", agglomerative.dendrogram(LinkageCriterion::Complete));
/// ```
///
/// <script src="https://is.gd/BFouBe"></script>
#[derive(Clone, Debug)]
pub struct Agglomerative<'a, P: 'a + Point> {
    points: Vec<&'a P>,
    distances: Vec<Vec<f64>>
}

impl<'a, P: 'a + Point> Agglomerative<'a, P> {
    /// Pre-compute the inter-point distances of `data`.
    pub fn new<T>(data: T) -> Agglomerative<'a, P>
        where T: IntoIterator<Item=&'a P>
    {
        let points:Vec<_> = data.into_iter().collect();
        let n = points.len();

        // pre-compute distances, using as few space as possible,
        // we assume that d(i,j) = d(j,i) and d(i,i) = 0
        let distances = (0..n).map(|i| {
            (i+1..n).map(|j| {
                points[i].dist(points[j])
            }).collect()
        }).collect();

        Agglomerative {
            points: points,
            distances: distances
        }
    }

    /// Perform the actual clustering.
    ///
    /// Starting with each point in its own cluster successively merge two clusters
    /// ,determined by `linkage`, until either only one is left or the maximal distance
    /// `threshold` has been exceeded.
    pub fn clusters(&'a self, linkage: LinkageCriterion, threshold: f64) -> Vec<Vec<&'a P>> {
        self.merge(linkage, threshold).iter().map(|cluster| {
            cluster.into_iter().map(|i| self.points[*i]).collect()
        }).collect()
    }

    /// Calculate the hierarchical dendrogram for the givan linkage criterion.
    pub fn dendrogram(&'a self, linkage: LinkageCriterion) -> Box<Dendrogram<&'a P>> {
        let mut x = self.merge(linkage, INFINITY);
        assert!(x.len() == 1);
        let x = x.pop().unwrap();

        self.convert(&x)
    }

    fn convert(&'a self, root: &Dendrogram<usize>) -> Box<Dendrogram<&'a P>> {
        Box::new(match root {
            &Dendrogram::Branch(d, ref a, ref b) =>
                Dendrogram::Branch(d, self.convert(a), self.convert(b)),
            &Dendrogram::Leaf(i) => Dendrogram::Leaf(self.points[i])
        })
    }

    fn merge(&'a self, linkage: LinkageCriterion, threshold: f64) -> Vec<Box<Dendrogram<usize>>> {
        let mut clusters: Vec<_> = (0..self.points.len()).map(|i|
            Box::new(Dendrogram::Leaf(i))).collect();

        // first abort criterion, there must be two clusters to merge them
        while clusters.len() > 1 {
            let (d, i, j) = (0..clusters.len())
                .flat_map(|i| (i+1..clusters.len()).map(move |j| (i, j)))
                .map(|(i, j)| (self.link(linkage, &clusters[i], &clusters[j]), i, j))
                .fold((INFINITY, 0, 0), |a, b| if a.0 < b.0 { a } else { b });

            // second abort criterium, max cluster distance
            if d > threshold {
                break;
            }

            let a = clusters.swap_remove(max(i, j));
            let b = clusters.swap_remove(min(i, j));

            clusters.push(Box::new(Dendrogram::Branch(d, a, b)));
        }

        clusters
    }

    #[allow(non_snake_case)]
    fn link(&self, linkage: LinkageCriterion, A: &Dendrogram<usize>, B: &Dendrogram<usize>) -> f64 {
        match linkage {
           LinkageCriterion::Complete => self.complete_linkage(A, B),
           LinkageCriterion::Single => self.single_linkage(A, B)
       }
    }

    #[allow(non_snake_case)]
    fn complete_linkage(&self, A: &Dendrogram<usize>, B: &Dendrogram<usize>) -> f64 {
        let mut maximal_distance = NEG_INFINITY;

        for &a in A {
            for &b in B {
                let distance = self.distance(a, b);

                if distance > maximal_distance {
                    maximal_distance = distance;
                }
            }
        }

        maximal_distance
    }

    #[allow(non_snake_case)]
    fn single_linkage(&self, A: &Dendrogram<usize>, B: &Dendrogram<usize>) -> f64 {
        let mut minimal_distance = INFINITY;

        for &a in A {
            for &b in B {
                let distance = self.distance(a, b);

                if distance < minimal_distance {
                    minimal_distance = distance;
                }
            }
        }

        minimal_distance
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        assert!(i != j);
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        self.distances[i][j - i - 1]
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
