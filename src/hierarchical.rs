use std::f64;

use {Point};


/// Hierarchical linkage criteria.
pub enum Linkage {
    Complete,
    Single
}


/// Agglomerative clustering, i.e., hierarchical bottum-up clustering.
///
/// TODO: Add a description
///
/// # Examples
///
/// ```rust
/// use cogset::{Agglomerative, Euclid, Linkage};
///
/// let data = [Euclid([0.0, 0.0]),
///             Euclid([1.0, 0.5]),
///             Euclid([0.2, 0.2]),
///             Euclid([0.3, 0.8]),
///             Euclid([0.0, 1.0])];
/// let k = 3;
///
/// let agglomerative = Agglomerative::new(&data, Linkage::Complete, 0.5);
///
/// println!("{:?}", agglomerative.clusters());
/// ```
pub struct Agglomerative<'a, P: Point + 'a> {
    clusters: Vec<Vec<&'a P>>
}

impl<'a, P: Point> Agglomerative<'a, P> {
    pub fn new<T>(data: T, linkage: Linkage, threshold: f64) -> Agglomerative<'a, P>
        where T: IntoIterator<Item=&'a P>
    {
        let mut clusters: Vec<_> = data.into_iter().map(|p| vec![p]).collect();

        let linkage: fn(&Vec<&P>, &Vec<&P>) -> f64 = match linkage {
            Linkage::Complete => complete_linkage,
            Linkage::Single => single_linkage
        };

        while clusters.len() > 1 {
            let (d, i, j) = merge(&clusters, linkage);

            if d > threshold {
                break;
            }

            let merge = clusters.swap_remove(j);
            clusters[i].extend(merge);
        }

        Agglomerative {
            clusters: clusters
        }
    }

    pub fn clusters(&'a self) -> &'a Vec<Vec<&'a P>> {
        &self.clusters
    }
}

fn merge<P, F>(clusters: &Vec<Vec<&P>>, linkage: F) -> (f64, usize, usize)
    where P: Point, F: Fn(&Vec<&P>, &Vec<&P>) -> f64
{
    let mut merge_distance = f64::INFINITY;
    let mut merge_pair = (0, 0);

    for i in 0..clusters.len() {
        for j in i+1..clusters.len() {
            let distance = linkage(&clusters[i], &clusters[j]);

            if distance < merge_distance {
                merge_distance = distance;
                merge_pair = (i, j);
            }
        }
    }

    (merge_distance, merge_pair.0, merge_pair.1)
}

#[allow(non_snake_case)]
fn complete_linkage<P: Point>(A: &Vec<&P>, B: &Vec<&P>) -> f64 {
    let mut maximal_distance = f64::NEG_INFINITY;

    for a in A {
        for b in B {
            let distance = a.dist(b);

            if distance > maximal_distance {
                maximal_distance = distance;
            }
        }
    }

    maximal_distance
}

#[allow(non_snake_case)]
fn single_linkage<P: Point>(A: &Vec<&P>, B: &Vec<&P>) -> f64 {
    let mut minimal_distance = f64::INFINITY;

    for a in A {
        for b in B {
            let distance = a.dist(b);

            if distance < minimal_distance {
                minimal_distance = distance;
            }
        }
    }

    minimal_distance
}
