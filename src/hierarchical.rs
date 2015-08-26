use std::f64;

use {Point};


/// Hierarchical linkage criteria.
pub enum Linkage {
    Complete
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

        let linkage: fn(&Vec<Vec<&P>>) -> (f64, usize, usize) = match linkage {
            Linkage::Complete => complete_linkage
        };

        while clusters.len() > 1 {
            let (d, i, j) = linkage(&clusters);

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

fn complete_linkage<P: Point>(clusters: &Vec<Vec<&P>>) -> (f64, usize, usize) {
    let mut merge_distance = f64::INFINITY;
    let mut merge_pair = (0, 0);

    for i in 0..clusters.len() {
        for j in i+1..clusters.len() {
            let mut maximal_distance = 0.0;

            for a in &clusters[i] {
                for b in &clusters[j] {
                    let distance = a.dist(b);

                    if distance > maximal_distance {
                        maximal_distance = distance;
                    }
                }
            }

            if maximal_distance < merge_distance {
                merge_distance = maximal_distance;
                merge_pair = (i, j);
            }
        }
    }

    (merge_distance, merge_pair.0, merge_pair.1)
}
