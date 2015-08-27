use std::f64;
use std::borrow::Borrow;
use std::iter::Cloned;

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
pub struct Agglomerative<'a, P: 'a + Point> {
    points: Vec<&'a P>,
    distances: Vec<Vec<f64>>,
    linkage: Linkage
}

impl<'a, P: 'a + Point> Agglomerative<'a, P> {
    pub fn new<T>(data: T, linkage: Linkage) -> Agglomerative<'a, P>
        where T: IntoIterator<Item=&'a P>
    {
        let points:Vec<_> = data.into_iter().collect();
        let n = points.len();

        // pre-compute distances, using as few space as possible,
        // we assume that d(i,j) = d(j,i)
        let distances = (0..n).map(|i| {
            (i+1..n).map(|j| {
                points[i].dist(points[j])
            }).collect()
        }).collect();

        Agglomerative {
            points: points,
            distances: distances,
            linkage: linkage
        }
    }

    pub fn clusters(&'a self, threshold: f64) -> Vec<Vec<&'a P>> {
        let mut clusters: Vec<Vec<_>> = (0..self.points.len()).map(|i| vec![i]).collect();

        // first abort criterion, there must be two clusters to merge them
        while clusters.len() > 1 {
            let mut merge_distance = f64::INFINITY;
            let mut merge_pair = (0, 0);

            for i in 0..clusters.len() {
                for j in i+1..clusters.len() {
                    let distance = match self.linkage {
                        Linkage::Complete => complete_linkage(|i,j| self.distance(i, j), &clusters[i], &clusters[j]),
                        Linkage::Single => single_linkage(|i,j| self.distance(i, j), &clusters[i], &clusters[j])
                    };

                    if distance < merge_distance {
                        merge_distance = distance;
                        merge_pair = (i, j);
                    }
                }
            }

            let (d, i, j) = (merge_distance, merge_pair.0, merge_pair.1);
            assert!(d > 0.0);
            assert!(i != j);

            if d > threshold {
                break;
            }

            // make sure that we remove the cluster with the higher index
            let (r, e) = if i > j {
                (i, j)
            } else {
                (j, i)
            };

            let merge = clusters.swap_remove(r);
            clusters[e].extend(merge);
        }

        clusters.iter().map(|cluster|
            cluster.iter().map(|&i| self.points[i]).collect()
        ).collect()
    }

/*
    pub fn dendrogram(&'a self) -> Box<Dendrogram<'a, P>> {
        //unimplemented!();

        let mut clusters: Vec<Box<Dendrogram<'a, P>>> = self.clusters.iter().map(|&p| Box::new(Dendrogram::Leaf(p))).collect();

        while clusters.len() > 1 {
            let mut merge_distance = f64::INFINITY;
            let mut merge_pair = (0, 0);

            for i in 0..clusters.len() {
                for j in i+1..clusters.len() {
                    let a = &clusters[i];
                    let b = &clusters[j];

                    let distance = complete_linkage2(a.borrow(), b.borrow());

                    if distance < merge_distance {
                        merge_distance = distance;
                        merge_pair = (i, j);
                    }
                }
            }

            let (i, j) = if merge_pair.1 > merge_pair.0 {
                (merge_pair.1, merge_pair.0)
            } else {
                merge_pair
            };



            let a = clusters.swap_remove(i);
            let b = clusters.swap_remove(j);

            clusters.push(Box::new(Dendrogram::Branch(merge_distance, a, b)));
        }

        clusters.pop().unwrap()
    }*/

    fn distance(&self, i: usize, j: usize) -> f64 {
        assert!(i != j);
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        self.distances[i][j - i - 0]
    }
}

#[derive(Clone, Debug)]
pub enum Dendrogram<'a, P: 'a + Point> {
    Branch(f64, Box<Dendrogram<'a, P>>, Box<Dendrogram<'a, P>>),
    Leaf(&'a P)
}

pub struct Elements<'a, P: 'a + Point> {
    items: Vec<&'a Dendrogram<'a, P>>
}

impl<'a, P: 'a + Point> Iterator for Elements<'a, P> {
    type Item = &'a P;

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.items.pop().and_then(|item| {
        println!("nexted2");
            match item.borrow() {
                &Dendrogram::Branch(_, ref a, ref b) => {
                    self.items.push(a);
                    self.items.push(b);
                    println!("nexted1");

                    self.next()
                },
                &Dendrogram::Leaf(p) => Some(p)
            }
        });

        if x.is_none() {
            println!("iter done");
        }

        x
    }
}

impl<'a, P: 'a + Point> IntoIterator for &'a Dendrogram<'a, P> {
    type Item = &'a P;
    type IntoIter = Elements<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        Elements {
            items: vec![self]
        }
    }
}

#[cfg(test)]
mod tests {
    use {Euclid, Agglomerative, Linkage};

    #[test]
    fn asd() {

        let data = [Euclid([1.0f64]), Euclid([4.0]), Euclid([5.0]), Euclid([6.0]), Euclid([7.0])];

        let a = Agglomerative::new(&data, Linkage::Complete, 0.0);

        //println!("data {:#?}", a.dendrogram());
        //assert!(false);
    }
}


#[allow(non_snake_case)]
fn complete_linkage<D>(d: D, A: &[usize], B: &[usize]) -> f64
    where D: Fn(usize, usize) -> f64
{
    let mut maximal_distance = f64::NEG_INFINITY;

    for a in A {
        for b in B {
            let distance = d(*a, *b);

            if distance > maximal_distance {
                maximal_distance = distance;
            }
        }
    }

    maximal_distance
}

#[allow(non_snake_case)]
fn single_linkage<D>(d: D, A: &[usize], B: &[usize]) -> f64
    where D: Fn(usize, usize) -> f64
{
    let mut minimal_distance = f64::INFINITY;

    for a in A {
        for b in B {
            let distance = d(*a, *b);

            if distance < minimal_distance {
                minimal_distance = distance;
            }
        }
    }

    minimal_distance
}
