use test::Bencher;
use {Euclid, Points, RegionQuery, ListPoints};
use std::slice::Iter;
use std::ops::{RangeFrom, Range};
use std::iter::{Enumerate, Zip};
use rand::{XorShiftRng, Rng};

type E = Euclid<[f64; 1]>;
pub struct SortedScan<'a> {
    x: &'a [E]
}
impl<'a> Points for SortedScan<'a> {
    type Point = usize;
}
impl<'a> RegionQuery for SortedScan<'a> {
    type Neighbours = Neighbours<'a>;
    fn neighbours(&self, &point: &usize, eps: f64) -> Neighbours<'a> {
        Neighbours {
            this: self.x[point].0[0],
            eps: eps,
            backward: self.x[..point].iter().enumerate(),
            forward: (point..).zip(self.x[point..].iter())
        }
    }
}
struct Neighbours<'a> {
    this: f64,
    eps: f64,
    backward: Enumerate<Iter<'a, E>>,
    forward: Zip<RangeFrom<usize>, Iter<'a, E>>,
}
impl<'a> Iterator for Neighbours<'a> {
    type Item = (f64, usize);
    fn next(&mut self) -> Option<(f64, usize)> {
        if let Some((i, p)) = self.backward.next_back() {
            let dist = (p.0[0] - self.this).abs();
            if dist <= self.eps {
                return Some((dist, i))
            }

            self.backward = (&[]).iter().enumerate()
        }
        self.forward.next()
                    .and_then(|(i, p)| {
                        let dist = (p.0[0] - self.this).abs();
                        if dist <= self.eps {Some((dist, i))} else {None}
                    })
    }
}
impl<'a> ListPoints for SortedScan<'a> {
    type AllPoints = Range<usize>;
    fn all_points(&self) -> Range<usize> {
        0..self.x.len()
    }
}

pub fn run<T, F: FnMut(SortedScan, f64, usize) -> T>(b: &mut Bencher, n: usize, mut f: F) {
    let mut rng = XorShiftRng::new_unseeded();
    for _ in 0..100 { rng.gen::<u32>(); }
    let mut points = rng
        .gen_iter::<f64>()
        .take(n)
        .map(|f| Euclid([f]))
        .collect::<Vec<_>>();
    points.sort_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap());

    let eps = 2.0 / n as f64;
    let min_pts = 5;

    b.iter(|| f(SortedScan { x: &points }, eps, min_pts))
}
#[macro_export]
macro_rules! make_benches {
    ($e: expr) => {
        #[cfg(all(test, feature = "unstable"))]
        mod benches {
            use test::Bencher;
            #[bench]
            pub fn small(b: &mut Bencher) {
                ::benches::run(b, 30, $e)
            }
            #[bench]
            pub fn medium(b: &mut Bencher) {
                ::benches::run(b, 100, $e)
            }
            #[bench]
            pub fn large(b: &mut Bencher) {
                ::benches::run(b, 1_000, $e)
            }
            #[bench]
            pub fn huge(b: &mut Bencher) {
                ::benches::run(b, 10_000, $e)
            }
        }
    }
}
