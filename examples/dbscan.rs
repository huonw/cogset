extern crate cogset;

fn main() {
    use std::str;
    use cogset::{Dbscan, BruteScan, Euclid};

    fn write_points<I>(output: &mut [u8; 76], byte: u8, it: I)
        where I: Iterator<Item = Euclid<[f64; 1]>>
    {
        for p in it { output[(p.0[0] * 30.0) as usize] = byte; }
    }

    // the points we're going to cluster, considered as points in ℝ
    // with the conventional distance.
    let points = [Euclid([0.25]), Euclid([0.9]), Euclid([2.0]), Euclid([1.2]),
                  Euclid([1.9]), Euclid([1.1]),  Euclid([1.35]), Euclid([1.85]),
                  Euclid([1.05]), Euclid([0.1]), Euclid([2.5]), Euclid([0.05]),
                  Euclid([0.6]), Euclid([0.55]), Euclid([1.6])];

    // print the points before clustering
    let mut original = [b' '; 76];
    write_points(&mut original, b'x', points.iter().cloned());
    println!("{}", str::from_utf8(&original).unwrap());

    // set-up the data structure that will manage the queries that
    // Dbscan needs to do.
    let scanner = BruteScan::new(&points);

    // create the clusterer: we need 3 points to consider a group a
    // cluster, and we're only looking at points 0.2 a part.
    let min_points = 3;
    let max_distance = 0.2;
    let mut dbscan = Dbscan::new(scanner, max_distance, min_points);

    let mut clustered = [b' '; 76];

    // run over all the clusters, writing each to the output
    for (i, cluster) in dbscan.by_ref().enumerate() {
        // `cluster` is a vector of indices into `points`, not the
        // points themselves, so lets map back to the points.
        let actual_points = cluster.iter().map(|idx| points[*idx]);

        write_points(&mut clustered, b'0' + i as u8,
                     actual_points)
    }
    // now run over the noise points, i.e. points that aren't close
    // enough to others to be in a cluster.
    let noise = dbscan.noise_points();
    write_points(&mut clustered, b'.',
                 noise.iter().map(|idx| points[*idx]));

    // print the numbered clusters
    println!("{}", str::from_utf8(&clustered).unwrap());
}
