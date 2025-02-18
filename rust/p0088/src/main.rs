use std::{cell::RefCell, collections::BTreeMap, fmt, fmt::Display, ops::Deref};

fn main() {
    let v = factorization::SinglePrimeFactorizationIter::new(3, 5).collect::<Vec<_>>();
    println!("{}\n", v.len());
    for factorization in v {
        println!("{factorization}");
    }
}

/// Returns the least positive integer n for which there exists a sequence of k positive numbers
/// where both the sum and the product of the sequence is equal to n.
///
/// The behavior is undefined for k < 2.
fn get_min_ps_number(k: u32) -> u32 {
    // TODO: make primes an argument so it can be shared between calls
    let mut primes = vec![2, 3, 5, 7, 11, 13, 17, 19];
    let mut n = 2;
    loop {
        let factors = get_prime_factors(n, &mut primes);
        if is_ps_number(n, &factors, k) {
            return n;
        }

        n += 1;
    }
}

fn get_prime_factors(mut n: u32, primes: &mut Vec<u32>) -> BTreeMap<u32, u32> {
    let mut factors = BTreeMap::new();

    for prime_index in 0.. {
        if prime_index == primes.len() {
            add_prime(primes);
        }

        let p = primes[prime_index];
        while n % p == 0 {
            n /= p;
            factors.entry(p).and_modify(|m| *m += 1).or_insert(1);
        }

        if n == 1 {
            break;
        }
    }

    factors
}

fn get_factorizations(prime_factorization: &BTreeMap<u32, u32>) {}

mod partition {
    use crate::{BTreeMap, Deref, RefCell};

    struct SizedPartitionSubIter(Box<RefCell<SizedPartitionIter>>);

    impl SizedPartitionSubIter {
        fn new(n: u32, len: u32, prefix: u32) -> Self {
            SizedPartitionSubIter(Box::new(RefCell::new(SizedPartitionIter::new_with_prefix(
                n, len, prefix,
            ))))
        }
    }

    impl Deref for SizedPartitionSubIter {
        type Target = Box<RefCell<SizedPartitionIter>>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    pub struct SizedPartitionIter {
        n: u32,
        prefix: u32,
        len: u32,
        sub_iter: Option<SizedPartitionSubIter>,
    }

    impl SizedPartitionIter {
        pub fn new(n: u32, len: u32) -> Self {
            SizedPartitionIter::with_prefix(n, len, 1)
        }

        fn with_prefix(n: u32, len: u32, prefix: u32) -> Self {
            assert!(len > 0);
            assert!(n >= len);
            assert!(prefix * len <= n);

            let sub_iter = if len == 1 {
                None
            } else {
                Some(SizedPartitionSubIter::new(n - prefix, len - 1, prefix))
            };

            SizedPartitionIter {
                n,
                prefix,
                len,
                sub_iter,
            }
        }
    }

    impl Iterator for SizedPartitionIter {
        type Item = BTreeMap<u32, u32>;

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(sub_iter) = &mut self.sub_iter {
                // For some reason the call to borrow_mut upsets the borrow checker if it's in the if
                // let statement, but is fine if we assign the result of the call to next to a
                // variable.
                let partition_option = sub_iter.borrow_mut().next();
                if let Some(mut partition) = partition_option {
                    partition
                        .entry(self.prefix)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);

                    Some(partition)
                } else if (self.prefix + 1) * self.len <= self.n {
                    self.prefix += 1;

                    let new_sub_iter =
                        SizedPartitionSubIter::new(self.n - self.prefix, self.len - 1, self.prefix);

                    let mut partition = new_sub_iter.borrow_mut().next().unwrap();
                    partition
                        .entry(self.prefix)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);

                    *sub_iter = new_sub_iter;

                    Some(partition)
                } else {
                    None
                }
            } else {
                // Since self.sub_iter == None (which implies self.len == 1), self.prefix <= self.n if
                // this is the first time next has been called.
                if self.prefix <= self.n {
                    self.prefix = self.n + 1;

                    let mut partition = BTreeMap::new();
                    partition.insert(self.n, 1);

                    Some(partition)
                } else {
                    None
                }
            }
        }
    }

    pub struct PartitionIter {
        n: u32,
        len: u32,
        max_len: u32,
        sized_partition_iter: SizedPartitionIter,
    }

    impl PartitionIter {
        pub fn new(n: u32) -> Self {
            Self::with_max_len(n, n)
        }

        pub fn with_max_len(n: u32, max_len: u32) -> Self {
            assert!(n >= max_len);

            PartitionIter {
                n,
                len: 1,
                max_len,
                sized_partition_iter: SizedPartitionIter::new(n, 1),
            }
        }
    }

    impl Iterator for PartitionIter {
        type Item = BTreeMap<u32, u32>;

        fn next(&mut self) -> Option<Self::Item> {
            let partition_option = self.sized_partition_iter.next();

            if partition_option.is_some() {
                partition_option
            } else if self.len < self.max_len {
                self.len += 1;
                self.sized_partition_iter = SizedPartitionIter::new(self.n, self.len);
                // A new SizedPartitionIter should always return at least one partition
                self.sized_partition_iter.next()
            } else {
                None
            }
        }
    }
}

mod factorization {
    use crate::{fmt, partition::PartitionIter, BTreeMap, Deref, Display, RefCell};

    pub struct Factorization(BTreeMap<u32, u32>);

    impl Deref for Factorization {
        type Target = BTreeMap<u32, u32>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Display for Factorization {
        fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            let mut i = self.iter();

            let (n, m) = i.next().unwrap();
            write!(f, "{n}")?;
            for _ in 0..m - 1 {
                write!(f, "*{n}")?;
            }

            for (n, m) in i {
                for _ in 0..*m {
                    write!(f, "*{n}")?;
                }
            }

            Result::Ok(())
        }
    }

    pub struct SinglePrimeFactorizationIter {
        p: u32,
        partition_iter: PartitionIter,
    }

    impl SinglePrimeFactorizationIter {
        pub fn new(p: u32, k: u32) -> Self {
            Self {
                p,
                partition_iter: PartitionIter::new(k),
            }
        }
    }

    impl Iterator for SinglePrimeFactorizationIter {
        type Item = Factorization;

        fn next(&mut self) -> Option<Self::Item> {
            let partition = self.partition_iter.next()?;
            let factorization = Factorization(
                partition
                    .into_iter()
                    .map(|(n, c)| (self.p.pow(n), c))
                    .collect(),
            );

            Some(factorization)
        }
    }

    type Distribution = BTreeMap<u32, u32>;

    struct DistributionIter {
        m_factorization: Factorization,
        p: u32,
        k: u32,
        unique_factor_distribution: BTreeMap<u32, u32>,
        distribution_iters: BTreeMap<u32, PartitionIter>,
        cur_distribution: BTreeMap<u32, BTreeMap<u32, u32>>,
    }

    impl DistributionIter {
        fn new(m_factorization: Factorization, p: u32, k: u32) -> Self {
            let mut unique_factor_distribution = m_factorization.keys().map(|&n| (n, 0)).collect();
            // HERE: I think PartitionIter::new(0) panics, I should probably fix that.
            let mut distribution_iters = Vec::new();

            Self {
                m_factorization,
                p,
                k,
                cur_distribution,
            }
        }
    }

    impl Iterator for DistributionIter {
        type Item = Factorization;

        fn next(&mut self) -> Option<Self::Item> {
            let factorization = self.m_factorization.iter().zip(&self.cur_distribution).flat_map(|((factor, copies), distribution)| )
            None
        }
    }

    struct FactorizationIter {
        p: u32,
        a: u32,
        k: u32,
        m_factorization_iter: Option<Box<RefCell<FactorizationIter>>>,
        cur_m_factorization: BTreeMap<u32, u32>,
        // TODO: rename PartitionIter to SizedPartitionIter and create a new PartitionIter that
        //          gives all partitions regardless of the length.
        k_partition_iter: Option<Box<RefCell<dyn Iterator<Item = BTreeMap<u32, u32>>>>>,
        cur_k_partition: Option<BTreeMap<u32, u32>>,
    }

    impl FactorizationIter {
        fn new(mut prime_factorization: BTreeMap<u32, u32>) -> Self {
            assert!(prime_factorization.len() > 0);

            let (p, a) = prime_factorization.pop_first().unwrap();
            let (m_factorization_iter, cur_m_factorization, k_partition_iter) =
                if prime_factorization.len() > 0 {
                    let mut i = FactorizationIter::new(prime_factorization);
                    let cur_m_factorization = i.next().unwrap();
                    let k_partition_iter = (0..=a).flat_map(move |len| PartitionIter::new(0));
                    let k_partition_iter = Some(Box::new(RefCell::new(k_partition_iter))
                        as Box<RefCell<dyn Iterator<Item = BTreeMap<u32, u32>>>>);
                    (
                        Some(Box::new(RefCell::new(i))),
                        cur_m_factorization,
                        k_partition_iter,
                    )
                } else {
                    (None, BTreeMap::new(), None)
                };

            FactorizationIter {
                p,
                a,
                k: 0,
                m_factorization_iter,
                cur_m_factorization,
                k_partition_iter,
                cur_k_partition: None,
            }
        }
    }

    impl Iterator for FactorizationIter {
        type Item = BTreeMap<u32, u32>;

        fn next(&mut self) -> Option<Self::Item> {
            None
        }
    }
}

// factorizations of 32:
//      Factors     Partition
//      32          5
//      2,16        1+4
//      4,8         2+3
//      2,2,8       1+1+3
//      2,4,4       1+2+2
//      2,2,2,4     1+1+1+2
//      2,2,2,2,2   1+1+1+1+1
// factorizations of 27:
//      Factors     Partition
//      27          3
//      3,9         1+2
//      3,3,3       1+1+1
//  factorizations of 864
//      Factors
//      27,32
//      864
//      ------------
//      2,16,27
//      2,432
//      16,54
//      ------------
//      4,8,27
//      4,216
//      8,108
//      ------------
//      2,2,8,27
//      2,2,216
//      2,8,54
//      ------------
//      2,4,4,27
//      2,4,108
//      4,4,54
//      ------------
//      2,2,2,4,27
//      2,2,2,108
//      2,2,4,54
//      ------------
//      2,2,2,2,2,27
//      2,2,2,2,54
//      ============
//      3,9,32
//      3,288
//      9,96
//      ------------
//      2,3,9,16
//      2,3,144
//      3,16,18
//      2,9,48
//      6,9,16
//
fn is_ps_number(n: u32, factors: &BTreeMap<u32, u32>, k: u32) -> bool {
    false
}

fn add_prime(primes: &mut Vec<u32>) {
    let n = primes.last().unwrap();

    // just trial division
    for m in n + 1.. {
        for p in primes.iter() {
            if p * p > m {
                primes.push(m);
                return;
            }

            if m % p == 0 {
                break;
            }
        }
    }
}
