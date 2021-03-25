use font8x8::BASIC_UNICODE;
use itertools::*;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph};
use rand::{
    distributions::{Uniform, WeightedIndex},
    prelude::{IteratorRandom, *},
    Rng, SeedableRng,
};
use rand_xorshift::XorShiftRng;

mod brain;
use brain::*;

type UsedRng = XorShiftRng;

fn main() {
    let font = BASIC_UNICODE;
    let possible_chars = ('a'..='z').chain('A'..='Z').collect_vec();

    let mut rng = UsedRng::from_entropy();
    let mut param = RandNetParam::default(3, 1);
    let mutparam = MutateParam::default();
    param.size = Uniform::new(20, 50);

    let mut alg = Alg::new(
        (0..100).fold(Vec::new(), |mut acc, _| {
            acc.push(Brain::rand_net(&RandNetParam::default(3, 1), &mut rng));
            acc
        }),
        |net, rng: &mut UsedRng, _: &()| {
            let a: f32 = rng.gen_range(-10.0..=10.0);
            let b: f32 = rng.gen_range(-10.0..=10.0);
            // 1: add, 2: sub, 3: div, 4: mult, 5: raise 6: root
            let op = rng.gen_range(1..=6);
            let expected_out = match op {
                1 => a + b,
                2 => a - b,
                3 => a / b,
                4 => a * b,
                5 => a.powf(b),
                6 => a.powf(1.0 / b),
                _ => panic!("bruh"),
            };
            (0..5).map(|_| net.run(&[a, b, op as f32])[0] - expected_out).sum()
        },
    );


    for _ in 0.. {
        alg.run(&(), &mut rng);
        alg.repopulate(&mutparam, &mut rng);
        println!("{}", alg.nets[alg.nets.len()/2].1);
        alg.zero();
    }
}

fn populate_node<'a, R, I>(
    net: &mut StableGraph<Node, f32>,
    connected_nodes: I,
    rng: &mut R,
    node_size: usize,
    conn_stregnth: Uniform<f32>,
    node: &NodeIndex,
) where
    R: Rng,
    I: Iterator<Item = &'a NodeIndex>,
{
    connected_nodes
        .choose_multiple(rng, node_size)
        .iter()
        .for_each(|&other_node| {
            net.add_edge(*other_node, *node, conn_stregnth.sample(rng));
        });
}

struct Alg<F, R: Rng> {
    fitness_fn: fn(brain: &mut Brain, rng: &mut R, extra_data: &F) -> f32, // user-defined function that generates ins and determines fitness
    nets: Vec<(Brain, f32)>,
}

impl<F, R: Rng> Alg<F, R> {
    fn new(
        nets: Vec<Brain>,
        fitness_fn: fn(&mut Brain, &mut R, extra_data: &F) -> f32,
    ) -> Alg<F, R> {
        let nets = nets.into_iter().map(|net| (net, 0.0)).collect_vec();
        Alg { fitness_fn, nets }
    }

    fn run(&mut self, fitness_data: &F, rng: &mut R) {
        let fitness_fn = self.fitness_fn;
        let nets = &mut self.nets;
        let data = fitness_data;
        for (net, fitness) in nets.iter_mut() {
            *fitness += fitness_fn(net, rng, fitness_data).abs();
        }
    }

    /// sets all of the node outputs to zero (resets the memory of the net)
    fn zero(&mut self) {
        for (_, out) in &mut self.nets {
            *out = 0.0;
        }
    }

    fn repopulate(&mut self, param: &MutateParam, rng: &mut R) {
        let nets = &mut self.nets;
        let good_nets = nets
            .iter()
            .enumerate()
            .sorted_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
            .collect_vec()
            .split_at(nets.len() / 4)
            .0
            .iter()
            .map(|(net, _)| *net)
            .collect_vec();

        let weighted_prob =
            WeightedIndex::new(nets.iter().map(|(_, fitness)| fitness.abs()).collect_vec())
                .unwrap();

        rng.sample_iter(weighted_prob)
            .take(nets.len() / 4)
            .collect_vec()
            .into_iter()
            .for_each(|i| {
                nets[i].0 = nets[*good_nets.choose(rng).unwrap()]
                    .0
                    .clone()
                    .mutate(&param, rng)
            });
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{Brain, MutateParam, RandNetParam};

    #[test]
    fn simple_test_net() {
        let mut rng = thread_rng();
        let mut brain = Brain::rand_net(&RandNetParam::default(5, 3), &mut rng);
        for _ in 0..20 {
            println!("{:?}", brain.run(&[2.1, 2.4, 3.5, 2.4, 1.5]));
        }
    }

    #[test]
    fn test_mutate() {
        let mut rng = thread_rng();
        let mut brain = Brain::rand_net(&RandNetParam::default(5, 3), &mut rng);
        for _ in 0..20 {
            println!("{:?}", brain.run(&[2.0, 2.4, 3.5, 2.4, 1.5]));
        }
        brain = brain.mutate(&MutateParam::default(), &mut rng);
        for _ in 0..20 {
            println!("{:?}", brain.run(&[2.1, 2.4, 3.5, 2.4, 1.5]));
        }
    }
}

/*
struct Node {
    activation_fn: fn(f32) -> f32,
    conns: Vec<Conn>,
    out: f32,
}

impl Node {
    fn run(&mut self) {
        let conns_total = self.conn
        self.out = (self.activation_fn)(conns_total);
    }
}

struct Conn {
    stregnth: f32,
    connects_to: usize,
}

struct Brain {
    inputs: Vec<f32>,
    hidden_nodes: Vec<Node>,
    output_nodes: Vec<Node>,
}

impl Brain {
    fn collect_garbage(&mut self) {
        for node in self.hidden_nodes.iter_mut() {
            unsafe {
                node.conns
                    .drain_filter(|conn| (*conn.as_ptr()).points_to.strong_count() == 0)
            };
        }
    }
}

fn main() {
    println!("Hello, world!");
}
*/
