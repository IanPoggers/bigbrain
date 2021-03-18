use font8x8::BASIC_UNICODE;
use itertools::*;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, EdgeDirection::Incoming};
use rand::{
    distributions::{Uniform, WeightedIndex},
    prelude::{IteratorRandom, *},
    Rng, SeedableRng,
};
use rand_xorshift::XorShiftRng;
use rayon::iter::ParallelIterator;
use rayon::{iter::*, prelude::*};

type UsedRng = XorShiftRng;

#[derive(Clone)]
struct Node {
    out: f32,
    activation_fn: fn(f32) -> f32,
}

impl Node {
    fn new(activation_fn: fn(f32) -> f32) -> Node {
        Node {
            out: 0f32,
            activation_fn,
        }
    }

    fn random_fn<R: Rng>(
        rng: &mut R,
        act_fns: &Vec<fn(f32) -> f32>,
        act_weights: &WeightedIndex<f32>,
    ) -> Node {
        Node::new(act_fns[act_weights.sample(rng)])
    }
}

#[derive(Clone)]
struct Brain {
    net: StableGraph<Node, f32>,
    input_nodes: Vec<NodeIndex>,
    hidden_nodes: Vec<NodeIndex>,
    output_nodes: Vec<NodeIndex>,
    inputs: usize,
    outputs: usize,
}

struct RandNetParam {
    /// The first value is the activation function, the second value is the relative probability of that function occurring in a node
    act_fns: Vec<(fn(f32) -> f32, f32)>,
    conn_count: Uniform<usize>,
    conn_stregnth: Uniform<f32>,
    size: Uniform<usize>,
    inputs: usize,
    outputs: usize,
}

struct MutateParam {
    act_fns: Vec<(fn(f32) -> f32, f32)>,
    node_insertion_prop: Uniform<f32>,
    conn_insertion_prop: Uniform<f32>,
    node_removal_prop: Uniform<f32>,
    conn_removal_prob: Uniform<f32>,
    new_conn_stregnth: Uniform<f32>,
    new_node_size: Uniform<usize>,
}

impl MutateParam {
    fn default() -> MutateParam {
        MutateParam {
            act_fns: vec![(|x| x / (1.0 + x.abs()), 1.0)],
            node_insertion_prop: Uniform::new(0.0, 0.06),
            conn_insertion_prop: Uniform::new(0.0, 0.06),
            node_removal_prop: Uniform::new(0.0, 0.04),
            conn_removal_prob: Uniform::new(0.0, 0.04),
            new_conn_stregnth: Uniform::new(-10.0, 10.0),
            new_node_size: Uniform::new(0, 20),
        }
    }
}

impl RandNetParam {
    fn default(inputs: usize, outputs: usize) -> RandNetParam {
        RandNetParam {
            act_fns: vec![(|x| x / (1.0 + x.abs()), 1.0)],
            conn_count: Uniform::new(5, 20),
            conn_stregnth: Uniform::new(-10.0, 10.0),
            size: Uniform::new(10, 20),
            inputs,
            outputs,
        }
    }
}

impl Brain {
    fn run(&mut self, ins: &[f32]) -> Vec<f32> {
        assert!(
            ins.len() == self.inputs as usize,
            "Wrong number of network inputs"
        );
        let net = &mut self.net;
        let input_nodes = net.externals(Incoming).collect_vec();
        let hidden_nodes = &mut self.hidden_nodes;
        let output_nodes = &mut self.output_nodes;

        for (inode, input) in input_nodes.iter().zip(ins) {
            net[*inode].out = *input;
        }

        hidden_nodes
            .par_iter()
            .chain(output_nodes.par_iter())
            .map(|&node| {
                let mut neighbors = net.neighbors_directed(node, Incoming).detach();

                let mut node_input = 0f32;
                while let Some((edge, connected_node)) = neighbors.next(net) {
                    node_input += net[connected_node].out * net[edge];
                }

                (node, (net[node].activation_fn)(node_input))
            })
            .collect::<Vec<_>>()
            .iter()
            .for_each(|(node, buffered_out)| net[*node].out = *buffered_out);

        self.output_nodes
            .iter()
            .map(|&node| net[node].out)
            .collect()
    }

    fn clear(&mut self) {
        let net = &mut self.net;
        net.node_indices().collect_vec().into_iter().for_each(|i| {
            net[i].out = 0.0;
        });
    }

    fn mutate<R: Rng>(mut self, param: &MutateParam, rng: &mut R) -> Self {
        let input_nodes = &self.input_nodes;
        let hidden_nodes = &mut self.hidden_nodes;
        let output_nodes = &self.output_nodes;
        let act_fns = &param.act_fns;
        let (act_fns, act_weights) = act_fns.iter().fold(
            (Vec::new(), Vec::new()),
            |(mut fns, mut weights), (a, b)| {
                fns.push(*a);
                weights.push(*b);
                (fns, weights)
            },
        );
        let act_weights = WeightedIndex::new(act_weights).unwrap();
        let net = &mut self.net;
        let node_removals =
            (param.node_removal_prop.sample(rng) * hidden_nodes.len() as f32) as usize;

        // remove random hidden nodes
        // @TODO: Can this be faster
        hidden_nodes
            .iter()
            .enumerate()
            .choose_multiple(rng, node_removals)
            .iter()
            .map(|(i, _)| *i)
            .collect_vec()
            .iter()
            .sorted()
            .rev()
            .for_each(|&i| {
                net.remove_node(hidden_nodes.remove(i));
            });

        // remove random connections
        let edge_removals =
            (param.conn_removal_prob.sample(rng) * net.edge_count() as f32) as usize;
        net.edge_indices()
            .choose_multiple(rng, edge_removals)
            .iter()
            .for_each(|&edge| {
                net.remove_edge(edge);
            });

        // @TODO: Make it so this isn't copy-pasted from rand_net
        // generate new nodes
        let new_nodes_count =
            (param.node_insertion_prop.sample(rng) * hidden_nodes.len() as f32) as usize;
        let mut new_nodes: Vec<NodeIndex> = Vec::new();
        for _ in 0..new_nodes_count {
            new_nodes.push(net.add_node(Node::random_fn(rng, &act_fns, &act_weights)));
        }

        // @TODO: Make it so this isn't copy-pasted from rand_net
        // connect new nodes
        for node in new_nodes.iter() {
            let node_size = param.new_node_size.sample(rng);
            populate_node(
                net,
                input_nodes.iter().chain(hidden_nodes.iter()),
                rng,
                node_size,
                param.new_conn_stregnth,
                node,
            );
            hidden_nodes.push(*node);
        }

        // Insert random connections
        let new_conns = (param.conn_insertion_prop.sample(rng)
            * (hidden_nodes.len() + output_nodes.len()) as f32) as usize;
        for _ in 0..new_conns {
            let node = hidden_nodes.iter().chain(output_nodes).choose(rng).unwrap();
            populate_node(
                net,
                input_nodes.iter().chain(hidden_nodes.iter()),
                rng,
                1,
                param.new_conn_stregnth,
                node,
            );
        }
        self
    }

    fn rand_net<R: Rng>(param: &RandNetParam, rng: &mut R) -> Brain {
        let size: usize = param.size.sample(rng);
        let inputs = param.inputs;
        let outputs = param.outputs;
        let act_fns = &param.act_fns;
        let (act_fns, act_weights) = act_fns.iter().fold(
            (Vec::new(), Vec::new()),
            |(mut fns, mut weights), (a, b)| {
                fns.push(*a);
                weights.push(*b);
                (fns, weights)
            },
        );
        let act_weights = WeightedIndex::new(act_weights).unwrap();

        let mut net: StableGraph<Node, f32> = StableGraph::with_capacity(size, 5);
        let mut input_nodes: Vec<NodeIndex> = Vec::new();
        let mut hidden_nodes: Vec<NodeIndex> = Vec::new();
        let mut output_nodes: Vec<NodeIndex> = Vec::new();

        // add input nodes
        for _ in 0..param.inputs {
            input_nodes.push(net.add_node(Node::new(|x| x)));
        }

        // generate nodes
        for _ in 0..size {
            hidden_nodes.push(net.add_node(Node::random_fn(rng, &act_fns, &act_weights)));
        }

        // generate output nodes
        for _ in 0..outputs {
            output_nodes.push(net.add_node(Node::new(|x| x)));
        }

        // generate connections
        for node in hidden_nodes.iter().chain(output_nodes.iter()) {
            let node_size = param.conn_count.sample(rng);
            populate_node(
                &mut net,
                input_nodes.iter().chain(hidden_nodes.iter()),
                rng,
                node_size,
                param.conn_stregnth,
                node,
            );
        }
        Brain {
            net,
            input_nodes,
            hidden_nodes,
            output_nodes,
            inputs,
            outputs,
        }
    }
}

struct Alg<F> {
    fitness_fn: fn(net_outs: &[f32], given_ins: &[f32], net: &Brain, extra_data: &F) -> f32, // user-defined function that determines fitness
    nets: Vec<(Brain, f32)>,
}

impl<F> Alg<F> {
    fn new(
        nets: Vec<Brain>,
        fitness_fn: fn(net_outs: &[f32], given_ins: &[f32], net: &Brain, extra_data: &F) -> f32,
    ) -> Alg<F> {
        let nets = nets.into_iter().map(|net| (net, 0.0)).collect_vec();
        Alg { fitness_fn, nets }
    }

    fn run(&mut self, ins: &[f32], fitness_data: &F) {
        let fitness_fn = self.fitness_fn;
        let nets = &mut self.nets;
        let data = fitness_data;
        for (net, fitness) in nets.iter_mut() {
            *fitness += fitness_fn(&net.run(&ins), ins, net, data).abs();
        }
    }

    /// sets all of the node outputs to zero (resets the memory of the net)
    fn zero(&mut self) {
        for (_, out) in &mut self.nets {
            *out = 0.0;
        }
    }

    fn repopulate<R: Rng>(&mut self, param: &MutateParam, rng: &mut R) {
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

static input_chars: usize = 1;

fn main() {
    let font = BASIC_UNICODE;
    let possible_chars = ('a'..='z').chain('A'..='Z').collect_vec();

    let mut rng = UsedRng::from_entropy();
    let mut param = RandNetParam::default(8 * 8 * input_chars, 2);
    let mutparam = MutateParam::default();
    param.size = Uniform::new(20, 50);
    let mut brains = Vec::new();
    for _ in 0..100 {
        brains.push(Brain::rand_net(&param, &mut rng))
    }
    let _dist = Uniform::new(-10.0, 10.0);

    let _get_char = |c: char| {
        font[c as usize]
            .byte_array()
            .iter()
            .map(|&line| {
                let mut o = [false; 8];
                for i in 0..8 {
                    o[i] = (line | (1 << i)) != 0;
                }
                o.to_vec()
            })
            .flatten()
            .collect_vec()
    };

    let mut alg = Alg::<Vec<char>>::new(
        brains,
        |net_outs, _given_ins, brain: &Brain, chars| -> f32 {
            net_outs
                .iter()
                .zip(chars)
                .map(|(&out, &exp)| (out - exp as i32 as f32).abs())
                .sum::<f32>()
                + ((brain.net.node_count() + brain.net.edge_count()) as f32 * 0.02)
        },
    );

    let ins_gen = Uniform::from(0..possible_chars.len());

    for _i in 0.. {
        let chars = ins_gen
            .sample_iter(&mut rng)
            .take(input_chars)
            .map(|i| possible_chars[i])
            .collect_vec();
        let ins = chars
            .iter()
            .map(|&c| {
                font[c as usize]
                    .byte_array()
                    .iter()
                    .map(|&byte| {
                        (0..8).map(move |i| {
                            if byte as i32 | (1 << i) != 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        })
                    })
                    .flatten()
                    .collect_vec()
            })
            .flatten()
            .collect_vec();
        for _ in 0..10 {
            alg.run(&ins, &chars);
        }
        alg.repopulate(&mutparam, &mut rng);
        println!("{}", alg.nets[alg.nets.len() / 2].1);
        alg.zero();
        //println!("{}", alg.nets.first().unwrap().1);
        alg.nets.iter_mut().for_each(|(x, _)| {
            x.clear();
        });
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
