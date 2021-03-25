use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph, EdgeDirection::Incoming};
use rand::{distributions::*, prelude::IteratorRandom, Rng};
use rayon::prelude::*;

use crate::populate_node;

#[derive(Clone)]
pub struct Node {
    out: f32,
    activation_fn: fn(f32) -> f32,
}

impl Node {
    pub fn new(activation_fn: fn(f32) -> f32) -> Node {
        Node {
            out: 0f32,
            activation_fn,
        }
    }

    pub fn random_fn<R: Rng>(
        rng: &mut R,
        act_fns: &Vec<fn(f32) -> f32>,
        act_weights: &WeightedIndex<f32>,
    ) -> Node {
        Node::new(act_fns[act_weights.sample(rng)])
    }
}

#[derive(Clone)]
pub struct Brain {
    pub net: StableGraph<Node, f32>,
    pub input_nodes: Vec<NodeIndex>,
    pub hidden_nodes: Vec<NodeIndex>,
    pub output_nodes: Vec<NodeIndex>,
    pub inputs: usize,
    pub outputs: usize,
}

pub struct RandNetParam {
    /// The first value is the activation function, the second value is the relative probability of that function occurring in a node
    pub act_fns: Vec<(fn(f32) -> f32, f32)>,
    pub conn_count: Uniform<usize>,
    pub conn_stregnth: Uniform<f32>,
    pub size: Uniform<usize>,
    pub inputs: usize,
    pub outputs: usize,
}

pub struct MutateParam {
    pub act_fns: Vec<(fn(f32) -> f32, f32)>,
    pub node_insertion_prop: Uniform<f32>,
    pub conn_insertion_prop: Uniform<f32>,
    pub node_removal_prop: Uniform<f32>,
    pub conn_removal_prob: Uniform<f32>,
    pub new_conn_stregnth: Uniform<f32>,
    pub new_node_size: Uniform<usize>,
}

impl MutateParam {
    pub fn default() -> MutateParam {
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
    pub fn default(inputs: usize, outputs: usize) -> RandNetParam {
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
    pub fn run(&mut self, ins: &[f32]) -> Vec<f32> {
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

    pub fn clear(&mut self) {
        let net = &mut self.net;
        net.node_indices().collect_vec().into_iter().for_each(|i| {
            net[i].out = 0.0;
        });
    }

    pub fn mutate<R: Rng>(mut self, param: &MutateParam, rng: &mut R) -> Self {
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

    pub fn rand_net<R: Rng>(param: &RandNetParam, rng: &mut R) -> Brain {
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
