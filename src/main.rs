use anyhow::*;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use petgraph::{
    graph::NodeIndex,
    stable_graph::StableGraph,
    EdgeDirection::{Incoming, Outgoing},
};
use rand::{
    prelude::{IteratorRandom, *},
    rngs, thread_rng, Rng,
};
use std::{borrow::Borrow, cmp::min, ops::RangeBounds, vec::Vec};

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
}

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
    activation_fns: Vec<(fn(f32) -> f32, f32)>,
    min_conn_count: usize,
    max_conn_count: usize,
    min_conn_stregnth: f32,
    max_conn_stregnth: f32,
    inputs: usize,
    outputs: usize,
    min_size: usize,
    max_size: usize,
}

impl RandNetParam {
    fn default(inputs: usize, outputs: usize) -> RandNetParam {
        RandNetParam {
            activation_fns: vec![(|x| x / (1f32 + x.abs()), 1f32)],
            min_conn_count: 2,
            max_conn_count: 10,
            min_conn_stregnth: -10f32,
            max_conn_stregnth: 10f32,
            inputs,
            outputs,
            min_size: 50,
            max_size: 200,
        }
    }
}

impl Brain {
    fn run(&mut self, ins: Vec<f32>) -> Vec<f32> {
        assert!(
            ins.len() == self.inputs as usize,
            "Wrong number of network inputs"
        );
        let net = &mut self.net;
        let input_nodes = net.externals(Incoming).collect_vec();

        for (inode, input) in input_nodes.iter().zip(ins) {
            net[*inode].out = input;
        }

        self.hidden_nodes
            .iter()
            .chain(self.output_nodes.iter())
            .map(|&node| -> (NodeIndex, f32) {
                let mut neighbors = net.neighbors_directed(node, Incoming).detach();

                let mut node_input = 0f32;
                while let Some((edge, connected_node)) = neighbors.next(net) {
                    node_input += net[connected_node].out * net[edge];
                }

                (node, (net[node].activation_fn)(node_input))
            })
            .collect_vec()
            .iter()
            .for_each(|(node, buffered_out)| net[*node].out = *buffered_out);

        self.output_nodes
            .iter()
            .map(|&node| net[node].out)
            .collect()
    }

    fn rand_net(param: &RandNetParam) -> Brain {
        let mut rng = thread_rng();
        let size: usize = rng.gen_range(param.min_size..=param.max_size);
        let inputs = param.inputs;
        let outputs = param.outputs;
        let act_fns = &param.activation_fns;

        let mut net: StableGraph<Node, f32> =
            StableGraph::with_capacity(size, (param.max_conn_count + param.min_conn_count) / 2);
        let mut input_nodes: Vec<NodeIndex> = Vec::new();
        let mut hidden_nodes: Vec<NodeIndex> = Vec::new();
        let mut output_nodes: Vec<NodeIndex> = Vec::new();

        // add input nodes
        for _ in 0..param.inputs {
            input_nodes.push(net.add_node(Node::new(|x| x)));
        }

        // generate nodes
        for _ in 0..size {
            let prob = rng.gen_range(0f32..=act_fns.iter().map(|&(_, x)| x).sum());
            let mut acc = 0f32;
            let &(act_fn, _) = act_fns
                .iter()
                .skip_while(|&&(_, x)| {
                    acc += x;
                    acc < prob
                })
                .next()
                .unwrap();
            hidden_nodes.push(net.add_node(Node::new(act_fn)));
        }

        // generate output nodes
        for _ in 0..outputs {
            output_nodes.push(net.add_node(Node::new(|x| x)));
        }

        // generate connections
        for &node in hidden_nodes.iter().chain(output_nodes.iter()) {
            let node_size = rng.gen_range(param.min_conn_count..=param.max_conn_count);
            input_nodes
                .iter()
                .chain(hidden_nodes.iter())
                .choose_multiple(&mut rng, node_size)
                .iter()
                .for_each(|&&other_node| {
                    let weight = rng.gen_range(param.min_conn_stregnth..=param.max_conn_stregnth);
                    net.add_edge(other_node, node, weight);
                })
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

fn main() {
    let mut brain = Brain::rand_net(&RandNetParam::default(5, 3));
    for _ in 0..20 {
        println!(
            "{:?}",
            brain.run(vec![2.1f32, 2.4f32, 3.5f32, 2.4f32, 1.5f32])
        );
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
