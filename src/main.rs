use anyhow::*;
use itertools::Itertools;
use petgraph::{stable_graph::StableGraph, EdgeDirection::Incoming};
use std::vec::Vec;

struct Node {
    out: f32,
    activation_fn: fn(f32) -> f32,
}

struct Brain {
    nodes: StableGraph<Node, f32>,
    inputs: i32,
    outputs: i32,
}

impl Brain {
    fn run(&mut self, ins: Vec<f32>) -> Vec<f32> {
        assert!(
            ins.len() == self.inputs as usize,
            "Wrong number of network inputs"
        );
        let mut outs: Vec<f32> = Vec::new();

        for (inode, input) in self.nodes.externals(Incoming).collect_vec().iter().zip(ins) {
            self.nodes[*inode].out = input;
        }

        outs
    }
}

fn main() {

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
