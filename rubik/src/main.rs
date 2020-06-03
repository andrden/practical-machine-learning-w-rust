#![feature(vec_remove_item)]

use std::vec::Vec;
use std::process::exit;
use std::env::args;

mod moves;
mod field;

fn main() {
    //tch_ex::run().expect("err example");
    moves::run();
}