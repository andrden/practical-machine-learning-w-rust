/// Data gathered from https://www.kaggle.com/vikrishnan/boston-house-prices
/// Boston dataset: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
/// This module shows how to run regression models
extern crate serde;
// This lets us write `#[derive(Deserialize)]`.
#[macro_use]
extern crate serde_derive;

use std::vec::Vec;
use std::process::exit;
use std::env::args;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

use rusty_machine;
use rusty_machine::linalg::Matrix;
// use rusty_machine::linalg::BaseMatrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;
use crate::predict1::myPredict1;
use crate::predict2::myPredict2;

mod lin_reg;
mod gaussian_process_reg;
mod glms;
mod predict1;
mod predict2;

fn main() {
    myPredict1();
    println!("---------------------------");
    myPredict2();
    let args: Vec<String> = args().collect();
    let model = if args.len() < 2 {
        None
    } else {
        Some(args[1].as_str())
    };
    let res = match model {
        None => {
            println!("nothing", );
            Ok(())
        }
        Some("lr") => lin_reg::run(),
        Some("gp") => gaussian_process_reg::run(),
        Some("glms") => glms::run(),
        Some(_) => lin_reg::run(),
    };
    // Putting the main code in another function serves two purposes:
    // 1. We can use the `?` operator.
    // 2. We can call exit safely, which does not run any destructors.
    exit(match res {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}