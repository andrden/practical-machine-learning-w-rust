use std::io;
use std::vec::Vec;
use std::error::Error;
use std::io::BufReader;
use std::fs::File;
use std::io::prelude::*;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use tch;
use tch::{nn, kind, Kind, Tensor, no_grad, vision, Device};
use tch::{nn::Module, nn::OptimizerConfig};

use ml_utils;
use ml_utils::datasets::Flower;

static FEATURE_DIM: i64 = 3;
static HIDDEN_NODES: i64 = 10;
static LABELS: i64 = 2;

#[derive(Debug)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let fc1 = nn::Linear::new(vs, FEATURE_DIM, HIDDEN_NODES, Default::default());
        let fc2 = nn::Linear::new(vs, HIDDEN_NODES, LABELS, Default::default());
        Net { fc1, fc2 }
    }
}

impl Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).relu().apply(&self.fc2)
    }
}

struct Row {
    src: Vec<f64>,
    res: f64,
}

fn parseRow(s: String) -> Row {
    let v: Vec<&str> = s.split_whitespace().collect();
    let b: Row = Row::new(v);
    b
}

impl Row {
    pub fn new(v: Vec<&str>) -> Row {
        let f64_formatted: Vec<f64> = v.iter().map(|s| s.parse().unwrap()).collect();
        let src = f64_formatted[0..f64_formatted.len() - 1].to_vec();
        let res = f64_formatted[f64_formatted.len() - 1];
        //src.iter().map(|n| print!("{}",n)).;
        for n in &src {
            print!("{}", n)
        }
        println!(" len={} res {}", src.len(), res);
        Row { src: src, res }
    }

    pub fn into_feature_vector(&self) -> Vec<f64> {
        self.src.to_vec()
    }

    pub fn into_targets(&self) -> f64 {
        self.res
    }
}

pub fn myData1() -> (Tensor, Tensor, usize) {
    let file = File::open("/home/andrii/development/rust/practical-machine-learning-w-rust/chapter2/predict1.txt").expect("no such file");
    let buf = BufReader::new(file);
    let rows: Vec<Row> = buf.lines().enumerate()
        .map(|(n, l)| {
            let exp = l.expect("err");
            //println!("{}: {}", n, exp);
            exp
        })
        .filter(|r| !r.starts_with('#'))
        //.map(|(n, l)| l.expect(&format!("Could not parse line no {}", n)))
        .map(|r| parseRow(r))
        .collect();
    println!("rows={}", rows.len());

    let x_train: Vec<f64> = rows.iter().flat_map(|r| r.into_feature_vector()).collect();
    let y_train: Vec<f64> = rows.iter().map(|r| r.into_targets()).collect();

    let x = Tensor::float_vec(x_train.as_slice());
    let y = Tensor::float_vec(y_train.as_slice()).to_kind(Kind::Int64);
    (x, y,y_train.len())
}


pub fn run() -> Result<(), Box<Error>> {
    let (x, y,train_size) = myData1();

    // print shape of all the data.
    println!("Training data shape {:?} {:?}", x.size(), x);
    println!("Training flower_y_train data shape {:?} {:?}", y.size(), y);
    y.print();

    // reshaping examples
    // one way to reshape is using unsqueeze
    //let flower_x_train1 = flower_x_train.unsqueeze(0); // Training data shape [1, 360]
    //println!("Training data shape {:?}", flower_x_train1.size());
    let train_size = train_size as i64;
    let flower_x_train = x.view(&[train_size, FEATURE_DIM]);
    let flower_y_train = y.view(&[train_size]);

    // working on a linear neural network with SGD
    let vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());
    let opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..500 {
        let loss = net
            .forward(&flower_x_train)
            .cross_entropy_for_logits(&flower_y_train);
        opt.backward_step(&loss);
        // let test_accuracy = net
        //     .forward(&flower_x_test)
        //     .accuracy_for_logits(&flower_y_test);
        if epoch % 10 == 0 {
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                -1//100. * f64::from(&test_accuracy),
            );
        }
    };
    net.forward(&flower_x_train).print();
    net.forward(&flower_x_train).argmax1(-1, false).print();

    Ok(())
}
