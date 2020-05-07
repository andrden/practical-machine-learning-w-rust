use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

use rusty_machine;
use rusty_machine::linalg::Matrix;
// use rusty_machine::linalg::BaseMatrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;

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


pub fn myPredict2() {
    let file = File::open("/home/andrii/development/rust/practical-machine-learning-w-rust/chapter2/predict2.txt").expect("no such file");
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

    let train_size = rows.len();
    let mx_train = Matrix::new(train_size, 5, x_train);
    let my_train = Vector::new(y_train);
    let mut lin_model = LinRegressor::default();
    lin_model.train(&mx_train, &my_train);
    println!("{:?}", lin_model);

    // Now we will predict
    let predictions = lin_model.predict(&mx_train).unwrap();
    for n in &predictions {
        //if n.round()>=0.5f64 {
        println!("prediction {}  {}", n.round(), n)
        // }else{
        //
        // }
    }
    //let predictions = Matrix::new(test_size, 1, predictions);
}
