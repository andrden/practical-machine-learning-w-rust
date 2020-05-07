extern crate rustlearn;
extern crate bincode;

use std::fs::File;
use std::io::prelude::*;

// use std::fs::File;
use std::io::BufReader;
// use std::io::prelude::*;

use rustlearn::prelude::*;
use rustlearn::linear_models::sgdclassifier::Hyperparameters;
use rustlearn::cross_validation::CrossValidation;
use rustlearn::datasets::iris;
use rustlearn::metrics::accuracy_score;
use bincode::{serialize, deserialize};
use rustlearn::trees::decision_tree;

struct Row {
    src: Vec<f32>,
    res: f32,
}

fn parseRow(s: String) -> Row {
    let v: Vec<&str> = s.split_whitespace().collect();
    let b: Row = Row::new(v);
    b
}

impl Row {
    pub fn new(v: Vec<&str>) -> Row {
        let f64_formatted: Vec<f32> = v.iter().map(|s| s.parse().unwrap()).collect();
        let src = f64_formatted[0..f64_formatted.len() - 1].to_vec();
        let res = f64_formatted[f64_formatted.len() - 1];
        //src.iter().map(|n| print!("{}",n)).;
        for n in &src {
            print!("{}", n)
        }
        println!(" len={} res {}", src.len(), res);
        Row { src: src, res }
    }

    pub fn into_feature_vector(&self) -> Vec<f32> {
        self.src.to_vec()
    }

    pub fn into_targets(&self) -> f32 {
        self.res
    }
}

pub fn myData1() -> (Array, Array) {
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

    let x_train: Vec<Vec<f32>> = rows.iter().map(|r| r.into_feature_vector()).collect();
    let y_train: Vec<f32> = rows.iter().map(|r| r.into_targets()).collect();
    (Array::from(&x_train), Array::from(y_train))

    // let train_size = rows.len();
    // let mx_train = Matrix::new(train_size, 3, x_train);
    // let my_train = Vector::new(y_train);
    // let mut lin_model = LinRegressor::default();
    // lin_model.train(&mx_train, &my_train);
    // println!("{:?}", lin_model);
    //
    // // Now we will predict
    // let predictions = lin_model.predict(&mx_train).unwrap();
    // for n in &predictions {
    //     //if n.round()>=0.5f64 {
    //     println!("prediction {}  {}", n.round(), n)
    //     // }else{
    //     //
    //     // }
    // }
    //let predictions = Matrix::new(test_size, 1, predictions);
}


fn predict1() -> std::io::Result<()> {
    let (x_train, y_train) = myData1();
    println!("x_train={:?}", x_train);
    println!("y_train={:?}", y_train);
    let mut model = Hyperparameters::new(x_train.cols())
        .learning_rate(0.01)
        .l2_penalty(0.1)
        .l1_penalty(0.0)
        //.build();
        .one_vs_rest();
    for _ in 0..500 {
        model.fit(&x_train, &y_train).unwrap();
    }
    let prediction = model.predict(&x_train).unwrap();
    println!("prediction={:?}", prediction);


    let mut decision_tree_model = decision_tree::Hyperparameters::new(x_train.cols())
        .one_vs_rest();
    decision_tree_model.fit(&x_train, &y_train).unwrap();

    let predictionTree = decision_tree_model.predict(&x_train).unwrap();
    println!("predictionTree={:?}", predictionTree);

    return Ok(());
}

fn main() -> std::io::Result<()> {
    predict1();

    let (X, y) = iris::load_data();
    let num_splits = 10;
    let num_epochs = 5;
    let mut accuracy = 0.0;
    let mut model = Hyperparameters::new(X.cols())
        .learning_rate(0.5)
        .l2_penalty(0.0)
        .l1_penalty(0.0)
        .one_vs_rest();

    for (train_idx, test_idx) in CrossValidation::new(X.rows(), num_splits) {
        println!("training {:?} {:?}", train_idx, test_idx);
        let X_train = X.get_rows(&train_idx);
        let y_train = y.get_rows(&train_idx);
        let X_test = X.get_rows(&test_idx);
        let y_test = y.get_rows(&test_idx);

        for _ in 0..num_epochs {
            model.fit(&X_train, &y_train).unwrap();
        }
        let prediction = model.predict(&X_test).unwrap();
        let present_acc = accuracy_score(&y_test, &prediction);
        accuracy += present_acc;
    }
    println!("accuracy: {:#?}", accuracy / num_splits as f32);

    // serialise the library
    //let encoded = serialize(&model).unwrap();
    //println!("{:?}", model.models());
    //let mut file = File::create("foo.txt")?;
    //file.write_all(encoded)?;
    Ok(())
}
