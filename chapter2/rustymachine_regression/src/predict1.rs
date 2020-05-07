use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

use rusty_machine;
use rusty_machine::linalg::Matrix;
// use rusty_machine::linalg::BaseMatrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;
use std::io;
use std::process;
use std::fmt::Display;
//use std::thread::
//use std::ops::
//use std::cell::
use rusty_machine::analysis::score::neg_mean_squared_error;

struct Row {
    src: Vec<f64>,
    res: f64,
}

struct AA;

impl Drop for AA{
    fn drop(&mut self) {
        println!("AA droppped");
    }
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

// fn maxx<'a, 'b>(s1: &'a str, s2: &'b str) -> &'a str {
//     //s2[0..2]
//     if (s1 == s2) {
//         s1
//     } else {
//         s2 // sdf
//     }
// }

pub fn myPredict1() {
    rust_features();


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

    let train_size = rows.len();
    let mx_train = Matrix::new(train_size, 3, x_train);
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

fn rust_features() {
    let rs: Result<u32, String> = Err("sdf".to_string());
// let config = rs.unwrap_or_else(|err| {
//     println!("Problem parsing arguments: {}", err);
//     process::exit(1);
// });
    "sdf".to_uppercase();
    let v = vec![1, 2, 3, 4];
    let vi = v.iter();
    vi.skip(1);
    let b = Box::new(5);
    assert_eq!(5, 5);
//std::mem::drop(b);
    let x = 1;
//let x:Option<()> = Some(());
    println!("sizeof = {}", std::mem::size_of_val(&x));
    let xx = std::cell::RefCell::new(vec![1]);
    xx.borrow_mut();
    "".lines().map(|s| s);
    std::thread::spawn(|| 10).join().unwrap();
//std::sync::
    let a1 = 10..20;
    let a2 = 10..=20;
    let xx: *mut char;
//=1 as *mut char;
    {
        let mut x = 'a';
        xx = &mut x;
    }
    println!("ref {}", xx as usize);
    unsafe {
        //*xx = 'd';
    }
    let mut buffer2 = [256; 512];
    let bbbb = 0u8;
    //String::from_utf8_lossy(&buffer[..]);
    //let mut v = vec![1, 2, "3", 4, 5, 6];
    //std::slice::
    unsafe {
        //println!("Absolute value of -3 according to C: {} t={}", abs(-3), ktime_get_ns());
    }
    let get = b"G  ET / HTTP/1.1\r\n";

    {let aa1 = AA{};}
    println!("after first AA");
    let aa2 = AA{};
}
// <T:Display> fun  sdf()-> (){
//
// }
extern "C" {
    fn abs(input: i32) -> i32;
    //fn ktime_get_ns() -> u64;
}