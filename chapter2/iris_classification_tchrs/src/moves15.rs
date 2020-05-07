use std::error::Error;
use std::fmt::Display;
use serde::export::Formatter;
use std::fmt;
use std::collections::HashSet;

use tch;
use tch::{nn, kind, Kind, Tensor, no_grad, vision, Device};
use tch::{nn::Module, nn::OptimizerConfig};

const SIZE: usize = 3;
const CHARS: &str = "0123456789ABCDEFGHIJK";

static FEATURE_DIM: i64 = (SIZE * SIZE) as i64;
static HIDDEN_NODES: i64 = 500;
static LABELS: i64 = (SIZE * SIZE) as i64;

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

#[derive(Clone)]
struct Field {
    empty: usize,
    cells: [u8; SIZE * SIZE],
}

impl Field {
    fn new() -> Field {
        let mut f = Field {
            empty: SIZE * SIZE - 1,
            cells: [0u8; SIZE * SIZE],
        };
        for i in 0..SIZE * SIZE - 1 {
            f.cells[i] = (i + 1) as u8;
        }
        f
    }
    fn is_done(&self) -> bool {
        self.cells == Field::new().cells
    }
    fn mov(&mut self, pos: usize) {
        let val = self.cells[pos];
        assert_eq!(0, self.cells[self.empty]);
        assert!(self.moves().contains(&pos));
        self.cells[pos] = 0;
        self.cells[self.empty] = val;
        self.empty = pos;
    }
    fn rowCol(pos: usize) -> (usize, usize) {
        (pos / SIZE, pos % SIZE)
    }
    fn moves(&self) -> Vec<usize> {
        let mut res = Vec::new();
        let (row, col) = Field::rowCol(self.empty);
        if col > 0 {
            res.push(self.empty - 1);
        }
        if col < SIZE - 1 {
            res.push(self.empty + 1);
        }
        if row > 0 {
            res.push(self.empty - SIZE);
        }
        if row < SIZE - 1 {
            res.push(self.empty + SIZE);
        }
        res
    }
}

impl Display for Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for i in 0..SIZE {
            write!(f, "{}", "|");
            for j in 0..SIZE {
                let val = self.cells[i * SIZE + j] as usize;
                if val == 0 {
                    f.write_str(" ");
                } else {
                    f.write_str(&CHARS[val..val + 1]);
                }
            }
            write!(f, "{}", "|\n");
        }
        Ok(())
    }
}

fn scrambled() -> Field {
    let mut f = Field::new();
    for i in 0..40 {
        let moves = f.moves();
        f.mov(moves[i % moves.len()]);
    }
    f
}

fn features(f: &Field) -> Tensor {
    let x_train: Vec<f64> = f.cells.iter().map(|v| *v as f64).collect();
    //println!("features {:?}", x_train);
    let x = Tensor::float_vec(x_train.as_slice());

    let train_size = 1 as i64;
    let flower_x_train = x.view(&[train_size, FEATURE_DIM]);
    flower_x_train
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let fieldInit = Field::new();
    let mut exploredVec = vec![fieldInit.clone()];
    let mut best_moves = vec![std::usize::MAX];
    let mut exploredSet = HashSet::new();
    exploredSet.insert(fieldInit.cells);
    for i in 0..5000 {
        if i >= exploredVec.len() {
            break;
        }
        let f = &exploredVec[i];
        let mut addVec = Vec::new();
        for m in f.moves() {
            let old_empty = f.empty;
            let mut moved = f.clone();
            moved.mov(m);
            if !exploredSet.contains(&moved.cells) {
                exploredSet.insert(moved.cells);
                addVec.push(moved);
                best_moves.push(old_empty);
            }
        }
        exploredVec.append(&mut addVec);
    }
    println!("vec len {}", exploredVec.len());
    //for i in 0..exploredVec.len() {
    for i in exploredVec.len() - 10..exploredVec.len() {
        let f = &exploredVec[i];
        println!("{} best move {}", f, best_moves[i]);
    }

    let x_train: Vec<f64> = exploredVec.iter().skip(1).flat_map(|f| f.cells.iter().map(|v| *v as f64)).collect();
    //println!("{:?}", x_train);
    let y_train: Vec<f64> = best_moves.iter().skip(1).map(|m| *m as f64).collect();
    //println!("{:?}", y_train);
    let x = Tensor::float_vec(x_train.as_slice());
    let y = Tensor::float_vec(y_train.as_slice()).to_kind(Kind::Int64);

    let train_size = (best_moves.len() - 1) as i64;
    let flower_x_train = x.view(&[train_size, FEATURE_DIM]);
    let flower_y_train = y.view(&[train_size]);

    // working on a linear neural network with SGD
    let vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());
    let opt = nn::Adam::default().build(&vs, 1e-3)?;
    //tch::Cpu::set_num_threads(4);
    //println!("train size {} threads={} CUDA_avail={}", train_size, tch::Cpu::get_num_threads(), tch::Cuda::is_available());
    println!("train size {} ", train_size);
    for epoch in 1..14500 {
        let loss = net
            .forward(&flower_x_train)
            .cross_entropy_for_logits(&flower_y_train);
        opt.backward_step(&loss);
        // let test_accuracy = net
        //     .forward(&flower_x_test)
        //     .accuracy_for_logits(&flower_y_test);
        if epoch % 400 == 0 {
            let pred_now = net.forward(&flower_x_train).argmax1(-1, false);
            let mut ok = 0;
            for i in 0..(train_size as usize) {
                let best = best_moves[i + 1];
                let pred = pred_now.double_value(&[i as i32]) as usize;
                if best == pred { ok = ok + 1; }
                //println!("i={} best={} pred={}", i, best, pred);
            }
            println!(
                //"epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                "epoch: {:4} train loss: {:8.5} err={} rate={}",
                epoch,
                f64::from(&loss),
                train_size - ok,//100. * f64::from(&test_accuracy),
                100 * ok / train_size
            );
        }
    };
    //net.forward(&flower_x_train).print();
    //net.forward(&flower_x_train).argmax1(-1, false).print();

    let mut scr = scrambled();
    for i in 0..15 {
        let fwd: Tensor = net.forward(&features(&scr));
        //fwd.print();
        let fwdMax = fwd.argmax1(-1, false);
        //fwdMax.print();
        let pred = fwdMax.double_value(&[0]);
        println!("{} num={} - scrambled; moves={:?} pred={}", scr, i, scr.moves(), pred);
        scr.mov(pred as usize);
        if scr.is_done() {
            println!("DONE!!!! i={}", i);
            break;
        }
    }
    // let mut f = Field::new();
    // println!("field \n{} {:?}", f, f.moves());
    // f.mov(1);
    //
    // println!("field \n{}", f);
    // f.mov(0);
    // println!("field \n{}", f);
    Ok(())
}