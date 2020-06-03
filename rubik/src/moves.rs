extern crate rand;

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fmt;
use std::collections::HashSet;

use tch;
use tch::{nn, kind, Kind, Tensor, no_grad, vision, Device, set_num_threads, get_num_threads, get_num_interop_threads, set_num_interop_threads};
use tch::{nn::Module, nn::OptimizerConfig};
use rand::Rng;
use self::rand::prelude::*;
use std::time::SystemTime;
use tch::nn::{Optimizer, Adam};
//use gtk::{Builder, Application, Window, Button};
use std::sync::Arc;
use std::borrow::Borrow;
use std::path::Path;
use crate::field::{Field, SIZE, scrambled};


// SIZE=3
// epoch: 39500 train loss:  0.03214 err=327 rate=98 sec=381
// epoch: 40000 train loss:  0.02062 err=532 rate=97 sec=388


const MODEL_STORE_PATH: &str = "rubik.ot";

static FEATURE_DIM: i64 = (16/*one-hot*/ * SIZE * SIZE) as i64;
//static HIDDEN_NODES: i64 = 5000;
static HIDDEN_NODES: i64 = 4096;
static HIDDEN_NODES2: i64 = 256;
// static HIDDEN_NODES3: i64 = 128;
// 128 - 64 epoch: 140000 train loss:  0.22475 err=13921 rate=88 sec=442
// 128-128 epoch: 140000 train loss:  0.18137 err=11954 rate=89 sec=491
// 256 - 64 poch: 140000 train loss:  0.17459 err=12117 rate=89 sec=499

// 512x64:
// epoch: 139500 train loss:  0.14538 err=9453 rate=92 sec=857
// epoch: 140000 train loss:  0.07407 err=9910 rate=91 sec=860

// 512x64x64
// epoch: 139500 train loss:  0.11576 err=9502 rate=92 sec=990
// epoch: 140000 train loss:  0.14076 err=9693 rate=91 sec=994

//1024x128x64
// epoch: 139000 train loss:  0.07673 err=5517 rate=95 sec=928
// epoch: 140000 train loss:  0.05649 err=6021 rate=94 sec=935


static LABELS: i64 = (SIZE * SIZE) as i64;

fn net(vs: &nn::Path) -> impl Module {
    // nn::seq()
    //     .add(nn::linear(vs / "layer1", FEATURE_DIM, HIDDEN_NODES, Default::default()))
    //     //.add_fn(|xs| xs.relu()) // 0.03111 err=445 rate=98 sec=1101
    //     .add_fn(|xs| xs.leaky_relu())
    //     .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))

    nn::seq()
        .add(nn::linear(vs / "layer1", FEATURE_DIM, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", HIDDEN_NODES, HIDDEN_NODES2, Default::default()))
        .add_fn(|xs| xs.leaky_relu())
        // .add(nn::linear(vs / "layer3", HIDDEN_NODES2, HIDDEN_NODES3, Default::default()))
        // .add_fn(|xs| xs.leaky_relu())
        .add(nn::linear(vs, HIDDEN_NODES2, LABELS, Default::default()))
}

// #[derive(Debug)]
// struct Net {
//     fc1: nn::Linear,
//     fc2: nn::Linear,
// }
//
// impl Net {
//     fn new(vs: &nn::Path) -> Net {
//         let fc1 = nn::Linear::new(vs, FEATURE_DIM, HIDDEN_NODES, Default::default());
//         let fc2 = nn::Linear::new(vs, HIDDEN_NODES, LABELS, Default::default());
//         Net { fc1, fc2 }
//     }
// }
//
// impl Module for Net {
//     fn forward(&self, xs: &Tensor) -> Tensor {
//         xs.apply(&self.fc1).relu().apply(&self.fc2)
//     }
// }

fn features(f: &Field) -> Tensor {
    //let x_train: Vec<f32> = f.cells.iter().map(|v| *v as f32).collect();
    let x_train: Vec<f32> = f.features();
    //println!("features {:?}", x_train);
    let x = Tensor::of_slice(x_train.as_slice());

    let train_size = 1 as i64;
    let flower_x_train = x.view((train_size, FEATURE_DIM));
    flower_x_train
}

struct MiniBatch {
    x: Tensor,
    y: Tensor,
}

fn prepare_train_data() -> (Vec<MiniBatch>, i64, Vec<usize>) {
    let fieldInit = Field::new();
    let mut exploredVec = vec![fieldInit.clone()];
    let mut best_moves = vec![std::usize::MAX];
    let mut exploredSet = HashSet::new();
    exploredSet.insert(fieldInit.cells);
    for i in 0..500_000 {
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

    let mut xy: Vec<_> = exploredVec.iter().zip(best_moves.iter()).skip(1).collect();
    let seed: [u8; 32] = b"123456789012345678901234567890AA".clone();
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    xy.shuffle(&mut rng);

    //let x_train: Vec<f32> = exploredVec.iter().skip(1).flat_map(|f| f.cells.iter().map(|v| *v as f32)).collect();
    let x_train: Vec<f32> = exploredVec.iter().skip(1).flat_map(|f| f.features()).collect();
    //println!("{:?}", x_train);
    let y_train: Vec<f32> = best_moves.iter().skip(1).map(|m| *m as f32).collect();
    //println!("{:?}", y_train);
    let train_size = (best_moves.len() - 1) as i64;
    let mut batches = Vec::new();
    const BATCH_SIZE: i64 = 128; //512 3x3: 700 sec, 94%  rate=93 sec=387

    // let seed: [u8; 32] = b"123456789012345678901234567890AA".clone();
    // let mut rng: StdRng = SeedableRng::from_seed(seed);
    // let mut x_train_batch = x_train.clone();
    // x_train_batch.shuffle(&rng);
    // let mut y_train_batch = y_train.clone();
    // y_train_batch.shuffle(&rng);
    let mut x_train_batch: Vec<f32> = Vec::new();
    let mut y_train_batch: Vec<f32> = Vec::new();
    for xyi in xy {
        //let mut xi: Vec<f32> = (*xyi.0).cells.iter().map(|v| *v as f32).collect();
        let mut xi: Vec<f32> = (*xyi.0).features();
        x_train_batch.append(&mut xi);
        y_train_batch.push(*xyi.1 as f32);
    }
    for i in 0..train_size / BATCH_SIZE {
        let beg = (i * BATCH_SIZE) as usize;
        let end = beg + BATCH_SIZE as usize;
        let begx = (beg * FEATURE_DIM as usize) as usize;
        let endx = ((beg + BATCH_SIZE as usize) * FEATURE_DIM as usize) as usize;
        let x = Tensor::of_slice(&x_train_batch[begx..endx]);
        let y = Tensor::of_slice(&y_train_batch[beg..end]).to_kind(Kind::Int64);

        let flower_x_train = x.view((BATCH_SIZE, FEATURE_DIM));
        let flower_y_train = y.view(BATCH_SIZE);
        batches.push(MiniBatch { x: flower_x_train, y: flower_y_train });
    }
    let x = Tensor::of_slice(x_train.as_slice());
    let y = Tensor::of_slice(y_train.as_slice()).to_kind(Kind::Int64);

    let flower_x_train = x.view((train_size, FEATURE_DIM));
    let flower_y_train = y.view(train_size);
    println!("train size {} threads={},interop={} CUDA_avail={}", train_size, get_num_threads(), get_num_interop_threads(), tch::Cuda::is_available());
    println!("train size {} batches={}", train_size, batches.len());

    (batches, train_size, best_moves)
}

fn train(mut opt: Optimizer<Adam>, net: &impl Module) {
    let (batches, train_size, best_moves) = prepare_train_data();

    let now = SystemTime::now();
    for epoch in 1..=80000 {
        let batch = &batches[epoch % batches.len()];
        let loss = net
            .forward(&batch.x)
            .cross_entropy_for_logits(&batch.y);
        opt.backward_step(&loss);
        // let loss = net
        //     .forward(&flower_x_train)
        //     .cross_entropy_for_logits(&flower_y_train);
        // opt.backward_step(&loss);
        // let test_accuracy = net
        //     .forward(&flower_x_test)
        //     .accuracy_for_logits(&flower_y_test);
        if epoch % 1000 == 0 {
            //let pred_now = net.forward(&flower_x_train).argmax(-1, false);
            let mut ok = 0;
            // for i in 0..(train_size as usize) {
            //     let best = best_moves[i + 1];
            //     let pred = pred_now.double_value(&[i as i64]) as usize;
            //     if best == pred { ok = ok + 1; }
            //     //println!("i={} best={} pred={}", i, best, pred);
            // }
            println!(
                //"epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                "epoch: {:4} train loss: {:8.5} err={} rate={:5.2} sec={}",
                epoch,
                f64::from(&loss),
                train_size - ok,//100. * f64::from(&test_accuracy),
                100f32 * ok as f32 / train_size as f32,
                now.elapsed().unwrap().as_secs(),
            );
        }
    };
}

pub fn run() -> Result<(), Box<dyn Error>> {
    //set_num_threads()
    //tch::Cpu::set_num_threads(4);
    set_num_interop_threads(8);
    // working on a linear neural network with SGD
    let mut vs = nn::VarStore::new(Device::Cpu);
    //let net = Net::new(&vs.root());
    let net = net(&vs.root());
//    println!("scrambled \n{}", scrambled());

    if Path::new(MODEL_STORE_PATH).exists() {
        println!("pre-trained model found in file {}", MODEL_STORE_PATH);
        vs.load(MODEL_STORE_PATH);
    } else {
        println!("training, then saving as {}", MODEL_STORE_PATH);
        let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
        train(opt, &net);
        vs.save(MODEL_STORE_PATH);
    }

    //net.forward(&flower_x_train).print();
    //net.forward(&flower_x_train).argmax1(-1, false).print();
    solve(&net, scrambled(), 500);
    Ok(())
}

fn solve(net: &impl Module, mut scr: Field, max_steps: i32) -> Option<i32> {
    //let mut rng = rand::thread_rng();
    let seed: [u8; 32] = b"123456789012345678901234567890A-".clone();
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let mut exploredSet = HashSet::new();
    exploredSet.insert(scr.cells);

    for mut i in 0..max_steps {
        let fwd: Tensor = net.forward(&features(&scr));
        //fwd.print();
        let fwdMax = fwd.argmax(-1, false);
        //fwdMax.print();
        let pred = fwdMax.double_value(&[0]);
        println!("{} num={} - scrambled; moves={:?} pred={}", scr, i, scr.moves(), pred);
        let old_empty = scr.empty;
        let mut moved = scr.mov_if_not_in(pred as usize, &exploredSet);
        if scr.is_done() {
            println!("DONE, SOLVED!!!! i={}", i);
            return Some(i + 1);
            //break;
        }

        let mut moves = scr.moves();
        let pred_usize = pred as usize;
        moves.remove_item(&pred_usize);
        while !moved && moves.len() > 0 {
            // if(moves.len()==0){
            //     moves = scr.moves();
            // }
            let mov = moves[rng.gen::<usize>() % moves.len()];
            println!("LOOP doing move {}", mov);
            moved = scr.mov_if_not_in(mov, &exploredSet);
            moves.remove_item(&mov);
        }
        if (!moved) {
            let mut moves = scr.moves();
            let mov = moves[rng.gen::<usize>() % moves.len()];
            println!("MOVE TO REPEATED STATE!!!!! {}", mov);
            scr.mov(mov);
        }
        exploredSet.insert(scr.cells);
    }
    None
}
