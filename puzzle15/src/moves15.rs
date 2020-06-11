extern crate gio;
extern crate gtk;
extern crate rand;

//use std::borrow::Borrow;
use std::collections::HashSet;
use std::error::Error;
//use std::fmt::{Display, Formatter};
//use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use gio::prelude::*;
//use gtk::{Builder, Application, Window, Button};
//use gtk::{Application, ApplicationWindow, Button};
use gtk::prelude::*;
use rand::Rng;
use tch;
use tch::{Device, get_num_interop_threads, get_num_threads, Kind, nn, set_num_interop_threads, Tensor};
use tch::{nn::Module, nn::OptimizerConfig};
use tch::nn::{Adam, Optimizer};

use crate::field::{Field, SIZE, examples};

use self::gtk::{Grid, GridBuilder, Button};
use self::rand::prelude::*;

// SIZE=3
// epoch: 39500 train loss:  0.03214 err=327 rate=98 sec=381
// epoch: 40000 train loss:  0.02062 err=532 rate=97 sec=388


const MODEL_STORE_PATH: &str = "puzzle15.ot";

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
    let flower_x_train = x.view((train_size, FEATURE_DIM)).to_device(Device::cuda_if_available());
    flower_x_train
}

struct MiniBatch {
    x: Tensor,
    y: Tensor,
}

fn prepare_train_data(steps: usize) -> (Vec<MiniBatch>, i64, Vec<usize>) {
    let field_init = Field::new();
    if !field_init.is_solvable() {
        panic!("not solvable1");
    }
    let mut explored_vec = vec![field_init.clone()];
    let mut best_moves = vec![std::usize::MAX];
    let mut explored_set = HashSet::new();
    explored_set.insert(field_init.cells);
    for i in 0..steps {
        if i >= explored_vec.len() {
            break;
        }
        let f = &explored_vec[i];
        let mut add_vec = Vec::new();
        for m in f.moves() {
            let old_empty = f.empty;
            let mut moved = f.clone();
            moved.mov(m);
            if !explored_set.contains(&moved.cells) {
                if !moved.is_solvable() {
                    panic!("not solvable");
                }
                explored_set.insert(moved.cells);
                add_vec.push(moved);
                best_moves.push(old_empty);
            }
        }
        explored_vec.append(&mut add_vec);
        if i % 100_000 == 0 {
            println!("steps {} explored_vec {}", i, explored_vec.len());
        }
    }
    println!("vec len {}", explored_vec.len());
    //for i in 0..exploredVec.len() {
    for i in explored_vec.len() - 10..explored_vec.len() {
        let f = &explored_vec[i];
        println!("{} best move {}", f, best_moves[i]);
    }

    let mut xy: Vec<_> = explored_vec.iter().zip(best_moves.iter()).skip(1).collect();
    let seed: [u8; 32] = b"123456789012345678901234567890AA".clone();
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    xy.shuffle(&mut rng);
    //let xy = &xy[0..take];

    //let x_train: Vec<f32> = exploredVec.iter().skip(1).flat_map(|f| f.cells.iter().map(|v| *v as f32)).collect();
    //let x_train: Vec<f32> = exploredVec.iter().skip(1).flat_map(|f| f.features()).collect();
    //println!("{:?}", x_train);
    //let y_train: Vec<f32> = best_moves.iter().skip(1).map(|m| *m as f32).collect();
    //println!("{:?}", y_train);
    let train_size = xy.len() as i64; // (best_moves.len() - 1) as i64;
    let mut batches = Vec::new();
    const BATCH_SIZE: i64 = 1024; //512 3x3: 700 sec, 94%  rate=93 sec=387

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

        let flower_x_train = x.view((BATCH_SIZE, FEATURE_DIM));//.to_device(Device::cuda_if_available());
        let flower_y_train = y.view(BATCH_SIZE);//.to_device(Device::cuda_if_available());
        batches.push(MiniBatch { x: flower_x_train, y: flower_y_train });
    }
    // let x = Tensor::of_slice(x_train.as_slice());
    // let y = Tensor::of_slice(y_train.as_slice()).to_kind(Kind::Int64);

    // let flower_x_train = x.view((train_size, FEATURE_DIM));
    // let flower_y_train = y.view(train_size);
    println!("train size {} threads={},interop={} CUDA_avail={}", train_size, get_num_threads(), get_num_interop_threads(), tch::Cuda::is_available());
    println!("train size {} batches={}", train_size, batches.len());

    (batches, train_size, best_moves)
}

fn train(mut opt: Optimizer<Adam>, net: &impl Module) {
    let (batches, train_size, _best_moves) = prepare_train_data(10_900_000);

    let now = SystemTime::now();
    for epoch in 1..=80000 {
        let batch = &batches[epoch % batches.len()];
        let loss = net
            .forward(&batch.x.to_device(Device::cuda_if_available()))
            .cross_entropy_for_logits(&batch.y.to_device(Device::cuda_if_available()));
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
            let ok = 0;
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
    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    //let net = Net::new(&vs.root());
    let net = net(&vs.root());
//    println!("scrambled \n{}", scrambled());

    if Path::new(MODEL_STORE_PATH).exists() {
        println!("pre-trained model found in file {}", MODEL_STORE_PATH);
        vs.load(MODEL_STORE_PATH)?;
        println!("pre-trained model loaded from file {}", MODEL_STORE_PATH);
    } else {
        println!("training, then saving as {}", MODEL_STORE_PATH);
        let opt = nn::Adam::default().build(&vs, 1e-3)?;
        train(opt, &net);
        vs.save(MODEL_STORE_PATH)?;
    }

    //net.forward(&flower_x_train).print();
    //net.forward(&flower_x_train).argmax1(-1, false).print();
    //solve(&net, scrambled(), 500);
    for example in examples() {
        println!("");
        if example.is_solvable() {
            solve(&net, example, 3500, false);
        } else {
            println!("{} example not solvable", example);
        }
    }
    println!("----GUI-----");
    main_gui();
    // let mut f = Field::new();
    // println!("field \n{} {:?}", f, f.moves());
    // f.mov(1);
    //
    // println!("field \n{}", f);
    // f.mov(0);
    // println!("field \n{}", f);
    Ok(())
}

fn solve(net: &impl Module, mut scr: Field, max_steps: i32, verbose: bool) -> Option<i32> {
    if !verbose {
        println!("{} to be solved: ", scr);
    }
    //let mut rng = rand::thread_rng();
    let seed: [u8; 32] = b"123456789012345678901234567890A-".clone();
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let mut explored_set = HashSet::new();
    explored_set.insert(scr.cells);
    let mut err_loop = 0;
    let mut err_revisit = 0;

    for i in 0..max_steps {
        if !scr.is_solvable() {
            println!("not solvable \n{}", scr);
            panic!("not solvable scrambled");
        }
        let fwd: Tensor = net.forward(&features(&scr));
        //fwd.print();
        let fwd_max = fwd.argmax(-1, false);
        //fwd_max.print();
        let pred = fwd_max.double_value(&[0]);
        if verbose {
            println!("{} num={} - scrambled; moves={:?} pred={}", scr, i, scr.moves(), pred);
        }
        //let old_empty = scr.empty;
        let mut moved = scr.mov_if_not_in(pred as usize, &explored_set, verbose);
        if scr.is_done() {
            println!("DONE, SOLVED!!!! i={} err_loop={} err_revisit={}", i, err_loop, err_revisit);
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
            err_loop += 1;
            if verbose {
                println!("LOOP doing move {}", mov);
            }
            moved = scr.mov_if_not_in(mov, &explored_set, verbose);
            moves.remove_item(&mov);
        }
        if !moved {
            let moves = scr.moves();
            let mov = moves[rng.gen::<usize>() % moves.len()];
            err_revisit += 1;
            if verbose {
                println!("MOVE TO REPEATED STATE!!!!! {}", mov);
            }
            scr.mov(mov);
        }
        explored_set.insert(scr.cells);
    }
    println!("FAILED TO SOLVE i={} err_loop={} err_revisit={}", max_steps, err_loop, err_revisit);
    None
}
/*
struct Gui {
    application: Arc<Application>,
    builder: Builder,
    field: Field,
}

impl Gui {
    fn new() -> Gui {
        let application = Arc::new(Application::new(
            Some("com.github.gtk-rs.examples.basic"),
            Default::default(),
        ).expect("failed to initialize GTK application"));
        let field = Field::new();
// we bake our glade file into the application code itself
        let glade_src = include_str!("window.glade");
        let builder = Builder::new_from_string(glade_src);

        Gui{
            application,
            builder,
            field
        }
    }
    fn init(&self){


        self.application.connect_activate( |_| {

// this builder provides access to all components of the defined ui

// glade allows us to get UI elements by id but we need to specify the type
            let window: Window = self.builder.get_object::<Window>("wnd_main").expect("Couldn't get window") as Window;
            window.set_title("Memegen");
            let app:&Application = self.application.borrow();
            window.set_application(Some(app));

//             for i in 0..16 {
//                 let bname = format!("b{}", i);
//                 let mut btn_save: Button = self.builder.get_object::<Button>(&bname).expect("Couldn't get btn_save");
//                 btn_save.set_label(&format!("{}", self.field.cells[i] + 1));
//
//                 btn_save.connect_clicked( |_| {
//                     println!("clicked");
//                     let mut b: Button = self.builder.get_object::<Button>("b0").expect("Couldn't get btn_save");
//                     b.set_label("3");
// //btn_save.la
//                 });
//             }
//let btnRef = &btn_save;
// let window = ApplicationWindow::new(app);
// window.set_title("First GTK+ Program");
// window.set_default_size(350, 70);


// let grid = GridBuilder::new().build();
// let button = Button::new_with_label("Click me!");
// button.connect_clicked(|_| {
//     println!("Clicked!");
// });
// window.add(&button);
//
// let button2 = Button::new_with_label("Click me2!");
// window.add(&button2);
            window.show_all();
        });

        self.application.run(&[]);

    }
}
*/

fn build_ui(application: &gtk::Application) {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let net = net(&vs.root());
    if let Ok(_) = vs.load(MODEL_STORE_PATH) {} else {
        panic!("build_ui vs.load");
    }

    let window = gtk::ApplicationWindow::new(application);

    window.set_title("First GTK+ Program");
    window.set_border_width(10);
    window.set_position(gtk::WindowPosition::Center);
    window.set_default_size(350, 70);

    let gbuilder: GridBuilder = GridBuilder::new();
    let grid: Grid = gbuilder.build();
    let mut buttons: Vec<Button> = Vec::new();
    for i in 0..16 {
        let label = if i == 15 { "".to_string() } else { format!("{}", i + 1) };
        let button = gtk::Button::new_with_label(&label);
        buttons.push(button);
        //gbuilder = gbuilder.child(&button);
    }
    let arc_btns = Arc::new(buttons);
    for i in 0..16i32 {
        //let label = if i == 15 { "".to_string() } else { format!("{}", i + 1) };
        //let mut button = gtk::Button::new_with_label(&label);
        let arc_btns_copy = arc_btns.clone();
        let button: &Button = &(&arc_btns_copy)[i as usize];
        let arc_btns_copy2 = arc_btns.clone();
        button.connect_clicked(move |_| {
            let mut field = field_from_buttons(&arc_btns_copy2);
            let moves = field.moves();
            let iusize = i as usize;
            let valid_move = moves.contains(&iusize);
            println!("Clicked! {} {:?} {}", field, moves, valid_move);
            if valid_move {
                field.mov(iusize);
                println!("Moved {}", field);
                relabel_buttons(&arc_btns_copy2, field)
            }
            //(&arc_btns_copy2)[0].set_label("==");
            //buttons[0].set_label("==");
        });
        grid.attach(button, (i % 4) * 50, (i / 4) * 20, 50, 20);
        //button.set_label("sdf");

        //(&mut arc_btns).push(button);
        //gbuilder = gbuilder.child(&button);
    }

    let solve_btn = gtk::Button::new_with_label("Solve");
    grid.attach(&solve_btn, 50 * 5, 0, 50, 20);
    let arc_btns_copy3 = arc_btns.clone();
    solve_btn.connect_clicked(move |b: &Button| {
        let solvable = solve(&net, field_from_buttons(&arc_btns_copy3), 50, true);

        let mut field = field_from_buttons(&arc_btns_copy3);
        let moves = field.moves();
        println!("Clicked [solve]! {} {:?} solvable={:?}", field, moves, solvable);
        let label = match solvable {
            Some(i) => format!("Solve-{}", i),
            _ => "Solve-?".to_string()
        };
        //solve_btn.set_label(&label);
        b.set_label(&label);


        let fwd: Tensor = net.forward(&features(&field));
        //fwd.print();
        let fwd_max = fwd.argmax(-1, false);
        //fwd_max.print();
        let pred = fwd_max.double_value(&[0]);
        println!("Clicked [solve]! {} {:?} pred={}", field, moves, pred);
        //println!("{} num={} - scrambled; moves={:?} pred={}", scr, i, scr.moves(), pred);

        field.mov(pred as usize);
        //field.mov(moves[0]);
        println!("Moved {}", field);
        relabel_buttons(&arc_btns_copy3, field)
    });

    //window.add(&button);
    window.add(&grid);

    window.show_all();
}

fn relabel_buttons(arc_btns_copy2: &Arc<Vec<Button>>, field: Field) -> () {
    for j in 0..SIZE * SIZE {
        let label = if field.cells[j] == 0 { "".to_string() } else { format!("{}", field.cells[j]) };
        (&arc_btns_copy2)[j].set_label(&label);
    }
}

fn field_from_buttons(buttons: &Vec<Button>) -> Field {
    let mut cells: [u8; SIZE * SIZE] = [0; SIZE * SIZE];
    for i in 0..buttons.len() {
        let label = buttons[i].get_label();
        let label_str = label.unwrap();
        if label_str.len() > 0 {
            cells[i] = label_str.parse::<u8>().unwrap();
        }
    }
    Field::new_with_cells(cells)
}

fn main_gui() {
    println!("{}", Field::new());

    // let gui = Gui::new();
    let application =
        gtk::Application::new(Some("com.github.gtk-rs.examples.basic"), Default::default())
            .expect("Initialization failed...");

    application.connect_activate(|app| {
        build_ui(app);
    });

    //application.run(&args().collect::<Vec<_>>());
    application.run(&[]);
}
