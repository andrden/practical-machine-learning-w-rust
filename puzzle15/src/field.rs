use std::fmt::{Display, Formatter};
use std::fmt;

extern crate rand;

use rand::Rng;
use self::rand::prelude::*;
use std::collections::HashSet;

pub const SIZE: usize = 4;
const CHARS: &str = "0123456789ABCDEFGHIJK";

#[derive(Clone)]
pub struct Field {
    pub empty: usize,
    pub cells: [u8; SIZE * SIZE],
}

impl Field {
    pub fn encode_move_as_direction(from: usize, to: usize) -> u8 {
        if to == from + 1 {
            return 0;
        }
        if to == from - 1 {
            return 1;
        }
        if to < from {
            return 2;
        }
        return 3;
    }
    pub fn new() -> Field {
        let mut f = Field {
            empty: SIZE * SIZE - 1,
            cells: [0u8; SIZE * SIZE],
        };
        for i in 0..SIZE * SIZE - 1 {
            f.cells[i] = (i + 1) as u8;
        }
        f
    }
    pub fn new_with_cells(cells: [u8; SIZE * SIZE]) -> Field {
        let mut e = 0;
        for i in 0..SIZE * SIZE {
            if cells[i] == 0 {
                e = i;
            }
        }
        Field { empty: e, cells }
    }
    pub fn features(&self) -> Vec<f32> {
        //let mut xi: Vec<f32> = (*xyi.0).cells.iter().map(|v| *v as f32).collect();
        let mut res: Vec<f32> = Vec::new();
        let mut back_ref = [0u8; SIZE * SIZE];
        for (index, i) in self.cells.iter().enumerate() {
            back_ref[*i as usize] = index as u8;
            Field::one_hot16(&mut res, i);
            // let mut val = *i;
            // for _ in 0..4 { // 4 bits
            //     res.push((val % 2) as f32);
            //     val = val / 2;
            // }
        }
        for i in &back_ref {
            Field::one_hot16(&mut res, i);
        }
        res
    }

    fn one_hot16(res: &mut Vec<f32>, i: &u8) -> () {
        for j in 0..16 {
            if j == *i {
                res.push(1f32);
            } else {
                res.push(0f32);
            }
        }
    }

    pub fn is_done(&self) -> bool {
        self.cells == Field::new().cells
    }
    pub fn mov(&mut self, pos: usize) {
        let val = self.cells[pos];
        assert_eq!(0, self.cells[self.empty]);
        assert!(self.moves().contains(&pos));
        self.cells[pos] = 0;
        self.cells[self.empty] = val;
        self.empty = pos;
    }
    // pub fn mov_if_can(&mut self, pos: usize) -> bool {
    //     let val = self.cells[pos];
    //     assert_eq!(0, self.cells[self.empty]);
    //     if !self.moves().contains(&pos) {
    //         println!("CANNOT MOVE!!!");
    //         return false;
    //     }
    //     self.cells[pos] = 0;
    //     self.cells[self.empty] = val;
    //     self.empty = pos;
    //     true
    // }

    pub fn mov_if_not_in(&mut self, pos: usize, old: &HashSet<[u8; SIZE * SIZE]>, verbose: bool) -> bool {
        let val = self.cells[pos];
        assert_eq!(0, self.cells[self.empty]);
        if !self.moves().contains(&pos) {
            if verbose {
                println!("CANNOT MOVE!!!");
            }
            return false;
        }
        self.cells[pos] = 0;
        self.cells[self.empty] = val;
        if old.contains(&self.cells) {
            // undo at once
            if verbose {
                println!("MOVE TO OLD STATE CANCELLED!!!");
            }
            self.cells[pos] = val;
            self.cells[self.empty] = 0;
            return false;
        }
        self.empty = pos;
        true
    }

    fn row_col(pos: usize) -> (usize, usize) {
        (pos / SIZE, pos % SIZE)
    }

    pub fn moves(&self) -> Vec<usize> {
        let mut res = Vec::new();
        let (row, col) = Field::row_col(self.empty);
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

    // fn get_inv_count(&self) -> i32 {
    //     let mut inv_count = 0;
    //     for i in 0..SIZE * SIZE {
    //         for j in (i+1)..SIZE * SIZE {
    //             // count pairs(i, j) such that i appears
    //             // before j, but i > j.
    //             // let mut vi = self.cells[i];
    //             // if vi==0 {
    //             //     vi = (SIZE*SIZE) as u8;
    //             // }
    //             // let mut vj = self.cells[j];
    //             // if vj==0 {
    //             //     vj = (SIZE*SIZE) as u8;
    //             // }
    //
    //             if self.cells[j] != 0 && self.cells[i] != 0 && self.cells[i] > self.cells[j] {
    //                 inv_count += 1;
    //             }
    //         }
    //     }
    //     inv_count
    // }

    fn get_inv_count(&self) -> i32 {
        let mut inv_count = 0;
        for i in 0..SIZE * SIZE {
            for j in (i + 1)..SIZE * SIZE {
                // count pairs(i, j) such that i appears
                // before j, but i > j.
                let mut vi = self.cells[i];
                if vi == 0 {
                    vi = (SIZE * SIZE) as u8;
                }
                let mut vj = self.cells[j];
                if vj == 0 {
                    vj = (SIZE * SIZE) as u8;
                }

                if vi > vj {
                    inv_count += 1;
                }
            }
        }
        inv_count
    }

    // find Position of blank from bottom
    // fn find_x_position(&self) -> usize {
    //     SIZE - (self.empty % SIZE)
    // }

    fn taxicab_space_distance(&self) -> usize {
        SIZE - 1 - (self.empty % SIZE) + SIZE - 1 - self.empty / SIZE
    }

    // This function returns true if given
    // instance of N*N - 1 puzzle is solvable
    pub fn is_solvable(&self) -> bool {
        // Count inversions in given puzzle
        let inv_count = self.get_inv_count();

        //let pos = self.find_x_position();
        //(pos + inv_count as usize) % 2 == 0
        (self.taxicab_space_distance() + inv_count as usize) % 2 == 0
    }
}

impl Display for Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for i in 0..SIZE {
            write!(f, "{}", "|")?;
            for j in 0..SIZE {
                let val = self.cells[i * SIZE + j] as usize;
                if val == 0 {
                    f.write_str(" ")?;
                } else {
                    f.write_str(&CHARS[val..val + 1])?;
                }
            }
            write!(f, "{}", "|\n")?;
        }
        Ok(())
    }
}

pub fn scrambled() -> Field {
    //let mut rng =  rand::thread_rng();
    let seed: [u8; 32] = b"123456789012345678901234567890Ab".clone();
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let mut f = Field::new();
    for _i in 0..20000 {
        let moves = f.moves();
        let n: usize = rng.gen();
        let mov = moves[n % moves.len()];
        f.mov(mov);
        //println!("scrambling mov={} of={:?} \n{}", mov, moves, f);
    }
    f
}

pub fn examples() -> Vec<Field> {
    vec![
        scrambled(), // solved in 187 steps => 79,167,87 in 16k-net (3 numbers - repeated training for more 30k epochs, 2000 seconds)
        Field::new_with_cells([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15]),
        Field::new_with_cells([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15]),
        Field::new_with_cells([0, 5, 3, 7, 2, 11, 4, 8, 1, 13, 6, 15, 10, 9, 12, 14]), // 29,137,29
        Field::new_with_cells([3, 2, 8, 4, 5, 6, 7, 14, 1, 10, 11, 12, 13, 9, 15, 0]), // 85,59,47 (actually can be solved in about 60 moves)
        Field::new_with_cells([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 13, 14, 0]), // solved in 17 moves
        Field::new_with_cells([2, 4, 1, 5, 6, 9, 10, 12, 7, 14, 11, 3, 13, 0, 8, 15]), // example not solvable
        Field::new_with_cells([4, 10, 11, 5, 15, 7, 9, 3, 12, 8, 13, 14, 2, 1, 6, 16]), // example not solvable
        Field::new_with_cells([1, 5, 8, 0, 11, 7, 13, 14, 9, 12, 10, 2, 3, 6, 4, 15]), // 102 moves => 74,102,74 in 16k-net
        Field::new_with_cells([1, 8, 15, 5, 6, 11, 12, 13, 9, 0, 3, 4, 10, 7, 2, 14]), // 160 moves => 62,206,78 in 16k-net
        Field::new_with_cells([ //- unsolvable, broken?
            15, 14, 8, 12,
            10, 11, 9, 13,
            2, 6, 5, 1,
            3, 7, 4, 0]),
        Field::new_with_cells([ // solved in 119 moves => 125,107,173 in 16k-net
            14, 15, 8, 12,
            10, 11, 9, 13,
            2, 6, 5, 1,
            3, 7, 4, 0]),
        Field::new_with_cells([ // solved in 201 moves,   https://www.jaapsch.net/puzzles/javascript/fifteenj.htm solves it in 184 (or 194) moves
            0, 12, 10, 13, // => 109,233,137 in 16k-net
            15, 11, 14, 9,
            7, 8, 6, 2,
            4, 3, 5, 1])]
    // if !res.is_solvable() {
    //     panic!("not solvable example()")
    // }
    // res
}

/*
after +90k or +120k epochs:
epoch: 29000 train loss:  0.00909 sum_loss=5.919632107776124 err=11168677 rate= 0.00 sec=2049
epoch: 30000 train loss:  0.00877 sum_loss=5.962859675229993 err=11168677 rate= 0.00 sec=2120
*/