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
        for i in &self.cells {
            for j in 0..16 {
                if j == *i {
                    res.push(1f32);
                } else {
                    res.push(0f32);
                }
            }
            // let mut val = *i;
            // for _ in 0..4 { // 4 bits
            //     res.push((val % 2) as f32);
            //     val = val / 2;
            // }
        }
        res
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
    pub fn mov_if_can(&mut self, pos: usize) -> bool {
        let val = self.cells[pos];
        assert_eq!(0, self.cells[self.empty]);
        if !self.moves().contains(&pos) {
            println!("CANNOT MOVE!!!");
            return false;
        }
        self.cells[pos] = 0;
        self.cells[self.empty] = val;
        self.empty = pos;
        true
    }
    pub fn mov_if_not_in(&mut self, pos: usize, old: &HashSet<[u8; SIZE * SIZE]>) -> bool {
        let val = self.cells[pos];
        assert_eq!(0, self.cells[self.empty]);
        if !self.moves().contains(&pos) {
            println!("CANNOT MOVE!!!");
            return false;
        }
        self.cells[pos] = 0;
        self.cells[self.empty] = val;
        if (old.contains(&self.cells)) {
            // undo at once
            println!("MOVE TO OLD STATE CANCELLED!!!");
            self.cells[pos] = val;
            self.cells[self.empty] = 0;
            return false;
        }
        self.empty = pos;
        true
    }
    fn rowCol(pos: usize) -> (usize, usize) {
        (pos / SIZE, pos % SIZE)
    }
    pub fn moves(&self) -> Vec<usize> {
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

pub fn scrambled() -> Field {
    //let mut rng =  rand::thread_rng();
    let seed: [u8; 32] = b"123456789012345678901234567890Ab".clone();
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let mut f = Field::new();
    for i in 0..20000 {
        let moves = f.moves();
        let n: usize = rng.gen();
        let mov = moves[n % moves.len()];
        f.mov(mov);
        //println!("scrambling mov={} of={:?} \n{}", mov, moves, f);
    }
    f
}

pub fn example() -> Field {
    //Field::new_with_cells([1,2,3,4, 5,6,7,8, 9,10,11,12, 15,13,14,0]) // solved in 17 moves
    // Field::new_with_cells([
    //     15, 14, 8, 12,
    //     10, 11, 9, 13,
    //     2, 6, 5, 1,
    //     3, 7, 4, 0])
    Field::new_with_cells([
        15, 14, 8, 12,
        10, 11, 9, 13,
        2, 6, 5, 1,
        3, 7, 4, 0])
}
