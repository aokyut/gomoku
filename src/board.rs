use crate::bitboard::{action_mask_from_bw, is_done, not, to_int64s, to_uint8, unpackbits};
use crate::const_vec::{REG_GO, REG_REN, REG_SAN, REG_SHI, REG_TATUSHI};
use numpy::ndarray::arr1;
use numpy::{convert::IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::{PyResult, Python};
use rand::Rng;
use regex::Regex;
use std::arch::x86_64::*;
use std::collections::HashMap;

// TODO: swappedの代わりにturn変数を使用する。

#[pyclass]
pub struct Board {
    b: __m256i,
    w: __m256i,
    pub is_black: bool,
    pub swap: bool,
    pub stone: i32,
    pub turn: i32,
    pub is_done: bool,
    pub value: i32,
}
pub unsafe fn valid_actions(b: &Board) -> Vec<i32> {
    let mask: [u8; 225] = unpackbits(not(_mm256_or_si256(b.b, b.w)));
    let mut action_vec = vec![0; 0];
    if b.turn % 2 == 0 || b.turn >= 5 {
        for i in 0..225 {
            if mask[i] == 1 {
                action_vec.push(i as i32)
            }
        }
    }
    if b.swap && b.turn % 2 == 1 && b.stone <= 3 {
        action_vec.push(225)
    }
    return action_vec;
}
pub unsafe fn hash(b: &Board) -> String {
    let mut s = String::new();
    let black: [i64; 4] = to_int64s(b.b);
    let white: [i64; 4] = to_int64s(b.w);
    for i in 0..4 {
        let black = black[i];
        let white = white[i];
        s.push_str(&String::from(black.to_string()));
        s.push_str(":");
        s.push_str(&String::from(white.to_string()));
        s.push_str(",");
    }
    s.push_str(&String::from((b.turn % 2).to_string()));
    return s;
}

pub unsafe fn next(b: &Board, action: i32) -> Board {
    if action == 225 {
        return Board {
            b: b.b,
            w: b.w,
            is_black: b.is_black,
            swap: b.swap,
            stone: b.stone,
            turn: b.turn + 1,
            is_done: b.is_done,
            value: 0,
        };
    }
    let action = 16 * (action / 15) + (action % 15);
    let action_vec: __m256i;
    if action < 128 {
        if action < 64 {
            action_vec = _mm256_set_epi64x(0, 0, 0, 1 << action);
        } else {
            action_vec = _mm256_set_epi64x(0, 0, 1 << (action - 64), 0);
        }
    } else {
        if action < 192 {
            action_vec = _mm256_set_epi64x(0, 1 << (action - 128), 0, 0);
        } else {
            action_vec = _mm256_set_epi64x(1 << (action - 192), 0, 0, 0);
        }
    }

    if b.is_black {
        let new_black = _mm256_or_si256(b.b, action_vec);
        if is_done(new_black) {
            return Board {
                b: new_black,
                w: b.w,
                is_black: false,
                swap: b.swap,
                stone: b.stone + 1,
                turn: b.turn + 1,
                is_done: true,
                value: if b.turn % 2 == 0 { 1 } else { -1 },
            };
        }
        return Board {
            b: new_black,
            w: b.w,
            is_black: false,
            swap: b.swap,
            stone: b.stone + 1,
            turn: b.turn + 1,
            is_done: b.stone + 1 >= 200,
            value: 0,
        };
    } else {
        let new_white = _mm256_or_si256(b.w, action_vec);
        if is_done(new_white) {
            return Board {
                b: b.b,
                w: new_white,
                is_black: true,
                swap: b.swap,
                stone: b.stone + 1,
                turn: b.turn + 1,
                is_done: true,
                value: if b.turn % 2 == 0 { 1 } else { -1 },
            };
        }
        return Board {
            b: b.b,
            w: new_white,
            is_black: true,
            swap: b.swap,
            stone: b.stone + 1,
            turn: b.turn + 1,
            is_done: b.stone + 1 >= 200,
            value: 0,
        };
    }
}

pub unsafe fn random_action(b: &Board) -> i32 {
    let actions = valid_actions(&b);
    let mut rng = rand::thread_rng();
    let action = rng.gen_range(0..actions.len());
    // println!(
    //     "stone:{}, swap:{}, turn:{}, action_len:{}, action:{}",
    //     b.stone,
    //     b.swap,
    //     b.turn,
    //     actions.len(),
    //     actions[action],
    // );
    return actions[action];
}

// 最後までランダムにゲームをプレイした際の結果を出力する。
unsafe fn playout(b: &Board) -> i32 {
    let action = random_action(b);
    let mut curr_b = next(b, action);

    loop {
        let action = random_action(&curr_b);
        let next_b = next(&curr_b, action);
        if next_b.is_done {
            return next_b.value;
        }
        curr_b = next_b;
    }
    return 0;
}

// uct
pub unsafe fn uct_action(b: &Board, search_n: i32) -> i32 {
    let actions = valid_actions(&b);
    // println!("turn:{}, actions:{}", b.turn, actions.len());
    let mut w = HashMap::new();
    let mut n = HashMap::new();
    let mut n_sum: f64 = 0.0;
    for i in &actions {
        n.insert(*i, 2.0);
        w.insert(*i, 1.0);
        n_sum += 2.0;
    }
    for i in 0..search_n {
        // actionを選択する
        // TODO: 高速な探索方法がありそう
        let mut max_score = 0.0;
        let mut max_action = -1;
        let mut max_n = -1.0;
        let mut max_w = -1.0;
        let ln_N: f64 = 2.0 * n_sum.ln();
        for j in &actions {
            let w = match w.get(j) {
                None => -1.0,
                Some(x) => *x,
            };
            let n = match n.get(j) {
                None => -1.0,
                Some(x) => *x,
            };
            let score: f64 = w / n + (&ln_N / n).sqrt();
            if score > max_score {
                max_score = score;
                max_action = *j;
                max_n = n;
                max_w = w;
            }
        }
        let v: i32 = playout(&next(b, max_action));
        n.insert(max_action, max_n + 1.0);
        w.insert(
            max_action,
            if b.turn % 2 == 0 && v > 0 {
                max_w + 1.0
            } else if b.turn % 2 == 1 && v < 0 {
                max_w + 1.0
            } else {
                max_w
            },
        );
    }
    let mut max_score = 0.0;
    let mut max_action = -1;
    let ln_N: f64 = 2.0 * n_sum.ln();
    for j in &actions {
        let n = match n.get(j) {
            None => -1.0,
            Some(x) => *x,
        };
        let score: f64 = n;
        // println!("action:{}, score:{}", j, score);
        if score > max_score {
            max_score = score;
            max_action = *j;
        }
    }
    return max_action;
}

pub unsafe fn mtcs_action(b: &Board, search_num: i32) -> i32 {
    return 0;
}

//三や連の数を数えてそれぞれの手の中で最も点数の高い行動を選択
const score_ren: f32 = 1.0;
const score_san: f32 = 2.0;
const score_ken: f32 = 4.0;
const score_shi: f32 = 1.0;
const score_tatushi: f32 = 5.0;
const score_go: f32 = 100.0;
struct V(i32, i32);
const search_vecs: [V; 4] = [V(0, 1), V(1, 0), V(1, 1), V(-1, 1)];

pub unsafe fn rule_action(b: &Board) -> i32 {
    if b.turn % 2 == 1 && b.turn < 6 {
        return 225;
    }

    let arr_tar: [u8; 225];
    let arr_enm: [u8; 225];
    if b.is_black {
        arr_tar = unpackbits(b.b);
        arr_enm = unpackbits(b.w);
    } else {
        arr_tar = unpackbits(b.w);
        arr_enm = unpackbits(b.b);
    }
    let mut max_score = -1.0;
    let mut max_action = 1;
    let mut arr_char: [char; 225] = ['B'; 225];
    let mut arr_char_enm: [char; 225] = ['B'; 225];
    let mut rng = rand::thread_rng();
    for i in 0..225 {
        if arr_enm[i] == 1 {
            arr_char[i] = 'X';
            arr_char_enm[i] = 'O';
        } else if arr_tar[i] == 1 {
            arr_char[i] = 'O';
            arr_char_enm[i] = 'X';
        }
    }

    for i in 0..225 {
        if arr_char[i] != 'B' {
            continue;
        }
        arr_char[i] = 'O';
        let mut score = 0.0;
        for search_vec in search_vecs {
            // 敵側の石を見て三や四があれば止める。
            let mut tar_s_enm = String::new();
            let mut tar_s = String::new();
            let x = (i % 15) as i32;
            let y = (i / 15) as i32;

            for j in -5..6 {
                tar_s_enm.push(get_char(
                    arr_char_enm,
                    y + search_vec.0 * j,
                    x + search_vec.1 * j,
                ));
                tar_s.push(get_char(
                    arr_char,
                    y + search_vec.0 * j,
                    x + search_vec.1 * j,
                ));
            }
            //敵の石に三か四が無いかチェックする
            if REG_SAN.is_match(&tar_s_enm) {
                score += score_san * 3.0;
            }
            if REG_REN.is_match(&tar_s_enm) {
                score += score_ren;
            }
            if REG_SHI.is_match(&tar_s_enm) {
                score += score_tatushi * 2.0;
            }

            if REG_REN.is_match(&tar_s) {
                score += score_ren;
            }
            if REG_SAN.is_match(&tar_s) {
                score += score_san;
            }
            if REG_SHI.is_match(&tar_s) {
                score += score_shi;
            }
            if REG_TATUSHI.is_match(&tar_s) {
                score += score_tatushi;
            }
            if REG_GO.is_match(&tar_s) {
                score += score_go;
            }
            // 行動に乱数を追加
            score += rng.gen::<f32>();
        }
        // println!("{}: {}", i, score);
        if max_score < score {
            max_score = score;
            max_action = i
        }
        arr_char[i] = 'B';
    }

    if max_score == 0.0 {
        return random_action(b);
    }
    return max_action as i32;
}

unsafe fn get_char(arr_char: [char; 225], i: i32, j: i32) -> char {
    if i < 0 || i > 14 || j < 0 || j > 14 {
        return 'X';
    }
    return arr_char[(15 * i + j) as usize];
}

pub unsafe fn show(b: &Board) -> () {
    let mut s = String::new();
    let black = to_int64s(b.b);
    let white = to_int64s(b.w);
    for i in 0..4 {
        let tar_black = black[i];
        let tar_white = white[i];
        for j in 0..64 {
            if j % 16 == 15 {
                continue;
            }
            if i == 3 && j == 48 {
                s.push('\n');
                break;
            }
            if (tar_black >> j) & 1 == 1 {
                s.push('X');
            } else if (tar_white >> j) & 1 == 1 {
                s.push('O');
            } else {
                s.push('-');
            }
            if j % 16 == 14 && i * j != 189 {
                s.push('\n');
            }
        }
    }
    println!("{}", s);
}

#[pymethods]
impl Board {
    #[new]
    #[args(swap = "true")]
    pub unsafe fn new(swap: bool) -> Self {
        Board {
            b: _mm256_setzero_si256(),
            w: _mm256_setzero_si256(),
            is_black: true,
            swap: swap,
            turn: 0,
            stone: 0,
            is_done: false,
            value: 0,
        }
    }

    unsafe fn show<'py>(&self, _py: Python) -> PyResult<()> {
        let mut s = String::new();
        let black = to_int64s(self.b);
        let white = to_int64s(self.w);
        for i in 0..4 {
            let tar_black = black[i];
            let tar_white = white[i];
            for j in 0..64 {
                if j % 16 == 15 {
                    continue;
                }
                if i == 3 && j == 48 {
                    s.push('\n');
                    break;
                }
                if (tar_black >> j) & 1 == 1 {
                    s.push('X');
                } else if (tar_white >> j) & 1 == 1 {
                    s.push('O');
                } else {
                    s.push('-');
                }
                if j % 16 == 14 && i * j != 189 {
                    s.push('\n');
                }
            }
        }
        println!("{}", s);
        Ok(())
    }

    unsafe fn action_mask<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(&'py PyArray1<u8>, &'py PyArray1<u8>)> {
        let swap = if self.swap && self.turn % 2 == 1 && self.stone <= 3 {
            1
        } else {
            0
        };
        let mut mask: [u8; 225];
        // 3手目以前の先手または3手目以降の時だけ石を打てる
        if self.turn % 2 == 0 || self.turn > 5 {
            mask = unpackbits(not(_mm256_or_si256(self.b, self.w)));
        } else {
            mask = [0; 225];
        }
        let swap: [u8; 1] = [swap];
        let mask: &PyArray1<u8> = arr1(&mask).into_pyarray(py);
        let swap: &PyArray1<u8> = arr1(&swap).into_pyarray(py);
        Ok((mask, swap))
    }

    unsafe fn reset<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(&'py PyArray1<u8>, &'py PyArray1<u8>, bool, bool)> {
        self.b = _mm256_setzero_si256();
        self.w = _mm256_setzero_si256();
        self.is_black = true;
        self.stone = 0;
        self.turn = 0;
        self.is_done = false;
        self.value = 0;
        let black_board = arr1(&unpackbits(self.b)).into_pyarray(py);
        let white_board = arr1(&unpackbits(self.w)).into_pyarray(py);
        Ok((black_board, white_board, false, true))
    }

    // TODO: 序盤は白番がswapのみしかできないようにする
    unsafe fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: usize,
    ) -> PyResult<(&'py PyArray1<u8>, &'py PyArray1<u8>, bool, bool, i32)> {
        self.turn += 1;
        if action == 225 {
            return Ok((
                arr1(&unpackbits(self.b)).into_pyarray(py),
                arr1(&unpackbits(self.w)).into_pyarray(py),
                false,
                self.is_black,
                0,
            ));
        }
        let action = 16 * (action / 15) + (action % 15);
        let action_vec: __m256i;
        if action < 128 {
            if action < 64 {
                action_vec = _mm256_set_epi64x(0, 0, 0, 1 << action);
            } else {
                action_vec = _mm256_set_epi64x(0, 0, 1 << (action - 64), 0);
            }
        } else {
            if action < 192 {
                action_vec = _mm256_set_epi64x(0, 1 << (action - 128), 0, 0);
            } else {
                action_vec = _mm256_set_epi64x(1 << (action - 192), 0, 0, 0);
            }
        }

        self.stone += 1;

        if self.is_black {
            self.b = _mm256_or_si256(self.b, action_vec);
            if is_done(self.b) {
                return Ok((
                    arr1(&unpackbits(self.b)).into_pyarray(py),
                    arr1(&unpackbits(self.w)).into_pyarray(py),
                    true,
                    true,
                    {
                        if self.turn % 2 == 1 {
                            1
                        } else {
                            -1
                        }
                    },
                ));
            }
        } else {
            self.w = _mm256_or_si256(self.w, action_vec);
            if is_done(self.w) {
                return Ok((
                    arr1(&unpackbits(self.b)).into_pyarray(py),
                    arr1(&unpackbits(self.w)).into_pyarray(py),
                    true,
                    false,
                    {
                        if self.turn % 2 == 1 {
                            1
                        } else {
                            -1
                        }
                    },
                ));
            }
        }
        self.is_black ^= true;

        Ok((
            arr1(&unpackbits(self.b)).into_pyarray(py),
            arr1(&unpackbits(self.w)).into_pyarray(py),
            self.stone >= 200,
            self.is_black,
            0,
        ))
    }

    unsafe fn get_random_action<'py>(&self, _py: Python) -> PyResult<i32> {
        Ok(random_action(self))
    }
    unsafe fn get_uct_action<'py>(&self, _py: Python, i: i32) -> PyResult<i32> {
        Ok(uct_action(self, i))
    }
    unsafe fn get_rule_action<'py>(&self, _py: Python) -> PyResult<i32> {
        Ok(rule_action(self))
    }
    fn get_turns<'py>(&self, _py: Python) -> PyResult<i32> {
        Ok(self.turn)
    }
    fn get_is_black<'py>(&self, _py: Python) -> PyResult<bool> {
        Ok(self.is_black)
    }
}
