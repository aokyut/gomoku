use crate::board::Board;
use crate::board::{next, random_action, rule_action, show, uct_action};
#[test]
fn board_test() -> () {
    let mut b = unsafe { Board::new(true) };
    unsafe {
        loop {
            show(&b);
            println!(
                "stone:{}, is_black:{}, is_done:{}",
                b.stone, b.is_black, b.is_done
            );
            let mut action;
            if b.turn % 2 == 1 {
                action = rule_action(&b);
            } else {
                action = rule_action(&b);
                // action = random_action(&b);
            }
            b = next(&b, action);
            if b.is_done {
                break;
            }
        }
        show(&b)
    }
}
