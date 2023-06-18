use pyo3::{pymodule, types::PyModule, PyResult, Python};
// use std::arch::x86_64::*;

#[macro_use]
extern crate lazy_static;

mod algorithm;
pub mod bitboard;
mod board;
mod test;

mod const_vec {
    use regex::Regex;
    use std::arch::x86_64::*;
    lazy_static! {
        pub static ref ONES: __m256i = unsafe {
            _mm256_set_epi16(
                -1i16, -1i16, -1i16, -1i16, -1i16, -1i16, -1i16, -1i16, -1i16, -1i16, -1i16, -1i16,
                -1i16, -1i16, -1i16, -1i16,
            )
        };
        pub static ref SHIFT_15_MASK: __m256i = unsafe {
            _mm256_set_epi16(
                0, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16,
                -4i16, -4i16, -4i16, 0,
            )
        };
        pub static ref SHIFT_15_REV_MASK: __m256i = unsafe {
            _mm256_set_epi16(
                0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE,
                0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0, 0,
            )
        };
        pub static ref SHIFT_17_MASK: __m256i = unsafe {
            _mm256_set_epi16(
                0, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE,
                0x7FFE, 0x7FFE, 0x7FFE, 0x7FFE, 0,
            )
        };
        pub static ref SHIFT_17_REV_MASK: __m256i = unsafe {
            _mm256_set_epi16(
                -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16, -4i16,
                -4i16, -4i16, 0, 0,
            )
        };
        pub static ref SHIFT_16_MASK: __m256i = unsafe {
            _mm256_set_epi16(
                0, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16,
                -2i16, -2i16, -2i16, 0,
            )
        };
        pub static ref SHIFT_16_REV_MASK: __m256i = unsafe {
            _mm256_set_epi16(
                -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16, -2i16,
                -2i16, -2i16, 0, 0,
            )
        };
        pub static ref REG_REN: Regex =
            Regex::new(r"[^O]B(OOBB|OBOB|OBBO|BOOB|BOBO|BBOO)B[^O]").unwrap();
        pub static ref REG_SAN: Regex = Regex::new(r"[^X]B(OOOB|OOBO|OBOO|BOOO)B[^O]").unwrap();
        pub static ref REG_SHI: Regex =
            Regex::new(r"[^O](OOOOB|OOOBO|OOBOO|OBOOO|BOOOO)[^O]").unwrap();
        pub static ref REG_TATUSHI: Regex = Regex::new(r"[^O]BOOOOB[^O]").unwrap();
        pub static ref REG_GO: Regex = Regex::new(r"[^O]OOOOO[^O]").unwrap();
    }
}

#[pymodule]
fn gomoku(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the python classes
    m.add_class::<board::Board>()?;
    // Register the functions as a submodule
    // let funcs = PyModule::new(py, "functions")?;
    // functions::init_module(funcs)?;
    // m.add_submodule(funcs)?;

    Ok(())
}
