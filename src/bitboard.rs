use crate::const_vec::{
    ONES, SHIFT_15_MASK, SHIFT_15_REV_MASK, SHIFT_16_MASK, SHIFT_16_REV_MASK, SHIFT_17_MASK,
    SHIFT_17_REV_MASK,
};
use std::arch::x86_64::*;

pub unsafe fn shift_16_boards(a: __m256i) -> __m256i {
    let a_per = _mm256_permute4x64_epi64::<0x90>(a);
    return _mm256_and_si256(
        *SHIFT_16_MASK,
        _mm256_or_si256(_mm256_slli_epi64::<16>(a), _mm256_srli_epi64::<48>(a_per)),
    );
}

pub unsafe fn shift_16_boards_rev(a: __m256i) -> __m256i {
    let a_per = _mm256_permute4x64_epi64::<0xF9>(a);
    return _mm256_and_si256(
        *SHIFT_16_REV_MASK,
        _mm256_or_si256(_mm256_srli_epi64::<16>(a), _mm256_slli_epi64::<48>(a_per)),
    );
}

pub unsafe fn shift_1_boards(a: __m256i) -> __m256i {
    return _mm256_slli_epi16::<1>(a);
}

pub unsafe fn shift_1_boards_rev(a: __m256i) -> __m256i {
    return _mm256_srli_epi16::<1>(a);
}

pub unsafe fn shift_15_boards(a: __m256i) -> __m256i {
    let a_per = _mm256_permute4x64_epi64::<0x90>(a);
    return _mm256_and_si256(
        *SHIFT_15_MASK,
        _mm256_or_si256(_mm256_slli_epi64::<15>(a), _mm256_srli_epi64::<49>(a_per)),
    );
}
pub unsafe fn shift_15_boards_rev(a: __m256i) -> __m256i {
    let a_per = _mm256_permute4x64_epi64::<0xF9>(a);
    return _mm256_and_si256(
        *SHIFT_15_REV_MASK,
        _mm256_or_si256(_mm256_srli_epi64::<15>(a), _mm256_slli_epi64::<49>(a_per)),
    );
}

pub unsafe fn shift_17_boards(a: __m256i) -> __m256i {
    let a_per = _mm256_permute4x64_epi64::<0x90>(a);
    return _mm256_and_si256(
        *SHIFT_17_MASK,
        _mm256_or_si256(_mm256_slli_epi64::<17>(a), _mm256_srli_epi64::<47>(a_per)),
    );
}

pub unsafe fn shift_17_boards_rev(a: __m256i) -> __m256i {
    let a_per = _mm256_permute4x64_epi64::<0xF9>(a);
    return _mm256_and_si256(
        *SHIFT_17_REV_MASK,
        _mm256_or_si256(_mm256_srli_epi64::<17>(a), _mm256_slli_epi64::<47>(a_per)),
    );
}

pub unsafe fn not(a: __m256i) -> __m256i {
    return _mm256_xor_si256(a, *ONES);
}

pub unsafe fn done(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
    e: __m256i,
    f: __m256i,
    g: __m256i,
) -> __m256i {
    return _mm256_and_si256(
        not(g),
        _mm256_and_si256(
            not(a),
            _mm256_and_si256(
                b,
                _mm256_and_si256(c, _mm256_and_si256(d, _mm256_and_si256(e, f))),
            ),
        ),
    );
}

pub unsafe fn is_done(a: __m256i) -> bool {
    let s1_a = shift_1_boards(a);
    let s2_a = shift_1_boards(s1_a);
    let s3_a = shift_1_boards(s2_a);
    let s4_a = shift_1_boards(s3_a);
    let s5_a = shift_1_boards(s4_a);
    let s6_a = shift_1_boards_rev(a);
    let done00 = done(s6_a, a, s1_a, s2_a, s3_a, s4_a, s5_a);
    let s1_a = shift_16_boards(a);
    let s2_a = shift_16_boards(s1_a);
    let s3_a = shift_16_boards(s2_a);
    let s4_a = shift_16_boards(s3_a);
    let s5_a = shift_16_boards(s4_a);
    let s6_a = shift_16_boards_rev(a);
    let done01 = done(s6_a, a, s1_a, s2_a, s3_a, s4_a, s5_a);
    let s1_a = shift_15_boards(a);
    let s2_a = shift_15_boards(s1_a);
    let s3_a = shift_15_boards(s2_a);
    let s4_a = shift_15_boards(s3_a);
    let s5_a = shift_15_boards(s4_a);
    let s6_a = shift_15_boards_rev(a);
    let done10 = done(s6_a, a, s1_a, s2_a, s3_a, s4_a, s5_a);
    let s1_a = shift_17_boards(a);
    let s2_a = shift_17_boards(s1_a);
    let s3_a = shift_17_boards(s2_a);
    let s4_a = shift_17_boards(s3_a);
    let s5_a = shift_17_boards(s4_a);
    let s6_a = shift_17_boards_rev(a);
    let done11 = done(s6_a, a, s1_a, s2_a, s3_a, s4_a, s5_a);
    let done = _mm256_or_si256(
        _mm256_or_si256(done01, done00),
        _mm256_or_si256(done10, done11),
    );
    let done = _mm256_or_si256(done, _mm256_permute4x64_epi64(done, 0xb1));
    let done = _mm256_or_si256(done, _mm256_permute4x64_epi64(done, 0x0a));
    return _mm256_extract_epi64(done, 0) > 0;
}

pub unsafe fn to_int64s(a: __m256i) -> [i64; 4] {
    let mask = _mm256_set_epi64x(-1i64, -1i64, -1i64, -1i64);
    let mut out: [i64; 4] = [0, 0, 0, 0];
    _mm256_maskstore_epi64(&mut (out[0]) as *mut i64, mask, a);
    return out;
}

pub unsafe fn action_mask_from_bw(b: __m256i, w: __m256i) -> [i64; 4] {
    let mask = _mm256_set_epi64x(-1i64, -1i64, -1i64, -1i64);
    let mut out: [i64; 4] = [0, 0, 0, 0];
    let action_mask = _mm256_xor_si256(mask, _mm256_or_si256(b, w));
    _mm256_maskstore_epi64(&mut (out[0]) as *mut i64, mask, action_mask);
    return out;
}

pub unsafe fn to_uint8(a: __m256i) -> [u8; 32] {
    let arr_i64: [i64; 4] = to_int64s(a);
    let arr_u8: [u8; 32] = [
        (arr_i64[0] & 0xFFi64) as u8,
        ((arr_i64[0] >> 8) & 0xFF) as u8,
        ((arr_i64[0] >> 16) & 0xFF) as u8,
        ((arr_i64[0] >> 24) & 0xFF) as u8,
        ((arr_i64[0] >> 32) & 0xFF) as u8,
        ((arr_i64[0] >> 40) & 0xFF) as u8,
        ((arr_i64[0] >> 48) & 0xFF) as u8,
        ((arr_i64[0] >> 56) & 0xFF) as u8,
        (arr_i64[1] & 0xFFi64) as u8,
        ((arr_i64[1] >> 8) & 0xFF) as u8,
        ((arr_i64[1] >> 16) & 0xFF) as u8,
        ((arr_i64[1] >> 24) & 0xFF) as u8,
        ((arr_i64[1] >> 32) & 0xFF) as u8,
        ((arr_i64[1] >> 40) & 0xFF) as u8,
        ((arr_i64[1] >> 48) & 0xFF) as u8,
        ((arr_i64[1] >> 56) & 0xFF) as u8,
        (arr_i64[2] & 0xFFi64) as u8,
        ((arr_i64[2] >> 8) & 0xFF) as u8,
        ((arr_i64[2] >> 16) & 0xFF) as u8,
        ((arr_i64[2] >> 24) & 0xFF) as u8,
        ((arr_i64[2] >> 32) & 0xFF) as u8,
        ((arr_i64[2] >> 40) & 0xFF) as u8,
        ((arr_i64[2] >> 48) & 0xFF) as u8,
        ((arr_i64[2] >> 56) & 0xFF) as u8,
        (arr_i64[3] & 0xFFi64) as u8,
        ((arr_i64[3] >> 8) & 0xFF) as u8,
        ((arr_i64[3] >> 16) & 0xFF) as u8,
        ((arr_i64[3] >> 24) & 0xFF) as u8,
        ((arr_i64[3] >> 32) & 0xFF) as u8,
        ((arr_i64[3] >> 40) & 0xFF) as u8,
        ((arr_i64[3] >> 48) & 0xFF) as u8,
        ((arr_i64[3] >> 56) & 0xFF) as u8,
    ];
    return arr_u8;
}

pub unsafe fn unpackbits(a: __m256i) -> [u8; 225] {
    let arr_i64: [i64; 4] = to_int64s(a);
    let arr_u8 = [
        ((arr_i64[0] >> 0) & 1) as u8,
        ((arr_i64[0] >> 1) & 1) as u8,
        ((arr_i64[0] >> 2) & 1) as u8,
        ((arr_i64[0] >> 3) & 1) as u8,
        ((arr_i64[0] >> 4) & 1) as u8,
        ((arr_i64[0] >> 5) & 1) as u8,
        ((arr_i64[0] >> 6) & 1) as u8,
        ((arr_i64[0] >> 7) & 1) as u8,
        ((arr_i64[0] >> 8) & 1) as u8,
        ((arr_i64[0] >> 9) & 1) as u8,
        ((arr_i64[0] >> 10) & 1) as u8,
        ((arr_i64[0] >> 11) & 1) as u8,
        ((arr_i64[0] >> 12) & 1) as u8,
        ((arr_i64[0] >> 13) & 1) as u8,
        ((arr_i64[0] >> 14) & 1) as u8,
        ((arr_i64[0] >> 16) & 1) as u8,
        ((arr_i64[0] >> 17) & 1) as u8,
        ((arr_i64[0] >> 18) & 1) as u8,
        ((arr_i64[0] >> 19) & 1) as u8,
        ((arr_i64[0] >> 20) & 1) as u8,
        ((arr_i64[0] >> 21) & 1) as u8,
        ((arr_i64[0] >> 22) & 1) as u8,
        ((arr_i64[0] >> 23) & 1) as u8,
        ((arr_i64[0] >> 24) & 1) as u8,
        ((arr_i64[0] >> 25) & 1) as u8,
        ((arr_i64[0] >> 26) & 1) as u8,
        ((arr_i64[0] >> 27) & 1) as u8,
        ((arr_i64[0] >> 28) & 1) as u8,
        ((arr_i64[0] >> 29) & 1) as u8,
        ((arr_i64[0] >> 30) & 1) as u8,
        ((arr_i64[0] >> 32) & 1) as u8,
        ((arr_i64[0] >> 33) & 1) as u8,
        ((arr_i64[0] >> 34) & 1) as u8,
        ((arr_i64[0] >> 35) & 1) as u8,
        ((arr_i64[0] >> 36) & 1) as u8,
        ((arr_i64[0] >> 37) & 1) as u8,
        ((arr_i64[0] >> 38) & 1) as u8,
        ((arr_i64[0] >> 39) & 1) as u8,
        ((arr_i64[0] >> 40) & 1) as u8,
        ((arr_i64[0] >> 41) & 1) as u8,
        ((arr_i64[0] >> 42) & 1) as u8,
        ((arr_i64[0] >> 43) & 1) as u8,
        ((arr_i64[0] >> 44) & 1) as u8,
        ((arr_i64[0] >> 45) & 1) as u8,
        ((arr_i64[0] >> 46) & 1) as u8,
        ((arr_i64[0] >> 48) & 1) as u8,
        ((arr_i64[0] >> 49) & 1) as u8,
        ((arr_i64[0] >> 50) & 1) as u8,
        ((arr_i64[0] >> 51) & 1) as u8,
        ((arr_i64[0] >> 52) & 1) as u8,
        ((arr_i64[0] >> 53) & 1) as u8,
        ((arr_i64[0] >> 54) & 1) as u8,
        ((arr_i64[0] >> 55) & 1) as u8,
        ((arr_i64[0] >> 56) & 1) as u8,
        ((arr_i64[0] >> 57) & 1) as u8,
        ((arr_i64[0] >> 58) & 1) as u8,
        ((arr_i64[0] >> 59) & 1) as u8,
        ((arr_i64[0] >> 60) & 1) as u8,
        ((arr_i64[0] >> 61) & 1) as u8,
        ((arr_i64[0] >> 62) & 1) as u8,
        ((arr_i64[1] >> 0) & 1) as u8,
        ((arr_i64[1] >> 1) & 1) as u8,
        ((arr_i64[1] >> 2) & 1) as u8,
        ((arr_i64[1] >> 3) & 1) as u8,
        ((arr_i64[1] >> 4) & 1) as u8,
        ((arr_i64[1] >> 5) & 1) as u8,
        ((arr_i64[1] >> 6) & 1) as u8,
        ((arr_i64[1] >> 7) & 1) as u8,
        ((arr_i64[1] >> 8) & 1) as u8,
        ((arr_i64[1] >> 9) & 1) as u8,
        ((arr_i64[1] >> 10) & 1) as u8,
        ((arr_i64[1] >> 11) & 1) as u8,
        ((arr_i64[1] >> 12) & 1) as u8,
        ((arr_i64[1] >> 13) & 1) as u8,
        ((arr_i64[1] >> 14) & 1) as u8,
        ((arr_i64[1] >> 16) & 1) as u8,
        ((arr_i64[1] >> 17) & 1) as u8,
        ((arr_i64[1] >> 18) & 1) as u8,
        ((arr_i64[1] >> 19) & 1) as u8,
        ((arr_i64[1] >> 20) & 1) as u8,
        ((arr_i64[1] >> 21) & 1) as u8,
        ((arr_i64[1] >> 22) & 1) as u8,
        ((arr_i64[1] >> 23) & 1) as u8,
        ((arr_i64[1] >> 24) & 1) as u8,
        ((arr_i64[1] >> 25) & 1) as u8,
        ((arr_i64[1] >> 26) & 1) as u8,
        ((arr_i64[1] >> 27) & 1) as u8,
        ((arr_i64[1] >> 28) & 1) as u8,
        ((arr_i64[1] >> 29) & 1) as u8,
        ((arr_i64[1] >> 30) & 1) as u8,
        ((arr_i64[1] >> 32) & 1) as u8,
        ((arr_i64[1] >> 33) & 1) as u8,
        ((arr_i64[1] >> 34) & 1) as u8,
        ((arr_i64[1] >> 35) & 1) as u8,
        ((arr_i64[1] >> 36) & 1) as u8,
        ((arr_i64[1] >> 37) & 1) as u8,
        ((arr_i64[1] >> 38) & 1) as u8,
        ((arr_i64[1] >> 39) & 1) as u8,
        ((arr_i64[1] >> 40) & 1) as u8,
        ((arr_i64[1] >> 41) & 1) as u8,
        ((arr_i64[1] >> 42) & 1) as u8,
        ((arr_i64[1] >> 43) & 1) as u8,
        ((arr_i64[1] >> 44) & 1) as u8,
        ((arr_i64[1] >> 45) & 1) as u8,
        ((arr_i64[1] >> 46) & 1) as u8,
        ((arr_i64[1] >> 48) & 1) as u8,
        ((arr_i64[1] >> 49) & 1) as u8,
        ((arr_i64[1] >> 50) & 1) as u8,
        ((arr_i64[1] >> 51) & 1) as u8,
        ((arr_i64[1] >> 52) & 1) as u8,
        ((arr_i64[1] >> 53) & 1) as u8,
        ((arr_i64[1] >> 54) & 1) as u8,
        ((arr_i64[1] >> 55) & 1) as u8,
        ((arr_i64[1] >> 56) & 1) as u8,
        ((arr_i64[1] >> 57) & 1) as u8,
        ((arr_i64[1] >> 58) & 1) as u8,
        ((arr_i64[1] >> 59) & 1) as u8,
        ((arr_i64[1] >> 60) & 1) as u8,
        ((arr_i64[1] >> 61) & 1) as u8,
        ((arr_i64[1] >> 62) & 1) as u8,
        ((arr_i64[2] >> 0) & 1) as u8,
        ((arr_i64[2] >> 1) & 1) as u8,
        ((arr_i64[2] >> 2) & 1) as u8,
        ((arr_i64[2] >> 3) & 1) as u8,
        ((arr_i64[2] >> 4) & 1) as u8,
        ((arr_i64[2] >> 5) & 1) as u8,
        ((arr_i64[2] >> 6) & 1) as u8,
        ((arr_i64[2] >> 7) & 1) as u8,
        ((arr_i64[2] >> 8) & 1) as u8,
        ((arr_i64[2] >> 9) & 1) as u8,
        ((arr_i64[2] >> 10) & 1) as u8,
        ((arr_i64[2] >> 11) & 1) as u8,
        ((arr_i64[2] >> 12) & 1) as u8,
        ((arr_i64[2] >> 13) & 1) as u8,
        ((arr_i64[2] >> 14) & 1) as u8,
        ((arr_i64[2] >> 16) & 1) as u8,
        ((arr_i64[2] >> 17) & 1) as u8,
        ((arr_i64[2] >> 18) & 1) as u8,
        ((arr_i64[2] >> 19) & 1) as u8,
        ((arr_i64[2] >> 20) & 1) as u8,
        ((arr_i64[2] >> 21) & 1) as u8,
        ((arr_i64[2] >> 22) & 1) as u8,
        ((arr_i64[2] >> 23) & 1) as u8,
        ((arr_i64[2] >> 24) & 1) as u8,
        ((arr_i64[2] >> 25) & 1) as u8,
        ((arr_i64[2] >> 26) & 1) as u8,
        ((arr_i64[2] >> 27) & 1) as u8,
        ((arr_i64[2] >> 28) & 1) as u8,
        ((arr_i64[2] >> 29) & 1) as u8,
        ((arr_i64[2] >> 30) & 1) as u8,
        ((arr_i64[2] >> 32) & 1) as u8,
        ((arr_i64[2] >> 33) & 1) as u8,
        ((arr_i64[2] >> 34) & 1) as u8,
        ((arr_i64[2] >> 35) & 1) as u8,
        ((arr_i64[2] >> 36) & 1) as u8,
        ((arr_i64[2] >> 37) & 1) as u8,
        ((arr_i64[2] >> 38) & 1) as u8,
        ((arr_i64[2] >> 39) & 1) as u8,
        ((arr_i64[2] >> 40) & 1) as u8,
        ((arr_i64[2] >> 41) & 1) as u8,
        ((arr_i64[2] >> 42) & 1) as u8,
        ((arr_i64[2] >> 43) & 1) as u8,
        ((arr_i64[2] >> 44) & 1) as u8,
        ((arr_i64[2] >> 45) & 1) as u8,
        ((arr_i64[2] >> 46) & 1) as u8,
        ((arr_i64[2] >> 48) & 1) as u8,
        ((arr_i64[2] >> 49) & 1) as u8,
        ((arr_i64[2] >> 50) & 1) as u8,
        ((arr_i64[2] >> 51) & 1) as u8,
        ((arr_i64[2] >> 52) & 1) as u8,
        ((arr_i64[2] >> 53) & 1) as u8,
        ((arr_i64[2] >> 54) & 1) as u8,
        ((arr_i64[2] >> 55) & 1) as u8,
        ((arr_i64[2] >> 56) & 1) as u8,
        ((arr_i64[2] >> 57) & 1) as u8,
        ((arr_i64[2] >> 58) & 1) as u8,
        ((arr_i64[2] >> 59) & 1) as u8,
        ((arr_i64[2] >> 60) & 1) as u8,
        ((arr_i64[2] >> 61) & 1) as u8,
        ((arr_i64[2] >> 62) & 1) as u8,
        ((arr_i64[3] >> 0) & 1) as u8,
        ((arr_i64[3] >> 1) & 1) as u8,
        ((arr_i64[3] >> 2) & 1) as u8,
        ((arr_i64[3] >> 3) & 1) as u8,
        ((arr_i64[3] >> 4) & 1) as u8,
        ((arr_i64[3] >> 5) & 1) as u8,
        ((arr_i64[3] >> 6) & 1) as u8,
        ((arr_i64[3] >> 7) & 1) as u8,
        ((arr_i64[3] >> 8) & 1) as u8,
        ((arr_i64[3] >> 9) & 1) as u8,
        ((arr_i64[3] >> 10) & 1) as u8,
        ((arr_i64[3] >> 11) & 1) as u8,
        ((arr_i64[3] >> 12) & 1) as u8,
        ((arr_i64[3] >> 13) & 1) as u8,
        ((arr_i64[3] >> 14) & 1) as u8,
        ((arr_i64[3] >> 16) & 1) as u8,
        ((arr_i64[3] >> 17) & 1) as u8,
        ((arr_i64[3] >> 18) & 1) as u8,
        ((arr_i64[3] >> 19) & 1) as u8,
        ((arr_i64[3] >> 20) & 1) as u8,
        ((arr_i64[3] >> 21) & 1) as u8,
        ((arr_i64[3] >> 22) & 1) as u8,
        ((arr_i64[3] >> 23) & 1) as u8,
        ((arr_i64[3] >> 24) & 1) as u8,
        ((arr_i64[3] >> 25) & 1) as u8,
        ((arr_i64[3] >> 26) & 1) as u8,
        ((arr_i64[3] >> 27) & 1) as u8,
        ((arr_i64[3] >> 28) & 1) as u8,
        ((arr_i64[3] >> 29) & 1) as u8,
        ((arr_i64[3] >> 30) & 1) as u8,
        ((arr_i64[3] >> 32) & 1) as u8,
        ((arr_i64[3] >> 33) & 1) as u8,
        ((arr_i64[3] >> 34) & 1) as u8,
        ((arr_i64[3] >> 35) & 1) as u8,
        ((arr_i64[3] >> 36) & 1) as u8,
        ((arr_i64[3] >> 37) & 1) as u8,
        ((arr_i64[3] >> 38) & 1) as u8,
        ((arr_i64[3] >> 39) & 1) as u8,
        ((arr_i64[3] >> 40) & 1) as u8,
        ((arr_i64[3] >> 41) & 1) as u8,
        ((arr_i64[3] >> 42) & 1) as u8,
        ((arr_i64[3] >> 43) & 1) as u8,
        ((arr_i64[3] >> 44) & 1) as u8,
        ((arr_i64[3] >> 45) & 1) as u8,
        ((arr_i64[3] >> 46) & 1) as u8,
    ];
    return arr_u8;
}

unsafe fn print_vec(a: __m256i) -> () {
    let a = to_int64s(a);
    let mut s = String::new();
    for i in 0..4 {
        let a = a[i];
        for j in 0..64 {
            if (a >> j) & 1 == 1 {
                s.push('O');
            } else {
                s.push('-');
            }
            if j % 16 == 15 {
                s.push('\n');
            }
        }
    }
    println!("{}", s);
}
