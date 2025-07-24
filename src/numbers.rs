use std::{
    str::FromStr,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::Number;

pub struct NumberCounter {
    // notation counter
    ru: AtomicUsize,
    en: AtomicUsize,
}
impl NumberCounter {
    pub fn new() -> NumberCounter {
        NumberCounter {
            ru: AtomicUsize::new(0),
            en: AtomicUsize::new(0),
        }
    }
    pub fn push(&self, num: &NumberChecker) {
        match &num.coma_prop {
            None => {}
            Some(Coma::Thousand) => {
                self.en.fetch_add(1, Ordering::Relaxed);
            }
            Some(Coma::Fraction) => {
                self.ru.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    pub fn stat(&self) -> Option<Coma> {
        match (
            self.en.load(Ordering::Relaxed),
            self.ru.load(Ordering::Relaxed),
        ) {
            (0, 0) => None,
            (_, 0) => Some(Coma::Thousand),
            (0, _) => Some(Coma::Fraction),
            (_, _) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum NumberCheckerInner<'s> {
    SimpleInt(i64),
    HugeInt(f64),
    SimpleFloat(f64),
    OverflowInt(&'s str),
    OverflowFloat(&'s str),
}
impl<'s> NumberCheckerInner<'s> {
    fn negative(&mut self) -> NumberCheckerInner<'s> {
        match self {
            NumberCheckerInner::SimpleInt(n) => NumberCheckerInner::SimpleInt(-*n),
            NumberCheckerInner::HugeInt(n) => NumberCheckerInner::HugeInt(-*n),
            NumberCheckerInner::SimpleFloat(n) => NumberCheckerInner::SimpleFloat(-*n),
            NumberCheckerInner::OverflowInt(s) => NumberCheckerInner::OverflowInt(s),
            NumberCheckerInner::OverflowFloat(s) => NumberCheckerInner::OverflowFloat(s),
        }
    }
    fn check_eps(&mut self) {
        match self {
            NumberCheckerInner::SimpleFloat(n) => {
                let toi = n.round();
                if (*n - toi).abs() < crate::EPS {
                    if ((i64::MIN as f64) < toi) && (toi < i64::MAX as f64) {
                        *self = NumberCheckerInner::SimpleInt(toi as i64);
                    }
                }
            }
            NumberCheckerInner::SimpleInt(_)
            | NumberCheckerInner::HugeInt(_)
            | NumberCheckerInner::OverflowInt(_)
            | NumberCheckerInner::OverflowFloat(_) => {}
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum Sign {
    Plus,
    Minus,
}
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Coma {
    Thousand,
    Fraction,
}
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct NumberChecker<'s> {
    pub src: &'s str,
    zero: bool,
    sign: Option<Sign>,
    subtype: NumberCheckerInner<'s>,
    coma_prop: Option<Coma>,
    pushed_sign: bool, // can be processed on output only
}
impl<'s> NumberChecker<'s> {
    pub fn new(
        src: &str,
        unknown_coma_as_dot: bool,
        unknown_by_stat: Option<Coma>,
    ) -> Option<NumberChecker> {
        let mut coma_prop = None;
        let (zero, sign) = match src.chars().next() {
            Some('0') => (true, None),
            Some('-') => (false, Some(Sign::Minus)),
            Some('+') => (false, Some(Sign::Plus)),
            _ => (false, None),
        };
        let mut subtype = match i64::from_str(src) {
            Ok(i) => NumberCheckerInner::SimpleInt(i),
            Err(_) => match f64::from_str(src) {
                Ok(f) => {
                    coma_prop = Some(Coma::Thousand);
                    NumberCheckerInner::SimpleFloat(f)
                }
                Err(_) => {
                    // checking only coma thousand-split + and dot
                    // russian notation: coma instead of dot
                    let mut coma_count = 0;
                    let mut dot_count = 0;
                    let mut digits = 0;
                    let mut first_digit_group = 0;

                    let s = match sign.is_some() {
                        true => &src[1..],
                        false => src,
                    };
                    for c in s.chars() {
                        match c {
                            _ if c.is_digit(10) => digits += 1,
                            ',' => {
                                if dot_count > 0 {
                                    return None;
                                }
                                if coma_count == 0 {
                                    first_digit_group = digits;
                                } else {
                                    if digits != 3 {
                                        // non 3-digit middle group
                                        return None;
                                    }
                                }
                                coma_count += 1;
                                digits = 0;
                            }
                            '.' => {
                                if dot_count > 0 {
                                    return None;
                                }
                                if coma_count == 0 {
                                    first_digit_group = digits;
                                } else {
                                    if digits != 3 {
                                        // non 3-digit middle group
                                        return None;
                                    }
                                }
                                dot_count += 1;
                                digits = 0;
                            }
                            _ => return None,
                        }
                    }
                    let last_digit_group = digits;
                    if (first_digit_group == 0) || (last_digit_group == 0) {
                        return None;
                    }

                    // number has only comas, digits and maybe dot
                    // coma or dot not first and not last
                    // all middle (between comas/dot) digit groups are of length 3
                    let mut number_without_sign = match (coma_count, dot_count) {
                        (0, 0) => {
                            // simple int ?, no comas or dot
                            match i64::from_str(s) {
                                Ok(i) => NumberCheckerInner::SimpleInt(i),
                                Err(_) => match f64::from_str(s) {
                                    Ok(f) => NumberCheckerInner::HugeInt(f),
                                    Err(_) => NumberCheckerInner::OverflowInt(src),
                                },
                            }
                        }
                        (1, 0) => {
                            // number with 1 coma only
                            match (first_digit_group, last_digit_group) {
                                (1, 3) | (2, 3) | (3, 3) => {
                                    // unknown
                                    let en_notation = match unknown_coma_as_dot {
                                        true => false,
                                        false => match unknown_by_stat {
                                            Some(Coma::Fraction) => false,
                                            Some(Coma::Thousand) => true,
                                            None => true, // by default en notation
                                        },
                                    };
                                    match en_notation {
                                        false => {
                                            // russian notation
                                            let s = s.replace(',', ".");
                                            match f64::from_str(&s) {
                                                Ok(f) => NumberCheckerInner::SimpleFloat(f),
                                                Err(_) => NumberCheckerInner::OverflowFloat(src),
                                            }
                                        }
                                        true => {
                                            // english notation
                                            let s = s.replace(',', "");
                                            match i64::from_str(&s) {
                                                Ok(i) => NumberCheckerInner::SimpleInt(i),
                                                Err(_) => match f64::from_str(&s) {
                                                    Ok(f) => NumberCheckerInner::HugeInt(f),
                                                    Err(_) => NumberCheckerInner::OverflowInt(src),
                                                },
                                            }
                                        }
                                    }
                                }
                                (_, _) => {
                                    // russian notation coma = dot
                                    coma_prop = Some(Coma::Fraction);
                                    let s = s.replace(',', ".");
                                    match f64::from_str(&s) {
                                        Ok(f) => NumberCheckerInner::SimpleFloat(f),
                                        Err(_) => NumberCheckerInner::OverflowFloat(src),
                                    }
                                }
                            }
                        }
                        (_, 0) => {
                            // integer, coma is a thousand splitter
                            coma_prop = Some(Coma::Thousand);
                            let s = s.replace(',', "");
                            match i64::from_str(&s) {
                                Ok(i) => NumberCheckerInner::SimpleInt(i),
                                Err(_) => match f64::from_str(&s) {
                                    Ok(f) => NumberCheckerInner::HugeInt(f),
                                    Err(_) => NumberCheckerInner::OverflowInt(src),
                                },
                            }
                        }
                        (0, 1) => {
                            // simple float ?, no comas
                            coma_prop = Some(Coma::Thousand);
                            match f64::from_str(s) {
                                Ok(f) => NumberCheckerInner::SimpleFloat(f),
                                Err(_) => NumberCheckerInner::OverflowFloat(src),
                            }
                        }
                        (_, 1) => {
                            // float, coma is a thousand splitter
                            coma_prop = Some(Coma::Thousand);
                            let s = s.replace(',', "");
                            match f64::from_str(&s) {
                                Ok(f) => NumberCheckerInner::SimpleFloat(f),
                                Err(_) => NumberCheckerInner::OverflowFloat(src),
                            }
                        }
                        (_, _) => {
                            // many dots
                            return None;
                        }
                    };

                    match sign {
                        Some(Sign::Minus) => number_without_sign.negative(),
                        Some(Sign::Plus) | None => number_without_sign,
                    }
                }
            },
        };
        subtype.check_eps();
        Some(NumberChecker {
            src,
            zero,
            sign,
            subtype,
            coma_prop,
            pushed_sign: false,
        })
    }
    pub fn push_sign(&mut self, sign: char) -> bool {
        match (self.sign, sign) {
            (None, '+') => {
                self.sign = Some(Sign::Plus);
                self.pushed_sign = true;
                true
            }
            (None, '-') => {
                self.sign = Some(Sign::Minus);
                self.subtype = self.subtype.negative();
                self.pushed_sign = true;
                true
            }
            (_, _) => false,
        }
    }

    pub fn into_number(&self) -> Option<Number> {
        // process subtype, zero and pushed_sign

        #[cfg(not(feature = "strings"))]
        fn zero_integer(n: i64, _s: &str, _pushed_sign: Option<Sign>) -> Number {
            Number::ZeroInteger { i: n }
        }

        #[cfg(feature = "strings")]
        fn zero_integer(n: i64, s: &str, pushed_sign: Option<Sign>) -> Number {
            let mut s = s.to_string();
            match pushed_sign {
                None => {}
                Some(Sign::Plus) => s.insert(0, '+'),
                Some(Sign::Minus) => s.insert(0, '-'),
            }
            Number::ZeroInteger { i: n, s }
        }

        let pushed_sign = match self.pushed_sign {
            true => self.sign,
            false => None,
        };
        match self.subtype {
            NumberCheckerInner::SimpleInt(n) => Some(match self.zero {
                true => zero_integer(n, self.src, pushed_sign),
                false => Number::Integer(n),
            }),
            NumberCheckerInner::HugeInt(f) => Some(Number::Float(f)),
            NumberCheckerInner::SimpleFloat(f) => Some(Number::Float(f)),
            NumberCheckerInner::OverflowInt(_s) => None,
            NumberCheckerInner::OverflowFloat(_s) => None,
        }
    }
}
