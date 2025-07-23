use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};
use unicode_segmentation::{UWordBounds, UnicodeSegmentation};

use text_parsing::{Local, Localize, Snip};

use crate::{
    TokenizerOptions,
    numbers::{NumberChecker, NumberCounter},
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum BasicToken<'t> {
    Alphanumeric(&'t str),
    Number(&'t str, NumberChecker<'t>),
    Punctuation(char),
    CurrencySymbol(char),
    Separator(char),
    Formatter(char),
    Mixed(&'t str),
}

enum Extra<'s> {
    None,
    Number(NumberChecker<'s>),
}

//#[derive(Debug)]
struct ExtWordBounds<'t> {
    offset: usize,
    char_offset: usize,
    initial: &'t str,
    bounds: UWordBounds<'t>,
    buffer: VecDeque<(Local<&'t str>, Extra<'t>)>,
    ext_spliters: BTreeSet<char>,
    allow_complex: bool,
    split_dot: bool,
    split_underscore: bool,
    split_colon: bool,
    split_semicolon: bool,
    number_unknown_coma_as_dot: bool,
    num_counter: Arc<NumberCounter>,
}
impl<'t> ExtWordBounds<'t> {
    fn new<'a>(
        s: &'a str,
        options: &BTreeSet<TokenizerOptions>,
        num_counter: Arc<NumberCounter>,
    ) -> ExtWordBounds<'a> {
        ExtWordBounds {
            offset: 0,
            char_offset: 0,
            initial: s,
            bounds: s.split_word_bounds(),
            buffer: VecDeque::new(),
            //exceptions: ['\u{200d}'].iter().cloned().collect(),
            ext_spliters: ['\u{200c}'].iter().cloned().collect(),
            allow_complex: !options.contains(&TokenizerOptions::NoComplexTokens),
            split_dot: options.contains(&TokenizerOptions::SplitDot),
            split_underscore: options.contains(&TokenizerOptions::SplitUnderscore),
            split_colon: options.contains(&TokenizerOptions::SplitColon),
            split_semicolon: options.contains(&TokenizerOptions::SplitSemiColon),
            number_unknown_coma_as_dot: options.contains(&TokenizerOptions::NumberUnknownComaAsDot),
            num_counter,
        }
    }
}
#[rustfmt::skip]
impl<'t> Iterator for ExtWordBounds<'t> {
    type Item = (Local<&'t str>, Extra<'t>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.len() > 0 {
            return self.buffer.pop_front();
        }
        match self.bounds.next() {
            None => None,
            Some(w) => {
                let mut len = 0;
                let mut char_len = 0;
                let mut chs = w.chars().peekable();
                let num = NumberChecker::new(w,self.number_unknown_coma_as_dot,self.num_counter.stat());
                if let Some(num) = &num {
                    self.num_counter.push(num);
                }
                let mut first = true;
                let mut split = false;
                while let Some(c) = chs.next() {
                    let ln = c.len_utf8();
                    let c_is_whitespace = c.is_whitespace();
                    let c_is_spliter = self.ext_spliters.contains(&c);
                    let c_is_punctuation =
                        c.general_category_group() == GeneralCategoryGroup::Punctuation;
                    if first && (c_is_whitespace || c_is_punctuation) && chs.peek().is_some() && num.is_none() {
                        let mut same = true;
                        for c2 in w.chars() {
                            same = same && (c2 == c);
                            if !same {
                                break;
                            }
                        }
                        split = same;
                        first = false;
                    };
                    if c_is_spliter //( c_is_other_format && !exceptions_contain_c )
                        || split
                        //|| ( c_is_whitespace && !self.merge_whites )
                        || ( (c == '\u{200d}') && chs.peek().is_none() )
                        || ( c_is_punctuation && num.is_none() && !self.allow_complex ) // && !exceptions_contain_c 
                        || ( (c == '.') && num.is_none() && self.split_dot )
                        || ( (c == '_') && num.is_none() && self.split_underscore )
                        || ( (c == ':') && self.split_colon )
                        || ( (c == ';') && self.split_semicolon )
                    {
                        if len > 0 {
                            let local = ().localize(
                                Snip { offset: self.char_offset, length: char_len },
                                Snip { offset: self.offset, length: len },
                            );
                            self.buffer.push_back((
                                local.local(&self.initial[self.offset..self.offset + len]),
                                Extra::None
                            ));
                            self.offset += len;
                            self.char_offset += char_len;
                            len = 0;
                            char_len = 0;
                        }
                        let local = ().localize(
                            Snip { offset: self.char_offset, length: 1 },
                            Snip { offset: self.offset, length: ln },
                        );                        
                        self.buffer
                            .push_back((local.local(&self.initial[self.offset..self.offset + ln]), Extra::None));
                        self.offset += ln;
                        self.char_offset += 1;
                    } else {
                        len += ln;
                        char_len += 1;
                    }
                }
                if len > 0 {
                    let local = ().localize(
                        Snip { offset: self.char_offset, length: char_len },
                        Snip { offset: self.offset, length: len },
                    );
                    let extra = match num {
                        None => Extra::None,
                        Some(num) => Extra::Number(num),
                    };
                    self.buffer
                        .push_back((local.local(&self.initial[self.offset..self.offset + len]),extra));
                    self.offset += len;
                    self.char_offset += char_len;
                }
                self.next()
            }
        }
    }
}

pub(crate) fn one_char_word(w: &str) -> Option<char> {
    // returns Some(char) if len in char == 1, None otherwise
    let mut cs = w.chars();
    match (cs.next(), cs.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

//#[derive(Debug)]
pub(crate) struct WordBreaker<'t> {
    initial: &'t str,
    prev_is_separator: bool,
    merge_whites: bool,
    merge_punct: bool,
    split_number_sign: bool,
    number_unknown_coma_as_dot: bool,
    bounds: std::iter::Peekable<ExtWordBounds<'t>>,
    num_counter: Arc<NumberCounter>,
}
impl<'t> WordBreaker<'t> {
    pub(crate) fn new<'a>(s: &'a str, options: &BTreeSet<TokenizerOptions>) -> WordBreaker<'a> {
        let num_counter = Arc::new(NumberCounter::new());
        WordBreaker {
            initial: s,
            prev_is_separator: true,
            merge_whites: options.contains(&TokenizerOptions::MergeWhites),
            merge_punct: options.contains(&TokenizerOptions::MergePunctuation),
            split_number_sign: options.contains(&TokenizerOptions::SplitNumberSign),
            number_unknown_coma_as_dot: options.contains(&TokenizerOptions::NumberUnknownComaAsDot),
            bounds: ExtWordBounds::new(s, options, num_counter.clone()).peekable(),
            num_counter,
        }
    }
    fn next_token(&mut self) -> Option<Local<BasicToken<'t>>> {
        // is_ascii_punctuation makes '$' a punctuation

        match self.bounds.next() {
            Some((w, extra)) => {
                let (local, w) = w.into_inner();
                if let Some(c) = one_char_word(w) {
                    if ((c == '+') || (c == '-'))
                        && self.prev_is_separator
                        && !self.split_number_sign
                    {
                        if let Some((w2, extra2)) = self.bounds.peek() {
                            let (loc2, _) = w2.into_inner();
                            if let Extra::Number(mut num) = *extra2 {
                                if num.push_sign(c) {
                                    if let Ok(local) = Local::from_segment(local, loc2) {
                                        self.bounds.next();
                                        let Snip {
                                            offset: off,
                                            length: len,
                                        } = local.bytes();
                                        let p = &self.initial[off..off + len];
                                        return Some(local.local(BasicToken::Number(p, num)));
                                    }
                                }
                            }
                        }
                    }
                    if c.is_ascii_punctuation()
                        || c.is_whitespace()
                        || (c.general_category_group() == GeneralCategoryGroup::Punctuation)
                        || (c.general_category() == GeneralCategory::CurrencySymbol)
                        || (c.general_category() == GeneralCategory::Format)
                    {
                        let mut local = local;
                        if (c.is_whitespace() && self.merge_whites)
                            || (c.is_ascii_punctuation() && self.merge_punct)
                            || ((c.general_category_group() == GeneralCategoryGroup::Punctuation)
                                && self.merge_punct)
                            || (c.general_category() == GeneralCategory::Format)
                        {
                            loop {
                                match self.bounds.peek() {
                                    Some((p, _extra)) if *p.data() == w => {
                                        let (loc2, _) = p.into_inner();
                                        match Local::from_segment(local, loc2) {
                                            Ok(new_loc) => local = new_loc,
                                            Err(_) => break,
                                        }
                                    }
                                    _ => break,
                                }
                                self.bounds.next();
                            }
                        }

                        if c.general_category() == GeneralCategory::CurrencySymbol {
                            // must be before 'is_ascii_punctuation' because of '$' sign
                            return Some(local.local(BasicToken::CurrencySymbol(c)));
                        }
                        if c.is_ascii_punctuation()
                            || (c.general_category_group() == GeneralCategoryGroup::Punctuation)
                        {
                            return Some(local.local(BasicToken::Punctuation(c)));
                        }
                        if c.general_category() == GeneralCategory::Format {
                            return Some(local.local(BasicToken::Formatter(c)));
                        } else {
                            return Some(local.local(BasicToken::Separator(c)));
                        }
                    }
                } // if c is one_char_word

                if let Extra::Number(num) = extra {
                    return Some(local.local(BasicToken::Number(w, num)));
                }

                if let Some(num) =
                    NumberChecker::new(w, self.number_unknown_coma_as_dot, self.num_counter.stat())
                {
                    self.num_counter.push(&num);
                    return Some(local.local(BasicToken::Number(w, num)));
                }

                let mut an = true;
                for c in w.chars() {
                    an = an
                        && (c.is_alphanumeric()
                            || (c == '.')
                            || (c == '\'')
                            || (c == '-')
                            || (c == '+')
                            || (c == '_'));
                }
                if an {
                    return Some(local.local(BasicToken::Alphanumeric(w)));
                }
                Some(local.local(BasicToken::Mixed(w)))
            }
            None => None,
        }
    }
}
impl<'t> Iterator for WordBreaker<'t> {
    type Item = Local<BasicToken<'t>>;
    fn next(&mut self) -> Option<Self::Item> {
        let tok = self.next_token();
        self.prev_is_separator = match &tok {
            Some(tok) => match tok.data() {
                BasicToken::Separator(..) => true,
                _ => false,
            },
            _ => false,
        };
        tok
    }
}
