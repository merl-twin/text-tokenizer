use unicode_segmentation::{UnicodeSegmentation,UWordBounds};
use unicode_properties::{
    GeneralCategoryGroup,
    UnicodeGeneralCategory,
    GeneralCategory,
};
use std::str::FromStr;
use std::collections::{VecDeque,BTreeSet};

use text_parsing::{
    Local, Snip,
    Localize,
};

use crate::{
    TokenizerOptions,
};

#[derive(Debug,Clone,Copy,PartialEq,PartialOrd,Eq,Ord)]
pub enum BasicToken<'t> {
    Alphanumeric(&'t str),
    Number(&'t str),
    Punctuation(char),
    Separator(char),
    Formatter(char),
    Mixed(&'t str),
}

//#[derive(Debug)]
struct ExtWordBounds<'t> {
    offset: usize,
    char_offset: usize,
    initial: &'t str,
    bounds: UWordBounds<'t>,
    buffer: VecDeque<Local<&'t str>>,
    ext_spliters: BTreeSet<char>,
    allow_complex: bool,
    split_dot: bool,
    split_underscore: bool,
    split_colon: bool,
}
impl<'t> ExtWordBounds<'t> {
    fn new<'a>(s: &'a str, options: &BTreeSet<TokenizerOptions>) -> ExtWordBounds<'a> {
        ExtWordBounds {
            offset: 0,
            char_offset: 0,
            initial: s,
            bounds: s.split_word_bounds(),
            buffer: VecDeque::new(),
            //exceptions: ['\u{200d}'].iter().cloned().collect(),
            ext_spliters: ['\u{200c}'].iter().cloned().collect(),
            allow_complex: if options.contains(&TokenizerOptions::NoComplexTokens) { false } else { true },
            split_dot: if options.contains(&TokenizerOptions::SplitDot) { true } else { false },
            split_underscore: if options.contains(&TokenizerOptions::SplitUnderscore) { true } else { false },
            split_colon: if options.contains(&TokenizerOptions::SplitColon) { true } else { false },
        }
    }
}
impl<'t> Iterator for ExtWordBounds<'t> {
    type Item = Local<&'t str>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.len() > 0 { return self.buffer.pop_front(); }
        match self.bounds.next() {
            None => None,
            Some(w) => {
                let mut len = 0;
                let mut char_len = 0;
                let mut chs = w.chars().peekable();
                let num = match f64::from_str(w) { Ok(_) => true, Err(_) => false };
                let mut first = true;
                let mut split = false;
                while let Some(c) = chs.next() {
                    let ln = c.len_utf8();
                    let c_is_whitespace = c.is_whitespace();
                    let c_is_spliter = self.ext_spliters.contains(&c);
                    let c_is_punctuation = c.general_category_group() == GeneralCategoryGroup::Punctuation;
                    if first && (c_is_whitespace || c_is_punctuation) && chs.peek().is_some() {
                        let mut same = true;
                        for c2 in w.chars() {
                            same = same && (c2 == c);
                            if !same { break; }
                        }
                        split = same;
                        first = false;
                    };
                    if  c_is_spliter //( c_is_other_format && !exceptions_contain_c )
                        || split
                        //|| ( c_is_whitespace && !self.merge_whites )
                        || ( (c == '\u{200d}') && chs.peek().is_none() ) 
                        || ( c_is_punctuation && !num && !self.allow_complex ) // && !exceptions_contain_c 
                        || ( (c == '.') && !num && self.split_dot )
                        || ( (c == '_') && !num && self.split_underscore )
                        || ( (c == ':') && self.split_colon )
                    {
                        if len > 0 {
                            let local = ().localize(Snip{ offset: self.char_offset, length: char_len },
                                                    Snip{ offset: self.offset, length: len });
                            self.buffer.push_back(local.local(&self.initial[self.offset .. self.offset+len]));
                            self.offset += len;
                            self.char_offset += char_len;
                            len = 0;
                            char_len = 0;
                        }                        
                        let local = ().localize(Snip{ offset: self.char_offset, length: 1 },
                                                Snip{ offset: self.offset, length: ln });
                        self.buffer.push_back(local.local(&self.initial[self.offset .. self.offset+ln]));
                        self.offset += ln;
                        self.char_offset += 1;                        
                    } else {
                        len += ln;
                        char_len += 1;
                    }
                }
                if len > 0 {
                    let local = ().localize(Snip{ offset: self.char_offset, length: char_len },
                                            Snip{ offset: self.offset, length: len });
                    self.buffer.push_back(local.local(&self.initial[self.offset .. self.offset+len]));
                    self.offset += len;
                    self.char_offset += char_len;
                }
                self.next()
            },
        }
    }
}

pub(crate) fn one_char_word(w: &str) -> Option<char> {
    // returns Some(char) if len in char == 1, None otherwise
    let mut cs = w.chars();
    match (cs.next(),cs.next()) {
        (Some(c),None) => Some(c),
        _ => None,
    }
}

//#[derive(Debug)]
pub(crate) struct WordBreaker<'t> {
    initial: &'t str,
    prev_is_separator: bool,
    merge_whites: bool,
    merge_punct: bool,
    bounds: std::iter::Peekable<ExtWordBounds<'t>>,
}
impl<'t> WordBreaker<'t> {
    pub(crate) fn new<'a>(s: &'a str, options: &BTreeSet<TokenizerOptions>) -> WordBreaker<'a> {
        WordBreaker {
            initial: s,
            prev_is_separator: true,
            merge_whites: if options.contains(&TokenizerOptions::MergeWhites) { true } else { false },
            merge_punct: if options.contains(&TokenizerOptions::MergePunctuation) { true } else { false },
            bounds: ExtWordBounds::new(s,options).peekable(),
        }
    }
    fn next_token(&mut self) -> Option<Local<BasicToken<'t>>> {
        match self.bounds.next() {
            Some(w) => {
                let (local,w) = w.into_inner();
                if let Some(c) = one_char_word(w) {
                    if ((c == '+')||(c == '-')) && self.prev_is_separator {
                        if let Some(w2) = self.bounds.peek() {
                            let (loc2,w2) = w2.into_inner();
                            let mut num = true;  
                            let mut dot_count = 0;
                            for c in w2.chars() {
                                num = num && (c.is_digit(10) || (c == '.'));
                                if c == '.' { dot_count += 1; }
                            }
                            if dot_count>1 { num = false; }

                            if num {
                                if let Ok(local) = Local::from_segment(local,loc2) {
                                    self.bounds.next();
                                    let Snip{ offset: off, length: len } = local.bytes();
                                    let p = &self.initial[off .. off+len];
                                    return Some(local.local(BasicToken::Number(p)));
                                }
                            }
                        }                                          
                    }
                    if c.is_ascii_punctuation() ||
                       c.is_whitespace() ||
                        (c.general_category_group() == GeneralCategoryGroup::Punctuation) ||                        
                        (c.general_category() == GeneralCategory::Format)
                    {
                        let mut local = local;
                        if (c.is_whitespace() && self.merge_whites) ||
                            (c.is_ascii_punctuation() && self.merge_punct ) ||
                            ((c.general_category_group() == GeneralCategoryGroup::Punctuation) && self.merge_punct) ||                        
                            (c.general_category() == GeneralCategory::Format)
                        {
                            loop {
                                match self.bounds.peek() {
                                    Some(p) if *p.data() == w => {
                                        let (loc2,_) = p.into_inner();
                                        match Local::from_segment(local,loc2) {
                                            Ok(new_loc) => local = new_loc,
                                            Err(_) => break,
                                        }
                                    },
                                    _ => break,
                                }
                                self.bounds.next();
                            }
                        }
                        
                        if c.is_ascii_punctuation() || (c.general_category_group() == GeneralCategoryGroup::Punctuation) {
                            return Some(local.local(BasicToken::Punctuation(c)));
                        }
                        if c.general_category() == GeneralCategory::Format {
                            return Some(local.local(BasicToken::Formatter(c)));
                        }
                        else {
                            return Some(local.local(BasicToken::Separator(c)));
                        }
                    }
                } // if c is one_char_word
                let mut an = true;
                let mut num = true;
                let mut dot_count = 0;
                for c in w.chars() {
                    an = an && (c.is_alphanumeric() || (c == '.') || (c == '\'') || (c == '-') || (c == '+') || (c == '_'));
                    num = num && (c.is_digit(10) || (c == '.') || (c == '-') || (c == '+'));
                    if c == '.' { dot_count += 1; }
                }
                if dot_count>1 { num = false; }
                if num {
                    return Some(local.local(BasicToken::Number(w)));
                }
                if an {
                    return Some(local.local(BasicToken::Alphanumeric(w)));
                }
                Some(local.local(BasicToken::Mixed(w)))
            },
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

