pub use unicode_categories::UnicodeCategories;
use regex::Regex;

use unicode_segmentation::{UnicodeSegmentation,UWordBounds};
use std::str::FromStr;
use std::collections::{VecDeque,BTreeSet};

mod emoji;

pub use emoji::EMOJIMAP;



#[derive(Debug,Clone,Copy,PartialEq,PartialOrd)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub enum Numerical {
    //Date(String),
    //Ip(String),
    DotSeparated(String),
    Measures(String),
    //Countable(String),
    Alphanumeric(String),
}

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd)]
pub enum Separator {
    Space,
    Tab,
    Newline,
    Unknown,
    Char(char),
}

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd)]
pub enum Formatter {
    Char(char),
    Joiner, // u{200d}
    Unknown,
}

#[derive(Debug,Clone,PartialEq,PartialOrd,Eq)]
pub enum BasicToken<'t> {
    Alphanumeric(&'t str),
    Number(&'t str),
    Punctuation(&'t str),
    Separator(&'t str),
    Formatter(&'t str),
    Mixed(&'t str),
}
impl<'t> BasicToken<'t> {
    fn len(&self) -> usize {
        match &self {
            BasicToken::Alphanumeric(s) |
            BasicToken::Number(s) |
            BasicToken::Punctuation(s) |
            BasicToken::Mixed(s) |
            BasicToken::Formatter(s) |
            BasicToken::Separator(s) => s.len(),
        }
    }
}

#[cfg(feature = "strings")]
#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub enum Token<T> {
    Word(String),
    StrangeWord(String),
    Hashtag(String),
    Mention(String),
    Punctuation(String),
    Numerical(Numerical),
    Number(Number),
    Emoji(&'static str),
    Unicode(String),
    Separator(Separator),
    UnicodeFormatter(Formatter),
    UnicodeModifier(char),
    Url(String),
    BBCode { left: Vec<T>, right: Vec<T> },
}
#[cfg(not(feature = "strings"))]
#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub enum Token<T> {
    Word,
    StrangeWord,
    Hashtag,
    Mention,
    Punctuation,
    Numerical(Numerical),
    Number(Number),
    Emoji(&'static str),
    Unicode(String),
    Separator(Separator),
    UnicodeFormatter(Formatter),
    UnicodeModifier(char),
    Url(String),
    BBCode { left: Vec<T>, right: Vec<T> },
}
impl<T> Token<T> {
    #[cfg(feature = "strings")]
    fn try_map<U,F,E>(self, func: F) -> Result<Token<U>,E>
        where F: Fn(T) -> Result<U,E>
    {
        match self {
            Token::Word(v) => Ok(Token::Word(v)),
            Token::StrangeWord(v) => Ok(Token::StrangeWord(v)),
            Token::Numerical(v) => Ok(Token::Numerical(v)),
            Token::Hashtag(v) => Ok(Token::Hashtag(v)),
            Token::Mention(v) => Ok(Token::Mention(v)),
            Token::Punctuation(v) => Ok(Token::Punctuation(v)),
            Token::Number(v) => Ok(Token::Number(v)),
            Token::Emoji(v) => Ok(Token::Emoji(v)),
            Token::Unicode(v) => Ok(Token::Unicode(v)),
            Token::Separator(v) => Ok(Token::Separator(v)),
            Token::UnicodeFormatter(v) => Ok(Token::UnicodeFormatter(v)),
            Token::UnicodeModifier(v) => Ok(Token::UnicodeModifier(v)),
            Token::Url(v) => Ok(Token::Url(v)),
            Token::BBCode { left, right } => Ok(Token::BBCode {
                left: {
                    let mut v = Vec::new();
                    for t in left { v.push(func(t)?); }
                    v
                },
                right: {
                    let mut v = Vec::new();
                    for t in right { v.push(func(t)?); }
                    v
                },
            }),
        }
    }

    #[cfg(not(feature = "strings"))]
        fn try_map<U,F,E>(self, func: F) -> Result<Token<U>,E>
        where F: Fn(T) -> Result<U,E>
    {
        match self {
            Token::Word => Ok(Token::Word),
            Token::StrangeWord => Ok(Token::StrangeWord),
            Token::Numerical(v) => Ok(Token::Numerical(v)),
            Token::Hashtag => Ok(Token::Hashtag),
            Token::Mention => Ok(Token::Mention),
            Token::Punctuation => Ok(Token::Punctuation),
            Token::Number(v) => Ok(Token::Number(v)),
            Token::Emoji(v) => Ok(Token::Emoji(v)),
            Token::Unicode(v) => Ok(Token::Unicode(v)),
            Token::Separator(v) => Ok(Token::Separator(v)),
            Token::UnicodeFormatter(v) => Ok(Token::UnicodeFormatter(v)),
            Token::UnicodeModifier(v) => Ok(Token::UnicodeModifier(v)),
            Token::Url(v) => Ok(Token::Url(v)),
            Token::BBCode { left, right } => Ok(Token::BBCode {
                left: {
                    let mut v = Vec::new();
                    for t in left { v.push(func(t)?); }
                    v
                },
                right: {
                    let mut v = Vec::new();
                    for t in right { v.push(func(t)?); }
                    v
                },
            }),
        }
    }
}

#[derive(Debug)]
pub struct CharBoundError(PositionalToken);
impl CharBoundError {
    pub fn into_inner(self) -> PositionalToken {
        self.0
    }
}

#[derive(Debug)]
struct ByteToChar(Vec<(usize,usize,usize)>); // byte-offset, char-offset, byte-length
impl ByteToChar {
    fn new(s: &str) -> ByteToChar {
        let mut v = Vec::new();
        for (char_off,(byte_off,c)) in s.char_indices().enumerate() {
            v.push((byte_off,char_off,c.len_utf8()));
        }
        ByteToChar(v)
    }
    #[allow(dead_code)]
    fn index(&self, byte: usize) -> Option<usize> {
        match self.0.binary_search_by_key(&byte,|(b,_,_)| *b) {
            Ok(idx) => Some(self.0[idx].1),
            Err(_) => None,
        }
    }
    fn sub(&self, offset: usize, length: usize) -> Option<(usize,usize)> {
        match (self.0.binary_search_by_key(&offset,|(b,_,_)| *b),self.0.binary_search_by_key(&(offset+length),|(b,_,_)| *b)) {
            (Ok(b),Ok(e)) => Some((self.0[b].1,e-b)),
            (Ok(b),Err(e)) if (e == self.0.len())&&(self.0[b..].iter().fold(0,|acc,(_,_,l)|acc+l) == length) => Some((self.0[b].1,e-b)),
            r @ _ => {
                println!("Err: ({},{}) -> {:?}\nBTOC: {:?}\n",offset,offset+length,r,self.0);
                None
            }
        }
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub struct CharToken {
    pub byte_offset: usize,
    pub byte_length: usize,
    pub char_offset: usize,
    pub char_length: usize,
    pub token: Token<CharToken>,   
}
impl CharToken {
    fn try_from(pt: PositionalToken, btoc: &ByteToChar) -> Result<CharToken,CharBoundError> {
        match btoc.sub(pt.offset,pt.length) {
            Some((co,cl)) => Ok(CharToken{
                byte_offset: pt.offset,
                byte_length: pt.length,
                char_offset: co,
                char_length: cl,
                token: pt.token.try_map(|pt| CharToken::try_from(pt,btoc))?,
            }),
            None => Err(CharBoundError(pt)),
        }
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub struct PositionalToken {
    pub offset: usize,
    pub length: usize,
    pub token: Token<PositionalToken>,   
}

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
pub enum TokenizerOptions {
    DetectBBCode,
    NoComplexTokens,
    StructTokens,
    SplitDot,
    SplitUnderscore,
    SplitColon,
}


enum ExceptionBounds<'t> {
    Bounds(UWordBounds<'t>),
    Vec {
        s: &'t str,
        v: std::vec::IntoIter<(usize,usize)>, // offset + length
    },
}
impl<'t> std::fmt::Debug for ExceptionBounds<'t> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExceptionBounds")
         .finish()
    }
}
impl<'t> Iterator for ExceptionBounds<'t> {
    type Item = &'t str;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ExceptionBounds::Bounds(bounds) => bounds.next(),
            ExceptionBounds::Vec{ s, v } => match v.next() {
                None => None,
                Some((o,l)) => Some(&s[o..(o+l)]),
            },
        }
    }
}

impl<'t> ExceptionBounds<'t> {
    fn new<'a>(s: &'a str) -> ExceptionBounds<'a> {
        match s.find("\u{0060}") {
            None => ExceptionBounds::Bounds(s.split_word_bounds()),
            Some(..) => {
                let new_s = s.replace("\u{0060}","\u{0027}");
                ExceptionBounds::Vec {
                    s: s,
                    v: new_s.split_word_bound_indices().into_iter().map(|(offset,st)| (offset,st.len())).collect::<Vec<_>>().into_iter(),
                }
            },
        }
    }
}

#[derive(Debug)]
struct ExtWordBounds<'t> {
    offset: usize,
    initial: &'t str,
    bounds: ExceptionBounds<'t>,
    buffer: VecDeque<&'t str>,
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
            initial: s,
            bounds: ExceptionBounds::new(s),
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
    type Item = &'t str;
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.len() > 0 { return self.buffer.pop_front(); }
        match self.bounds.next() {
            None => None,
            Some(w) => {
                let mut len = 0;
                let mut chs = w.chars().peekable();
                let num = match f64::from_str(w) { Ok(_) => true, Err(_) => false };
                while let Some(c) = chs.next() {
                    //let c_is_other_format = c.is_other_format();
                    //let exceptions_contain_c = self.exceptions.contains(&c);
                    let c_is_spliter = self.ext_spliters.contains(&c);
                    let c_is_punctuation = c.is_punctuation();                    
                    if  c_is_spliter //( c_is_other_format && !exceptions_contain_c )
                        || ( (c == '\u{200d}') && chs.peek().is_none() ) 
                        || ( c_is_punctuation && !num && !self.allow_complex ) // && !exceptions_contain_c 
                        || ( (c == '.') && !num && self.split_dot )
                        || ( (c == '_') && !num && self.split_underscore )
                        || ( (c == ':') && self.split_colon )
                    {
                        if len > 0 {
                            self.buffer.push_back(&self.initial[self.offset .. self.offset+len]);
                            self.offset += len;
                            len = 0;
                        }
                        self.buffer.push_back(&self.initial[self.offset .. self.offset+c.len_utf8()]);
                        self.offset += c.len_utf8();
                        
                    } else {
                        len += c.len_utf8();
                    }
                }
                if len > 0 {
                    self.buffer.push_back(&self.initial[self.offset .. self.offset+len]);
                    self.offset += len;
                }
                self.next()
            },
        }
    }
}

fn one_char_word(w: &str) -> Option<char> {
    // returns Some(char) if len in char == 1, None otherwise
    let mut cs = w.chars();
    match (cs.next(),cs.next()) {
        (Some(c),None) => Some(c),
        _ => None,
    }
}

#[derive(Debug)]
pub struct Breaker<'t> {
    offset: usize,
    initial: &'t str,
    prev_is_separator: bool,
    bounds: std::iter::Peekable<ExtWordBounds<'t>>,
}
impl<'t> Breaker<'t> {
    pub fn new<'a>(s: &'a str, options: &BTreeSet<TokenizerOptions>) -> Breaker<'a> {
        Breaker {
            offset: 0,
            initial: s,
            prev_is_separator: true,
            bounds: ExtWordBounds::new(s,options).peekable(),
        }
    }
    fn next_token(&mut self) -> Option<BasicToken<'t>> {
        match self.bounds.next() {
            Some(w) => {
                if let Some(c) = one_char_word(w) {
                    if ((c == '+')||(c == '-')) && self.prev_is_separator {
                        let mut len = c.len_utf8();                        
                        let num = if let Some(w2) = self.bounds.peek() {
                            let mut num = true;  
                            let mut dot_count = 0;
                            for c in w2.chars() {
                                num = num && (c.is_digit(10) || (c == '.'));
                                if c == '.' { dot_count += 1; }
                            }
                            if dot_count>1 { num = false; }
                            len += w2.len();
                            num
                        } else { false };
                        if num {
                            self.bounds.next();
                            let p = &self.initial[self.offset .. self.offset+len];
                            self.offset += len;
                            return Some(BasicToken::Number(p));
                        }                  
                    }
                    if c.is_ascii_punctuation() || c.is_punctuation() || c.is_whitespace() || c.is_other_format() {
                        let mut len = c.len_utf8();
                        loop {
                            match self.bounds.peek() {
                                Some(p) if *p==w => len += c.len_utf8(),
                                _ => break,
                            }
                            self.bounds.next();
                        }
                        let p = &self.initial[self.offset .. self.offset+len];
                        self.offset += len;
                        if c.is_ascii_punctuation() || c.is_punctuation() {
                            return Some(BasicToken::Punctuation(p));
                        }
                        if c.is_other_format() {
                            return Some(BasicToken::Formatter(p));
                        } else {
                            return Some(BasicToken::Separator(p));
                        }
                    }
                }
                let mut an = true;
                let mut num = true;
                let mut dot_count = 0;
                for c in w.chars() {
                    an = an && (c.is_alphanumeric() || (c == '.') || (c == '\'') || (c == '-') || (c == '+') || (c == '_'));
                    num = num && (c.is_digit(10) || (c == '.') || (c == '-') || (c == '+'));
                    if c == '.' { dot_count += 1; }
                }
                if dot_count>1 { num = false; }
                self.offset += w.len();
                if num {
                    return Some(BasicToken::Number(w));
                }
                if an {
                    return Some(BasicToken::Alphanumeric(w));
                }
                Some(BasicToken::Mixed(w))
            },
            None => None,
        }
    }
}
impl<'t> Iterator for Breaker<'t> {
    type Item = BasicToken<'t>;
    fn next(&mut self) -> Option<Self::Item> {
        let tok = self.next_token();
        self.prev_is_separator = match tok {
            Some(BasicToken::Separator(..)) => true,
            _ => false,
        };
        tok
    }
}

pub trait Tokenizer {
    fn next_token(&mut self) -> Option<PositionalToken>;
    fn next_char_token(&mut self) -> Option<Result<CharToken,CharBoundError>>;
}

fn detect_bbcodes(s: &str) -> VecDeque<(usize,usize,usize)> {
    lazy_static::lazy_static! {
        static ref RE: Regex = Regex::new(r"\[(.*?)\|(.*?)\]").unwrap();
    }
    let mut res = VecDeque::new(); 
    for cap in RE.captures_iter(s) {
        //println!("{:?} {:?}",cap,cap.get(0).map(|m0| (m0.start(),m0.end()-m0.start())));
        match (cap.get(0),cap.get(1),cap.get(2)) {
            (Some(m0),Some(m1),Some(m2)) => res.push_back((m0.start(),m1.end()-m1.start(),m2.end()-m2.start())),
            _ => continue,
        }
    }
    res
}

#[derive(Debug)]
pub struct Tokens<'t> {
    offset: usize,
    bounds: Breaker<'t>,
    buffer: VecDeque<BasicToken<'t>>,
    bbcodes: VecDeque<(usize,usize,usize)>,
    btoc: ByteToChar,
    allow_structs: bool,
}
impl<'t> Tokens<'t> {
    fn new<'a>(s: &'a str, options: BTreeSet<TokenizerOptions>) -> Tokens<'a> {
        Tokens {
            offset: 0,
            bounds: Breaker::new(s, &options),
            buffer: VecDeque::new(),
            bbcodes: if options.contains(&TokenizerOptions::DetectBBCode) { detect_bbcodes(s) } else { VecDeque::new() },
            btoc: ByteToChar::new(s),
            allow_structs: if options.contains(&TokenizerOptions::StructTokens) { true } else { false },
        }
    }
    fn basic<'a>(s: &'a str) -> Tokens<'a> {
        let options = vec![TokenizerOptions::NoComplexTokens].into_iter().collect();
        Tokens {
            offset: 0,
            bounds: Breaker::new(s, &options),
            buffer: VecDeque::new(),
            bbcodes: VecDeque::new(),
            btoc: ByteToChar::new(s),
            allow_structs: false,
        }
    }
    fn basic_separator_to_pt(&mut self, s: &str) -> PositionalToken {
        let tok = PositionalToken {
            offset: self.offset,
            length: s.len(),
            token: Token::Separator(match s.chars().next() {
                Some(' ') => Separator::Space,
                Some('\n') => Separator::Newline,
                Some('\t') => Separator::Tab,
                Some(c) => Separator::Char(c),
                None => Separator::Unknown,
            })
        };
        self.offset += s.len();
        tok
    }
    fn basic_formater_to_pt(&mut self, s: &str) -> PositionalToken {
        let tok = PositionalToken {
            offset: self.offset,
            length: s.len(),
            token: Token::UnicodeFormatter(match s.chars().next() {
                Some('\u{200d}') => Formatter::Joiner,
                Some(c) => Formatter::Char(c),
                None => Formatter::Unknown,
            }),
        };
        self.offset += s.len();
        tok
    }   
    fn basic_number_to_pt(&mut self, s: &str) -> PositionalToken {
        let tok = PositionalToken {
            offset: self.offset,
            length: s.len(),
            token: match i64::from_str(s) {
                Ok(n) => Token::Number(Number::Integer(n)),
                Err(_) => {
                    match f64::from_str(s) {
                        Ok(n) => Token::Number(Number::Float(n)),
                        Err(..) => {
                            #[cfg(feature = "strings")]
                            { Token::Word(s.to_string()) }
                            #[cfg(not(feature = "strings"))]
                            { Token::Word }  
                        },
                    }
                }
            },
        };
        self.offset += s.len();
        tok
    }
    fn basic_mixed_to_pt(&mut self, s: &str) -> PositionalToken {
        let tok = PositionalToken {
            offset: self.offset,
            length: s.len(),
            token: {
                let mut word = true;
                let mut has_word_parts = false;
                let mut first = true;
                let mut same = false;
                let mut one_c = ' ';
                for c in s.chars() {
                    match c.is_alphanumeric() || c.is_digit(10) || c.is_punctuation() || (c == '\u{0060}') {
                        true => { has_word_parts = true; },
                        false => { word = false; },
                    }
                    match first {
                        true => { one_c = c; first = false; same = true; },
                        false => if one_c != c { same = false; }
                    }
                }
                if !first && same && (one_c.is_whitespace() || one_c.is_other_format()) {
                    if one_c.is_whitespace() {
                        return self.basic_separator_to_pt(s);                       
                    } else {
                        return self.basic_formater_to_pt(s)
                    }
                }
                if word {
                    #[cfg(feature = "strings")]
                    { Token::StrangeWord(s.to_string()) }
                    #[cfg(not(feature = "strings"))]
                    { Token::StrangeWord }  
                } else {
                    let rs = s.replace("\u{fe0f}","");
                    match EMOJIMAP.get(&rs as &str) {
                        Some(em) => Token::Emoji(em),
                        None => match one_char_word(&rs) {
                            Some(c) if c.is_symbol_modifier() => Token::UnicodeModifier(c),
                            Some(_) | None => match has_word_parts {
                                true => {
                                    #[cfg(feature = "strings")]
                                    { Token::StrangeWord(s.to_string()) }
                                    #[cfg(not(feature = "strings"))]
                                    { Token::StrangeWord }  
                                },
                                false => Token::Unicode({
                                    let mut us = "".to_string();
                                    for c in rs.chars() {
                                        if us!="" { us += "_"; }
                                        us += "u";
                                        let ns = format!("{}",c.escape_unicode());
                                        us += &ns[3 .. ns.len()-1];
                                    }
                                    us
                                }),
                            }
                        },
                    }
                }
            }
        };
        self.offset += s.len();
        tok
    }
    fn basic_alphanumeric_to_pt(&mut self, s: &str) -> PositionalToken {
        /*
        Word
        StrangeWord
        pub enum Numerical {
            Date(String),
            Ip(String),
            DotSeparated(String),
            Countable(String),
            Measures(String),
            Alphanumeric(String),
        }*/
        //let mut wrd = true;
        let mut digits = false;
        let mut digits_begin_only = false;
        let mut dots = false;
        let mut alphas_and_apos = false;
        let mut other = false;

        let mut start_digit = true;
        for c in s.chars() {
            if start_digit && (!c.is_digit(10)) { start_digit = false; }
            match c {
                c @ _ if c.is_digit(10) => {
                    digits = true;
                    if start_digit { digits_begin_only = true; }
                    else { digits_begin_only = false; }
                },
                c @ _ if c.is_alphabetic() => { alphas_and_apos = true; },
                '\'' => { alphas_and_apos = true; },
                '.' => { dots = true; },
                _ => { other = true; },
            }
        }
        let tok = PositionalToken {
            offset: self.offset,
            length: s.len(),
            token: match (digits,digits_begin_only,dots,alphas_and_apos,other) {
                (true,false,true,false,false) => {
                    // TODO: Date, Ip, DotSeparated
                    Token::Numerical(Numerical::DotSeparated(s.to_string()))
                },
                (true,true,_,true,false) => {
                    // TODO: Countable or Measures
                    Token::Numerical(Numerical::Measures(s.to_string()))
                },
                (true, _, _, _, _) => {
                    // Numerical trash, ids, etc.
                    Token::Numerical(Numerical::Alphanumeric(s.to_string()))
                }
                (false,false,_,true,false) => {
                    // Word
                    #[cfg(feature = "strings")]
                    { Token::Word(s.to_string()) }
                    #[cfg(not(feature = "strings"))]
                    { Token::Word }  
                },
                (false,false,_,_,_) => {
                    // Strange                    
                    #[cfg(feature = "strings")]
                    { Token::StrangeWord(s.to_string()) }
                    #[cfg(not(feature = "strings"))]
                    { Token::StrangeWord }   
                },
                (false,true,_,_,_) => unreachable!(),
            },
        };
        self.offset += s.len();
        tok
    }
    fn basic_punctuation_to_pt(&mut self, s: &str) -> PositionalToken {
        let tok = PositionalToken {
            offset: self.offset,
            length: s.len(),
            token: {
                #[cfg(feature = "strings")]
                { Token::Punctuation(s.to_string()) }
                #[cfg(not(feature = "strings"))]
                { Token::Punctuation }
            },
        };
        self.offset += s.len();
        tok
    }
    fn check_url(&mut self) -> Option<PositionalToken> {
        if !self.allow_structs { return None; }
        let check = if self.buffer.len()>3 {
            match (&self.buffer[0],&self.buffer[1],&self.buffer[2]) {
                (BasicToken::Alphanumeric("http"),BasicToken::Punctuation(":"),BasicToken::Punctuation("//")) |
                (BasicToken::Alphanumeric("https"),BasicToken::Punctuation(":"),BasicToken::Punctuation("//")) => true,
                _ => false,
            }
        } else { false };
        if check {
            let mut url = "".to_string();
            let tag_bound = {
                if self.bbcodes.len()>0 { Some(self.bbcodes[0].0) } else { None }
            };
            loop {
                if let Some(b) = tag_bound {
                    if (self.offset + url.len()) >= b { break; }
                }
                match self.buffer.pop_front() {
                    None => break,
                    Some(BasicToken::Separator(s)) => {
                        self.buffer.push_front(BasicToken::Separator(s));
                        break;
                    },
                    Some(BasicToken::Alphanumeric(s)) |
                    Some(BasicToken::Number(s)) |
                    Some(BasicToken::Punctuation(s)) |
                    Some(BasicToken::Formatter(s)) |
                    Some(BasicToken::Mixed(s)) => {
                        url += s;
                    },
                }
            }
            let len = url.len();
            let tok = PositionalToken {
                offset: self.offset,
                length: len,
                token: Token::Url(url),
            };
            self.offset += len;
            Some(tok)
        } else { None }
    }
    fn check_hashtag(&mut self) -> Option<PositionalToken> {
        if !self.allow_structs { return None; }
        let tok = if self.buffer.len()>1 {
            match (&self.buffer[0],&self.buffer[1]) {
                (BasicToken::Punctuation("#"),BasicToken::Alphanumeric(s)) |
                (BasicToken::Punctuation("#"),BasicToken::Number(s)) => {
                    let tok = PositionalToken {
                        offset: self.offset,
                        length: s.len()+1,
                        token: {
                            #[cfg(feature = "strings")]
                            { Token::Hashtag(s.to_string()) }
                            #[cfg(not(feature = "strings"))]
                            { Token::Hashtag }
                        },
                    };
                    self.offset += s.len()+1;
                    Some(tok)
                },
                _ => None,
            }
        } else { None };
        if tok.is_some() {
            self.buffer.pop_front();
            self.buffer.pop_front();
        }
        tok
    }
    fn check_mention(&mut self) -> Option<PositionalToken> {
        if !self.allow_structs { return None; }
        let tok = if self.buffer.len()>1 {
            match (&self.buffer[0],&self.buffer[1]) {
                (BasicToken::Punctuation("@"),BasicToken::Alphanumeric(s)) |
                (BasicToken::Punctuation("@"),BasicToken::Number(s)) => {
                    let tok = PositionalToken {
                        offset: self.offset,
                        length: s.len()+1,
                        token: {
                            #[cfg(feature = "strings")]
                            { Token::Mention(s.to_string()) }
                            #[cfg(not(feature = "strings"))]
                            { Token::Mention }
                        },
                    };
                    self.offset += s.len()+1;
                    Some(tok)
                },
                _ => None,
            }
        } else { None };
        if tok.is_some() {
            self.buffer.pop_front();
            self.buffer.pop_front();
        }
        tok
    }
    fn check_bb_code(&mut self, text_len: usize, data_len: usize) -> Option<PositionalToken> {
        if self.buffer.len() >= (text_len+data_len+3) {
            if (self.buffer[0] == BasicToken::Punctuation("["))&&
                (self.buffer[text_len+1] == BasicToken::Punctuation("|"))&&
                (self.buffer[text_len+data_len+2] == BasicToken::Punctuation("]")) {
                    let offset = self.offset;
                    self.buffer.pop_front(); self.offset += 1;
                    let mut tail = self.buffer.split_off(text_len);
                    let mut text_vec = Vec::new(); 
                    while let Some(t) = self.next_from_buffer() {
                        text_vec.push(t);
                    }
                    std::mem::swap(&mut tail,&mut self.buffer);
                    self.buffer.pop_front(); self.offset += 1;
                    tail = self.buffer.split_off(data_len);
                    let mut data_vec = Vec::new(); 
                    while let Some(t) = self.next_from_buffer() {
                        data_vec.push(t);
                    }
                    std::mem::swap(&mut tail,&mut self.buffer);
                    self.buffer.pop_front(); self.offset += 1;
                    Some(PositionalToken {
                        offset: offset,
                        length: self.offset - offset,
                        token: Token::BBCode{ left: text_vec, right: data_vec },
                    })
                } else { None }
        } else { None }

    }
    fn next_from_buffer(&mut self) -> Option<PositionalToken> {
        if let Some(t) = self.check_url() { return Some(t); }
        if let Some(t) = self.check_hashtag() { return Some(t); }
        if let Some(t) = self.check_mention() { return Some(t); }
        match self.buffer.pop_front() {
            Some(BasicToken::Alphanumeric(s)) => Some(self.basic_alphanumeric_to_pt(s)),
            Some(BasicToken::Number(s)) => Some(self.basic_number_to_pt(s)),
            Some(BasicToken::Punctuation(s)) => Some(self.basic_punctuation_to_pt(s)),
            Some(BasicToken::Mixed(s)) => Some(self.basic_mixed_to_pt(s)),
            Some(BasicToken::Separator(s)) => Some(self.basic_separator_to_pt(s)),
            Some(BasicToken::Formatter(s)) => Some(self.basic_formater_to_pt(s)),
            None => None,
        }
    }
}

impl<'t> Tokenizer for Tokens<'t> {
    fn next_token(&mut self) -> Option<PositionalToken> {
        loop {
            if self.buffer.len()>0 {
                if (self.bbcodes.len()>0)&&(self.bbcodes[0].0 == self.offset) {
                    let get_len = self.bbcodes[0].1 + self.bbcodes[0].2 + 3;
                    let (text_from,text_len) = (self.bbcodes[0].0+1,self.bbcodes[0].1);
                    let (text2_from,text2_len) = (self.bbcodes[0].0+self.bbcodes[0].1+2,self.bbcodes[0].2);
                    let mut cur_len = 0;
                    let mut cur_off = self.offset;
                    let mut buf1_len = 0;
                    let mut buf2_len = 0;
                    for bt in &self.buffer {
                        if (cur_off>=text_from)&&(cur_off<(text_from+text_len)) { buf1_len += 1; } 
                        if (cur_off>=text2_from)&&(cur_off<(text2_from+text2_len)) { buf2_len += 1; }
                        cur_off += bt.len();
                        cur_len += bt.len();
                    }
                    while cur_len<get_len {
                        match self.bounds.next() {
                            None => break,
                            Some(bt) => {
                                if (cur_off>=text_from)&&(cur_off<(text_from+text_len)) { buf1_len += 1; } 
                                if (cur_off>=text2_from)&&(cur_off<(text2_from+text2_len)) { buf2_len += 1; }
                                cur_off += bt.len();
                                cur_len += bt.len();
                                self.buffer.push_back(bt);
                            }
                        }
                    }
                    //println!("{:?} {} {} {}",self.bbcodes[0],self.buffer.len(),buf1_len,buf2_len);
                    //println!("{:?}",self.buffer);
                    self.bbcodes.pop_front();
                    if let Some(t) = self.check_bb_code(buf1_len,buf2_len) { return Some(t); }
                }
                return self.next_from_buffer();
            } else {
                loop {
                    match self.bounds.next() {
                        Some(BasicToken::Separator(s)) => {
                            self.buffer.push_back(BasicToken::Separator(s));
                            return self.next_token();
                        },
                        Some(bt) => self.buffer.push_back(bt),
                        None if self.buffer.len()>0 => return self.next_token(),
                        None => return None,
                    }
                }
            }
        }
    }
    fn next_char_token(&mut self) -> Option<Result<CharToken,CharBoundError>> {
        self.next_token().map(|pt| CharToken::try_from(pt,&self.btoc))
    }
}

impl<'t> Iterator for Tokens<'t> {
    type Item = PositionalToken;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

pub trait IntoTokenizer {
    type IntoTokens: Tokenizer;
    fn into_tokens(self) -> Self::IntoTokens;
    fn into_tokens_with_options(self, options:BTreeSet<TokenizerOptions>) -> Self::IntoTokens;
    fn basic_tokens(self) -> Self::IntoTokens;
    fn complex_tokens(self) -> Self::IntoTokens;
}
impl<'t> IntoTokenizer for &'t str {
    type IntoTokens = Tokens<'t>;
    fn into_tokens(self) -> Self::IntoTokens {
        Tokens::new(self,vec![TokenizerOptions::SplitDot,TokenizerOptions::SplitUnderscore,TokenizerOptions::SplitColon].into_iter().collect())
    }
    fn into_tokens_with_options(self, options:BTreeSet<TokenizerOptions>) -> Self::IntoTokens {
        Tokens::new(self,options)
    }
    fn complex_tokens(self) -> Self::IntoTokens {
        Tokens::new(self,vec![TokenizerOptions::DetectBBCode,TokenizerOptions::StructTokens].into_iter().collect())
    }
    fn basic_tokens(self) -> Self::IntoTokens {
        Tokens::basic(self)
    }
}



#[cfg(test)]
mod test {
    use super::*;

    #[allow(dead_code)]
    fn print_pt(tok: &PositionalToken) -> String {
        let mut r = match &tok.token {
            Token::BBCode{ left, right } => {
                let left = print_pts(left);
                let right = print_pts(right);
                format!("PositionalToken {{ offset: {}, length: {}, token: Token::BBCode {{ left: vec![\n{}], right: vec![\n{}] }} }},",tok.offset,tok.length,left,right)
            },
            _ => format!("PositionalToken {{ offset: {}, length: {}, token: Token::{:?} }},",tok.offset,tok.length,tok.token),
        };
        r = r.replace("\")","\".to_string())");
        r
    }
    #[allow(dead_code)]
    fn print_pts(lib_res: &Vec<PositionalToken>) -> String {
        let mut r = String::new();
        for tok in lib_res {        
            r += &print_pt(&tok);
            r += "\n";
        }
        r
    }
    #[allow(dead_code)]
    fn print_result(lib_res: &Vec<PositionalToken>) {
        let mut r = print_pts(lib_res);
        r = r.replace("Separator(","Separator(Separator::");
        r = r.replace("UnicodeFormatter(","UnicodeFormatter(Formatter::");
        r = r.replace("Number(","Number(Number::");
        r = r.replace("Numerical(","Numerical(Numerical::");
        println!("{}",r);
    }

    #[allow(dead_code)]
    fn print_ct(tok: &CharToken) -> String {
        let mut r = match &tok.token {
            Token::BBCode{ left, right } => {
                let left = print_cts(left);
                let right = print_cts(right);
                format!("CharToken {{ byte_offset: {}, byte_length: {}, char_offset: {}, char_length: {}, token: Token::BBCode {{ left: vec![\n{}], right: vec![\n{}] }} }},",tok.byte_offset,tok.byte_length,tok.char_offset,tok.char_length,left,right)
            },
            _ => format!("CharToken {{ byte_offset: {}, byte_length: {}, char_offset: {}, char_length: {}, token: Token::{:?} }},",tok.byte_offset,tok.byte_length,tok.char_offset,tok.char_length,tok.token),
        };
        r = r.replace("\")","\".to_string())");
        r
    }
    #[allow(dead_code)]
    fn print_cts(lib_res: &Vec<CharToken>) -> String {
        let mut r = String::new();
        for tok in lib_res {        
            r += &print_ct(&tok);
            r += "\n";
        }
        r
    }
    #[allow(dead_code)]
    fn print_cresult(lib_res: &Vec<CharToken>) {
        let mut r = print_cts(lib_res);
        r = r.replace("Separator(","Separator(Separator::");
        r = r.replace("UnicodeFormatter(","UnicodeFormatter(Formatter::");
        r = r.replace("Number(","Number(Number::");
        r = r.replace("Numerical(","Numerical(Numerical::");
        println!("{}",r);
    }

    fn check_results(result: &Vec<PositionalToken>, lib_res: &Vec<PositionalToken>, _uws: &str) {
        assert_eq!(result.len(),lib_res.len());
        for i in 0 .. result.len() {
            assert_eq!(result[i],lib_res[i]);
        }
    }

    fn check_cresults(result: &Vec<CharToken>, lib_res: &Vec<CharToken>, _uws: &str) {
        assert_eq!(result.len(),lib_res.len());
        for i in 0 .. result.len() {
            assert_eq!(result[i],lib_res[i]);
        }
    }

    fn check<T: PartialEq + std::fmt::Debug>(res: &Vec<T>, lib: &Vec<T>, _uws: &str) {
        let mut lib = lib.iter();
        let mut res = res.iter();
        let mut diff = Vec::new();
        loop {
            match (lib.next(),res.next()) {
                (Some(lw),Some(rw)) => {
                    if lw != rw {
                        diff.push(format!("LIB:  {:?}",lw));
                        diff.push(format!("TEST: {:?}",rw));
                        diff.push("".to_string())
                    }
                },
                (Some(lw),None) => {
                    diff.push(format!("LIB:  {:?}",lw));
                    diff.push("TEST: ----".to_string());
                    diff.push("".to_string())
                },
                (None,Some(rw)) => {
                    diff.push("LIB:  ----".to_string());
                    diff.push(format!("TEST: {:?}",rw));
                    diff.push("".to_string())
                },
                (None,None) => break,
            }
        }
        if diff.len() > 0 {
            for ln in &diff { println!("{}",ln); }
            panic!("Diff count: {}",diff.len()/3);
        }
    }

    #[test]
    fn spaces() {
        let uws = "    spaces    too   many   apces   ";
        let result = vec![
            PositionalToken { offset: 0, length: 4, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 4, length: 6, token: Token::Word("spaces".to_string()) },
            PositionalToken { offset: 10, length: 4, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 14, length: 3, token: Token::Word("too".to_string()) },
            PositionalToken { offset: 17, length: 3, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 20, length: 4, token: Token::Word("many".to_string()) },
            PositionalToken { offset: 24, length: 3, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 27, length: 5, token: Token::Word("apces".to_string()) },
            PositionalToken { offset: 32, length: 3, token: Token::Separator(Separator::Space) },
        ];        
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
    }
    
    #[test]
    fn word_with_inner_hyphens() {
        let uws = "Опро­сы по­ка­зы­ва­ют";
        let result = vec![PositionalToken { offset: 0, length: 14, token: Token::StrangeWord("Опро­сы".to_string()) },
                          PositionalToken { offset: 14, length: 1, token: Token::Separator(Separator::Space) },
                          PositionalToken { offset: 15, length: 28, token: Token::StrangeWord("по­ка­зы­ва­ют".to_string()) }];        
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);        
    }

    #[test]
    fn mixed_but_word() {
        let uws = "L’Oreal";
        let result = vec![PositionalToken { offset: 0, length: 9, token: Token::StrangeWord("L’Oreal".to_string()) }];        
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
    }

    #[test]
    fn apostrophe() {
        let uws = "l'oreal; l\u{0060}oreal";
        let result = vec![
            PositionalToken { offset: 0, length: 7, token: Token::Word("l\'oreal".to_string()) },
            PositionalToken { offset: 7, length: 1, token: Token::Punctuation(";".to_string()) },
            PositionalToken { offset: 8, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 9, length: 7, token: Token::StrangeWord("l`oreal".to_string()) },
        ];        
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
    }

    #[test]
    fn char_tokens() {
        let uws = "[Oxana Putan|1712640565] shared the quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n+Done! Готово";
        let result = vec![
            CharToken { byte_offset: 0, byte_length: 24, char_offset: 0, char_length: 24, token: Token::BBCode { left: vec![
                CharToken { byte_offset: 1, byte_length: 5, char_offset: 1, char_length: 5, token: Token::Word("Oxana".to_string()) },
                CharToken { byte_offset: 6, byte_length: 1, char_offset: 6, char_length: 1, token: Token::Separator(Separator::Space) },
                CharToken { byte_offset: 7, byte_length: 5, char_offset: 7, char_length: 5, token: Token::Word("Putan".to_string()) },
                ], right: vec![
                CharToken { byte_offset: 13, byte_length: 10, char_offset: 13, char_length: 10, token: Token::Number(Number::Integer(1712640565)) },
                ] } },
            CharToken { byte_offset: 24, byte_length: 1, char_offset: 24, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 25, byte_length: 6, char_offset: 25, char_length: 6, token: Token::Word("shared".to_string()) },
            CharToken { byte_offset: 31, byte_length: 1, char_offset: 31, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 32, byte_length: 3, char_offset: 32, char_length: 3, token: Token::Word("the".to_string()) },
            CharToken { byte_offset: 35, byte_length: 1, char_offset: 35, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 36, byte_length: 5, char_offset: 36, char_length: 5, token: Token::Word("quick".to_string()) },
            CharToken { byte_offset: 41, byte_length: 1, char_offset: 41, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 42, byte_length: 1, char_offset: 42, char_length: 1, token: Token::Punctuation("(".to_string()) },
            CharToken { byte_offset: 43, byte_length: 1, char_offset: 43, char_length: 1, token: Token::Punctuation("\"".to_string()) },
            CharToken { byte_offset: 44, byte_length: 5, char_offset: 44, char_length: 5, token: Token::Word("brown".to_string()) },
            CharToken { byte_offset: 49, byte_length: 1, char_offset: 49, char_length: 1, token: Token::Punctuation("\"".to_string()) },
            CharToken { byte_offset: 50, byte_length: 1, char_offset: 50, char_length: 1, token: Token::Punctuation(")".to_string()) },
            CharToken { byte_offset: 51, byte_length: 1, char_offset: 51, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 52, byte_length: 3, char_offset: 52, char_length: 3, token: Token::Word("fox".to_string()) },
            CharToken { byte_offset: 55, byte_length: 1, char_offset: 55, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 56, byte_length: 5, char_offset: 56, char_length: 5, token: Token::Word("can\'t".to_string()) },
            CharToken { byte_offset: 61, byte_length: 1, char_offset: 61, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 62, byte_length: 4, char_offset: 62, char_length: 4, token: Token::Word("jump".to_string()) },
            CharToken { byte_offset: 66, byte_length: 1, char_offset: 66, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 67, byte_length: 4, char_offset: 67, char_length: 4, token: Token::Number(Number::Float(32.3)) },
            CharToken { byte_offset: 71, byte_length: 1, char_offset: 71, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 72, byte_length: 4, char_offset: 72, char_length: 4, token: Token::Word("feet".to_string()) },
            CharToken { byte_offset: 76, byte_length: 1, char_offset: 76, char_length: 1, token: Token::Punctuation(",".to_string()) },
            CharToken { byte_offset: 77, byte_length: 1, char_offset: 77, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 78, byte_length: 5, char_offset: 78, char_length: 5, token: Token::Word("right".to_string()) },
            CharToken { byte_offset: 83, byte_length: 1, char_offset: 83, char_length: 1, token: Token::Punctuation("?".to_string()) },
            CharToken { byte_offset: 84, byte_length: 1, char_offset: 84, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 85, byte_length: 4, char_offset: 85, char_length: 4, token: Token::Numerical(Numerical::Measures("4pda".to_string())) },
            CharToken { byte_offset: 89, byte_length: 1, char_offset: 89, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 90, byte_length: 3, char_offset: 90, char_length: 3, token: Token::Word("etc".to_string()) },
            CharToken { byte_offset: 93, byte_length: 1, char_offset: 93, char_length: 1, token: Token::Punctuation(".".to_string()) },
            CharToken { byte_offset: 94, byte_length: 1, char_offset: 94, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 95, byte_length: 3, char_offset: 95, char_length: 3, token: Token::Word("qeq".to_string()) },
            CharToken { byte_offset: 98, byte_length: 1, char_offset: 98, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 99, byte_length: 5, char_offset: 99, char_length: 5, token: Token::Word("U.S.A".to_string()) },
            CharToken { byte_offset: 104, byte_length: 2, char_offset: 104, char_length: 2, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 106, byte_length: 3, char_offset: 106, char_length: 3, token: Token::Word("asd".to_string()) },
            CharToken { byte_offset: 109, byte_length: 3, char_offset: 109, char_length: 3, token: Token::Separator(Separator::Newline) },
            CharToken { byte_offset: 112, byte_length: 3, char_offset: 112, char_length: 3, token: Token::Word("Brr".to_string()) },
            CharToken { byte_offset: 115, byte_length: 1, char_offset: 115, char_length: 1, token: Token::Punctuation(",".to_string()) },
            CharToken { byte_offset: 116, byte_length: 1, char_offset: 116, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 117, byte_length: 4, char_offset: 117, char_length: 4, token: Token::Word("it\'s".to_string()) },
            CharToken { byte_offset: 121, byte_length: 1, char_offset: 121, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 122, byte_length: 4, char_offset: 122, char_length: 4, token: Token::Number(Number::Float(29.3)) },
            CharToken { byte_offset: 126, byte_length: 2, char_offset: 126, char_length: 1, token: Token::Unicode("ub0".to_string()) },
            CharToken { byte_offset: 128, byte_length: 1, char_offset: 127, char_length: 1, token: Token::Word("F".to_string()) },
            CharToken { byte_offset: 129, byte_length: 1, char_offset: 128, char_length: 1, token: Token::Punctuation("!".to_string()) },
            CharToken { byte_offset: 130, byte_length: 1, char_offset: 129, char_length: 1, token: Token::Separator(Separator::Newline) },
            CharToken { byte_offset: 131, byte_length: 1, char_offset: 130, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 132, byte_length: 14, char_offset: 131, char_length: 7, token: Token::Word("Русское".to_string()) },
            CharToken { byte_offset: 146, byte_length: 1, char_offset: 138, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 147, byte_length: 22, char_offset: 139, char_length: 11, token: Token::Word("предложение".to_string()) },
            CharToken { byte_offset: 169, byte_length: 1, char_offset: 150, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 170, byte_length: 5, char_offset: 151, char_length: 5, token: Token::Hashtag("36.6".to_string()) },
            CharToken { byte_offset: 175, byte_length: 1, char_offset: 156, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 176, byte_length: 6, char_offset: 157, char_length: 3, token: Token::Word("для".to_string()) },
            CharToken { byte_offset: 182, byte_length: 1, char_offset: 160, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 183, byte_length: 24, char_offset: 161, char_length: 12, token: Token::Word("тестирования".to_string()) },
            CharToken { byte_offset: 207, byte_length: 1, char_offset: 173, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 208, byte_length: 14, char_offset: 174, char_length: 7, token: Token::Word("деления".to_string()) },
            CharToken { byte_offset: 222, byte_length: 1, char_offset: 181, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 223, byte_length: 4, char_offset: 182, char_length: 2, token: Token::Word("по".to_string()) },
            CharToken { byte_offset: 227, byte_length: 1, char_offset: 184, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 228, byte_length: 12, char_offset: 185, char_length: 6, token: Token::Word("юникод".to_string()) },
            CharToken { byte_offset: 240, byte_length: 1, char_offset: 191, char_length: 1, token: Token::Punctuation("-".to_string()) },
            CharToken { byte_offset: 241, byte_length: 12, char_offset: 192, char_length: 6, token: Token::Word("словам".to_string()) },
            CharToken { byte_offset: 253, byte_length: 3, char_offset: 198, char_length: 3, token: Token::Punctuation("...".to_string()) },
            CharToken { byte_offset: 256, byte_length: 1, char_offset: 201, char_length: 1, token: Token::Separator(Separator::Newline) },
            CharToken { byte_offset: 257, byte_length: 8, char_offset: 202, char_length: 2, token: Token::Emoji("russia") },
            CharToken { byte_offset: 265, byte_length: 1, char_offset: 204, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 266, byte_length: 8, char_offset: 205, char_length: 2, token: Token::Emoji("sao_tome_and_principe") },
            CharToken { byte_offset: 274, byte_length: 1, char_offset: 207, char_length: 1, token: Token::Separator(Separator::Newline) },
            CharToken { byte_offset: 275, byte_length: 8, char_offset: 208, char_length: 2, token: Token::Emoji("blond_haired_person_dark_skin_tone") },
            CharToken { byte_offset: 283, byte_length: 8, char_offset: 210, char_length: 2, token: Token::Emoji("baby_medium_skin_tone") },
            CharToken { byte_offset: 291, byte_length: 8, char_offset: 212, char_length: 2, token: Token::Emoji("man_medium_skin_tone") },
            CharToken { byte_offset: 299, byte_length: 1, char_offset: 214, char_length: 1, token: Token::Separator(Separator::Newline) },
            CharToken { byte_offset: 300, byte_length: 1, char_offset: 215, char_length: 1, token: Token::Punctuation("+".to_string()) },
            CharToken { byte_offset: 301, byte_length: 4, char_offset: 216, char_length: 4, token: Token::Word("Done".to_string()) },
            CharToken { byte_offset: 305, byte_length: 1, char_offset: 220, char_length: 1, token: Token::Punctuation("!".to_string()) },
            CharToken { byte_offset: 306, byte_length: 1, char_offset: 221, char_length: 1, token: Token::Separator(Separator::Space) },
            CharToken { byte_offset: 307, byte_length: 12, char_offset: 222, char_length: 6, token: Token::Word("Готово".to_string()) },
            ];
        let lib_res = {
            let mut v = Vec::new();
            let mut iter = uws.complex_tokens();
            while let Some(rct) = iter.next_char_token() {
                v.push(rct.unwrap());
            }
            v
        };
        //print_cresult(); panic!();
        check_cresults(&result,&lib_res,uws);
    }

    #[test]
    fn general_default() {
        let uws = "The quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n";
        let result = vec![
            PositionalToken { offset: 0, length: 3, token: Token::Word("The".to_string()) },
            PositionalToken { offset: 3, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 4, length: 5, token: Token::Word("quick".to_string()) },
            PositionalToken { offset: 9, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 10, length: 1, token: Token::Punctuation("(".to_string()) },
            PositionalToken { offset: 11, length: 1, token: Token::Punctuation("\"".to_string()) },
            PositionalToken { offset: 12, length: 5, token: Token::Word("brown".to_string()) },
            PositionalToken { offset: 17, length: 1, token: Token::Punctuation("\"".to_string()) },
            PositionalToken { offset: 18, length: 1, token: Token::Punctuation(")".to_string()) },
            PositionalToken { offset: 19, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 20, length: 3, token: Token::Word("fox".to_string()) },
            PositionalToken { offset: 23, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 24, length: 5, token: Token::Word("can\'t".to_string()) },
            PositionalToken { offset: 29, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 30, length: 4, token: Token::Word("jump".to_string()) },
            PositionalToken { offset: 34, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 35, length: 4, token: Token::Number(Number::Float(32.3)) },
            PositionalToken { offset: 39, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 40, length: 4, token: Token::Word("feet".to_string()) },
            PositionalToken { offset: 44, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 45, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 46, length: 5, token: Token::Word("right".to_string()) },
            PositionalToken { offset: 51, length: 1, token: Token::Punctuation("?".to_string()) },
            PositionalToken { offset: 52, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 53, length: 4, token: Token::Numerical(Numerical::Measures("4pda".to_string())) }, // TODO
            PositionalToken { offset: 57, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 58, length: 3, token: Token::Word("etc".to_string()) },
            PositionalToken { offset: 61, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 62, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 63, length: 3, token: Token::Word("qeq".to_string()) },
            PositionalToken { offset: 66, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 67, length: 1, token: Token::Word("U".to_string()) },
            PositionalToken { offset: 68, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 69, length: 1, token: Token::Word("S".to_string()) },
            PositionalToken { offset: 70, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 71, length: 1, token: Token::Word("A".to_string()) },
            PositionalToken { offset: 72, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 74, length: 3, token: Token::Word("asd".to_string()) },
            PositionalToken { offset: 77, length: 3, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 80, length: 3, token: Token::Word("Brr".to_string()) },
            PositionalToken { offset: 83, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 84, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 85, length: 4, token: Token::Word("it\'s".to_string()) },
            PositionalToken { offset: 89, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 90, length: 4, token: Token::Number(Number::Float(29.3)) },
            PositionalToken { offset: 94, length: 2, token: Token::Unicode("ub0".to_string()) },
            PositionalToken { offset: 96, length: 1, token: Token::Word("F".to_string()) },
            PositionalToken { offset: 97, length: 1, token: Token::Punctuation("!".to_string()) },
            PositionalToken { offset: 98, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 99, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 100, length: 14, token: Token::Word("Русское".to_string()) },
            PositionalToken { offset: 114, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 115, length: 22, token: Token::Word("предложение".to_string()) },
            PositionalToken { offset: 137, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 138, length: 1, token: Token::Punctuation("#".to_string()) },
            PositionalToken { offset: 139, length: 4, token: Token::Number(Number::Float(36.6)) },
            PositionalToken { offset: 143, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 144, length: 6, token: Token::Word("для".to_string()) },
            PositionalToken { offset: 150, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 151, length: 24, token: Token::Word("тестирования".to_string()) },
            PositionalToken { offset: 175, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 176, length: 14, token: Token::Word("деления".to_string()) },
            PositionalToken { offset: 190, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 191, length: 4, token: Token::Word("по".to_string()) },
            PositionalToken { offset: 195, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 196, length: 12, token: Token::Word("юникод".to_string()) },
            PositionalToken { offset: 208, length: 1, token: Token::Punctuation("-".to_string()) },
            PositionalToken { offset: 209, length: 12, token: Token::Word("словам".to_string()) },
            PositionalToken { offset: 221, length: 3, token: Token::Punctuation("...".to_string()) },
            PositionalToken { offset: 224, length: 1, token: Token::Separator(Separator::Newline) },
            ];
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
    }

    #[test]
    fn general_no_split() {
        let uws = "The quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n";
        let result = vec![
            PositionalToken { offset: 0, length: 3, token: Token::Word("The".to_string()) },
            PositionalToken { offset: 3, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 4, length: 5, token: Token::Word("quick".to_string()) },
            PositionalToken { offset: 9, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 10, length: 1, token: Token::Punctuation("(".to_string()) },
            PositionalToken { offset: 11, length: 1, token: Token::Punctuation("\"".to_string()) },
            PositionalToken { offset: 12, length: 5, token: Token::Word("brown".to_string()) },
            PositionalToken { offset: 17, length: 1, token: Token::Punctuation("\"".to_string()) },
            PositionalToken { offset: 18, length: 1, token: Token::Punctuation(")".to_string()) },
            PositionalToken { offset: 19, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 20, length: 3, token: Token::Word("fox".to_string()) },
            PositionalToken { offset: 23, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 24, length: 5, token: Token::Word("can\'t".to_string()) },
            PositionalToken { offset: 29, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 30, length: 4, token: Token::Word("jump".to_string()) },
            PositionalToken { offset: 34, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 35, length: 4, token: Token::Number(Number::Float(32.3)) },
            PositionalToken { offset: 39, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 40, length: 4, token: Token::Word("feet".to_string()) },
            PositionalToken { offset: 44, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 45, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 46, length: 5, token: Token::Word("right".to_string()) },
            PositionalToken { offset: 51, length: 1, token: Token::Punctuation("?".to_string()) },
            PositionalToken { offset: 52, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 53, length: 4, token: Token::Numerical(Numerical::Measures("4pda".to_string())) }, // TODO
            PositionalToken { offset: 57, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 58, length: 3, token: Token::Word("etc".to_string()) },
            PositionalToken { offset: 61, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 62, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 63, length: 3, token: Token::Word("qeq".to_string()) },
            PositionalToken { offset: 66, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 67, length: 5, token: Token::Word("U.S.A".to_string()) },
            PositionalToken { offset: 72, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 74, length: 3, token: Token::Word("asd".to_string()) },
            PositionalToken { offset: 77, length: 3, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 80, length: 3, token: Token::Word("Brr".to_string()) },
            PositionalToken { offset: 83, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 84, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 85, length: 4, token: Token::Word("it\'s".to_string()) },
            PositionalToken { offset: 89, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 90, length: 4, token: Token::Number(Number::Float(29.3)) },
            PositionalToken { offset: 94, length: 2, token: Token::Unicode("ub0".to_string()) },
            PositionalToken { offset: 96, length: 1, token: Token::Word("F".to_string()) },
            PositionalToken { offset: 97, length: 1, token: Token::Punctuation("!".to_string()) },
            PositionalToken { offset: 98, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 99, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 100, length: 14, token: Token::Word("Русское".to_string()) },
            PositionalToken { offset: 114, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 115, length: 22, token: Token::Word("предложение".to_string()) },
            PositionalToken { offset: 137, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 138, length: 1, token: Token::Punctuation("#".to_string()) },
            PositionalToken { offset: 139, length: 4, token: Token::Number(Number::Float(36.6)) },
            PositionalToken { offset: 143, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 144, length: 6, token: Token::Word("для".to_string()) },
            PositionalToken { offset: 150, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 151, length: 24, token: Token::Word("тестирования".to_string()) },
            PositionalToken { offset: 175, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 176, length: 14, token: Token::Word("деления".to_string()) },
            PositionalToken { offset: 190, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 191, length: 4, token: Token::Word("по".to_string()) },
            PositionalToken { offset: 195, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 196, length: 12, token: Token::Word("юникод".to_string()) },
            PositionalToken { offset: 208, length: 1, token: Token::Punctuation("-".to_string()) },
            PositionalToken { offset: 209, length: 12, token: Token::Word("словам".to_string()) },
            PositionalToken { offset: 221, length: 3, token: Token::Punctuation("...".to_string()) },
            PositionalToken { offset: 224, length: 1, token: Token::Separator(Separator::Newline) },
            ];
        let lib_res = uws.into_tokens_with_options(BTreeSet::new()).collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
    }
    
    #[test]
    fn general_complex() {
        let uws = "The quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n";
        let result = vec![
            PositionalToken { offset: 0, length: 3, token: Token::Word("The".to_string()) },
            PositionalToken { offset: 3, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 4, length: 5, token: Token::Word("quick".to_string()) },
            PositionalToken { offset: 9, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 10, length: 1, token: Token::Punctuation("(".to_string()) },
            PositionalToken { offset: 11, length: 1, token: Token::Punctuation("\"".to_string()) },
            PositionalToken { offset: 12, length: 5, token: Token::Word("brown".to_string()) },
            PositionalToken { offset: 17, length: 1, token: Token::Punctuation("\"".to_string()) },
            PositionalToken { offset: 18, length: 1, token: Token::Punctuation(")".to_string()) },
            PositionalToken { offset: 19, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 20, length: 3, token: Token::Word("fox".to_string()) },
            PositionalToken { offset: 23, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 24, length: 5, token: Token::Word("can\'t".to_string()) },
            PositionalToken { offset: 29, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 30, length: 4, token: Token::Word("jump".to_string()) },
            PositionalToken { offset: 34, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 35, length: 4, token: Token::Number(Number::Float(32.3)) },
            PositionalToken { offset: 39, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 40, length: 4, token: Token::Word("feet".to_string()) },
            PositionalToken { offset: 44, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 45, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 46, length: 5, token: Token::Word("right".to_string()) },
            PositionalToken { offset: 51, length: 1, token: Token::Punctuation("?".to_string()) },
            PositionalToken { offset: 52, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 53, length: 4, token: Token::Numerical(Numerical::Measures("4pda".to_string())) }, // TODO
            PositionalToken { offset: 57, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 58, length: 3, token: Token::Word("etc".to_string()) },
            PositionalToken { offset: 61, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 62, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 63, length: 3, token: Token::Word("qeq".to_string()) },
            PositionalToken { offset: 66, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 67, length: 5, token: Token::Word("U.S.A".to_string()) },
            PositionalToken { offset: 72, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 74, length: 3, token: Token::Word("asd".to_string()) },
            PositionalToken { offset: 77, length: 3, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 80, length: 3, token: Token::Word("Brr".to_string()) },
            PositionalToken { offset: 83, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 84, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 85, length: 4, token: Token::Word("it\'s".to_string()) },
            PositionalToken { offset: 89, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 90, length: 4, token: Token::Number(Number::Float(29.3)) },
            PositionalToken { offset: 94, length: 2, token: Token::Unicode("ub0".to_string()) },
            PositionalToken { offset: 96, length: 1, token: Token::Word("F".to_string()) },
            PositionalToken { offset: 97, length: 1, token: Token::Punctuation("!".to_string()) },
            PositionalToken { offset: 98, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 99, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 100, length: 14, token: Token::Word("Русское".to_string()) },
            PositionalToken { offset: 114, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 115, length: 22, token: Token::Word("предложение".to_string()) },
            PositionalToken { offset: 137, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 138, length: 5, token: Token::Hashtag("36.6".to_string()) },
            PositionalToken { offset: 143, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 144, length: 6, token: Token::Word("для".to_string()) },
            PositionalToken { offset: 150, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 151, length: 24, token: Token::Word("тестирования".to_string()) },
            PositionalToken { offset: 175, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 176, length: 14, token: Token::Word("деления".to_string()) },
            PositionalToken { offset: 190, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 191, length: 4, token: Token::Word("по".to_string()) },
            PositionalToken { offset: 195, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 196, length: 12, token: Token::Word("юникод".to_string()) },
            PositionalToken { offset: 208, length: 1, token: Token::Punctuation("-".to_string()) },
            PositionalToken { offset: 209, length: 12, token: Token::Word("словам".to_string()) },
            PositionalToken { offset: 221, length: 3, token: Token::Punctuation("...".to_string()) },
            PositionalToken { offset: 224, length: 1, token: Token::Separator(Separator::Newline) },
            ];
        let lib_res = uws.complex_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
    }

    #[test]
    fn plus_minus() {
        let uws = "+23 -4.5 -34 +25.7 - 2 + 5.6";
        let result = vec![
            PositionalToken { offset: 0, length: 3, token: Token::Number(Number::Integer(23)) },
            PositionalToken { offset: 3, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 4, length: 4, token: Token::Number(Number::Float(-4.5)) },
            PositionalToken { offset: 8, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 9, length: 3, token: Token::Number(Number::Integer(-34)) },
            PositionalToken { offset: 12, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 13, length: 5, token: Token::Number(Number::Float(25.7)) },
            PositionalToken { offset: 18, length: 1, token: Token::Separator(Separator::Space) },           
            PositionalToken { offset: 19, length: 1, token: Token::Punctuation("-".to_string()) },
            PositionalToken { offset: 20, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 21, length: 1, token: Token::Number(Number::Integer(2)) },
            PositionalToken { offset: 22, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 23, length: 1, token: Token::Punctuation("+".to_string()) },
            PositionalToken { offset: 24, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 25, length: 3, token: Token::Number(Number::Float(5.6)) },
            ];
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check(&result,&lib_res,uws);
        //print_result(&lib_res); panic!("")
    } 
    
    #[test]
    #[ignore]
    fn woman_bouncing_ball() {
        let uws = "\u{26f9}\u{200d}\u{2640}";
        let result = vec![PositionalToken { offset: 0, length: 9, token: Token::Emoji("woman_bouncing_ball") }];
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!("")
    } 
    
    #[test]
    fn emoji_and_rusabbr_default() {
        let uws = "🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n👱\nС.С.С.Р.\n👨‍👩‍👦‍👦\n🧠\n";
        let result = vec![
            PositionalToken { offset: 0, length: 8, token: Token::Emoji("russia") },
            PositionalToken { offset: 8, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 9, length: 8, token: Token::Emoji("sao_tome_and_principe") },
            PositionalToken { offset: 17, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 18, length: 8, token: Token::Emoji("blond_haired_person_dark_skin_tone") },
            PositionalToken { offset: 26, length: 8, token: Token::Emoji("baby_medium_skin_tone") },
            PositionalToken { offset: 34, length: 8, token: Token::Emoji("man_medium_skin_tone") },
            PositionalToken { offset: 42, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 43, length: 4, token: Token::Emoji("blond_haired_person") },
            PositionalToken { offset: 47, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 48, length: 2, token: Token::Word("С".to_string()) },
            PositionalToken { offset: 50, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 51, length: 2, token: Token::Word("С".to_string()) },
            PositionalToken { offset: 53, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 54, length: 2, token: Token::Word("С".to_string()) },
            PositionalToken { offset: 56, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 57, length: 2, token: Token::Word("Р".to_string()) },
            PositionalToken { offset: 59, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 60, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 61, length: 25, token: Token::Emoji("family_man_woman_boy_boy") },
            PositionalToken { offset: 86, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 87, length: 4, token: Token::Emoji("brain") },
            PositionalToken { offset: 91, length: 1, token: Token::Separator(Separator::Newline) },
            ];
        
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!();
    }

    #[test]
    fn emoji_and_rusabbr_no_split() {
        let uws = "🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n👱\nС.С.С.Р.\n👨‍👩‍👦‍👦\n🧠\n";
        let result = vec![
            PositionalToken { offset: 0, length: 8, token: Token::Emoji("russia") },
            PositionalToken { offset: 8, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 9, length: 8, token: Token::Emoji("sao_tome_and_principe") },
            PositionalToken { offset: 17, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 18, length: 8, token: Token::Emoji("blond_haired_person_dark_skin_tone") },
            PositionalToken { offset: 26, length: 8, token: Token::Emoji("baby_medium_skin_tone") },
            PositionalToken { offset: 34, length: 8, token: Token::Emoji("man_medium_skin_tone") },
            PositionalToken { offset: 42, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 43, length: 4, token: Token::Emoji("blond_haired_person") },
            PositionalToken { offset: 47, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 48, length: 11, token: Token::Word("С.С.С.Р".to_string()) },
            PositionalToken { offset: 59, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 60, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 61, length: 25, token: Token::Emoji("family_man_woman_boy_boy") },
            PositionalToken { offset: 86, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 87, length: 4, token: Token::Emoji("brain") },
            PositionalToken { offset: 91, length: 1, token: Token::Separator(Separator::Newline) },
            ];
        
        let lib_res = uws.into_tokens_with_options(BTreeSet::new()).collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!();
    }

    #[test]
    fn hashtags_mentions_urls() {
        let uws = "\nSome ##text with #hashtags and @other components\nadfa wdsfdf asdf asd http://asdfasdfsd.com/fasdfd/sadfsadf/sdfas/12312_12414/asdf?fascvx=fsfwer&dsdfasdf=fasdf#fasdf asdfa sdfa sdf\nasdfas df asd who@bla-bla.com asdfas df asdfsd\n";
        let result = vec![
            PositionalToken { offset: 0, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 1, length: 4, token: Token::Word("Some".to_string()) },
            PositionalToken { offset: 5, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 6, length: 2, token: Token::Punctuation("##".to_string()) },
            PositionalToken { offset: 8, length: 4, token: Token::Word("text".to_string()) },
            PositionalToken { offset: 12, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 13, length: 4, token: Token::Word("with".to_string()) },
            PositionalToken { offset: 17, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 18, length: 9, token: Token::Hashtag("hashtags".to_string()) },
            PositionalToken { offset: 27, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 28, length: 3, token: Token::Word("and".to_string()) },
            PositionalToken { offset: 31, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 32, length: 6, token: Token::Mention("other".to_string()) },
            PositionalToken { offset: 38, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 39, length: 10, token: Token::Word("components".to_string()) },
            PositionalToken { offset: 49, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 50, length: 4, token: Token::Word("adfa".to_string()) },
            PositionalToken { offset: 54, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 55, length: 6, token: Token::Word("wdsfdf".to_string()) },
            PositionalToken { offset: 61, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 62, length: 4, token: Token::Word("asdf".to_string()) },
            PositionalToken { offset: 66, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 67, length: 3, token: Token::Word("asd".to_string()) },
            PositionalToken { offset: 70, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 71, length: 95, token: Token::Url("http://asdfasdfsd.com/fasdfd/sadfsadf/sdfas/12312_12414/asdf?fascvx=fsfwer&dsdfasdf=fasdf#fasdf".to_string()) },
            PositionalToken { offset: 166, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 167, length: 5, token: Token::Word("asdfa".to_string()) },
            PositionalToken { offset: 172, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 173, length: 4, token: Token::Word("sdfa".to_string()) },
            PositionalToken { offset: 177, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 178, length: 3, token: Token::Word("sdf".to_string()) },
            PositionalToken { offset: 181, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 182, length: 6, token: Token::Word("asdfas".to_string()) },
            PositionalToken { offset: 188, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 189, length: 2, token: Token::Word("df".to_string()) },
            PositionalToken { offset: 191, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 192, length: 3, token: Token::Word("asd".to_string()) },
            PositionalToken { offset: 195, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 196, length: 3, token: Token::Word("who".to_string()) },
            PositionalToken { offset: 199, length: 4, token: Token::Mention("bla".to_string()) },
            PositionalToken { offset: 203, length: 1, token: Token::Punctuation("-".to_string()) },
            PositionalToken { offset: 204, length: 7, token: Token::Word("bla.com".to_string()) },
            PositionalToken { offset: 211, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 212, length: 6, token: Token::Word("asdfas".to_string()) },
            PositionalToken { offset: 218, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 219, length: 2, token: Token::Word("df".to_string()) },
            PositionalToken { offset: 221, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 222, length: 6, token: Token::Word("asdfsd".to_string()) },
            PositionalToken { offset: 228, length: 1, token: Token::Separator(Separator::Newline) },
            ];
        let lib_res = uws.complex_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!("")
    }

    #[test]
    fn bb_code() {
        let uws = "[Oxana Putan|1712640565] shared a [post|100001150683379_1873048549410150]. \nAndrew\n[link|https://www.facebook.com/100001150683379/posts/1873048549410150]\nДрузья мои, издатели, редакторы, просветители, культуртрегеры, субъекты мирового рынка и ту хум ит ещё мей консёрн.\nНа текущий момент я лишен былой подвижности, хоть и ковыляю по больничных коридорам по разным нуждам и за кипятком.\nВрачи обещают мне заживление отверстых ран моих в течение полугода и на этот период можно предполагать с уверенностью преимущественно домашний образ жизни.\n[|]";
        let result = vec![
            PositionalToken { offset: 0, length: 24, token: Token::BBCode { left: vec![
                PositionalToken { offset: 1, length: 5, token: Token::Word("Oxana".to_string()) },
                PositionalToken { offset: 6, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 7, length: 5, token: Token::Word("Putan".to_string()) },
                ], right: vec![
                PositionalToken { offset: 13, length: 10, token: Token::Number(Number::Integer(1712640565)) },
                ] } },
            PositionalToken { offset: 24, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 25, length: 6, token: Token::Word("shared".to_string()) },
            PositionalToken { offset: 31, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 32, length: 1, token: Token::Word("a".to_string()) },
            PositionalToken { offset: 33, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 34, length: 39, token: Token::BBCode { left: vec![
                PositionalToken { offset: 35, length: 4, token: Token::Word("post".to_string()) },
                ], right: vec![
                PositionalToken { offset: 40, length: 32, token: Token::Numerical(Numerical::Alphanumeric("100001150683379_1873048549410150".to_string())) },
                ] } },
            PositionalToken { offset: 73, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 74, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 75, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 76, length: 6, token: Token::Word("Andrew".to_string()) },
            PositionalToken { offset: 82, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 83, length: 70, token: Token::BBCode { left: vec![
                PositionalToken { offset: 84, length: 4, token: Token::Word("link".to_string()) },
                ], right: vec![
                PositionalToken { offset: 89, length: 63, token: Token::Url("https://www.facebook.com/100001150683379/posts/1873048549410150".to_string()) },
                ] } },
            PositionalToken { offset: 153, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 154, length: 12, token: Token::Word("Друзья".to_string()) },
            PositionalToken { offset: 166, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 167, length: 6, token: Token::Word("мои".to_string()) },
            PositionalToken { offset: 173, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 174, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 175, length: 16, token: Token::Word("издатели".to_string()) },
            PositionalToken { offset: 191, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 192, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 193, length: 18, token: Token::Word("редакторы".to_string()) },
            PositionalToken { offset: 211, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 212, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 213, length: 24, token: Token::Word("просветители".to_string()) },
            PositionalToken { offset: 237, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 238, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 239, length: 28, token: Token::Word("культуртрегеры".to_string()) },
            PositionalToken { offset: 267, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 268, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 269, length: 16, token: Token::Word("субъекты".to_string()) },
            PositionalToken { offset: 285, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 286, length: 16, token: Token::Word("мирового".to_string()) },
            PositionalToken { offset: 302, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 303, length: 10, token: Token::Word("рынка".to_string()) },
            PositionalToken { offset: 313, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 314, length: 2, token: Token::Word("и".to_string()) },
            PositionalToken { offset: 316, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 317, length: 4, token: Token::Word("ту".to_string()) },
            PositionalToken { offset: 321, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 322, length: 6, token: Token::Word("хум".to_string()) },
            PositionalToken { offset: 328, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 329, length: 4, token: Token::Word("ит".to_string()) },
            PositionalToken { offset: 333, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 334, length: 6, token: Token::Word("ещё".to_string()) },
            PositionalToken { offset: 340, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 341, length: 6, token: Token::Word("мей".to_string()) },
            PositionalToken { offset: 347, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 348, length: 14, token: Token::Word("консёрн".to_string()) },
            PositionalToken { offset: 362, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 363, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 364, length: 4, token: Token::Word("На".to_string()) },
            PositionalToken { offset: 368, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 369, length: 14, token: Token::Word("текущий".to_string()) },
            PositionalToken { offset: 383, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 384, length: 12, token: Token::Word("момент".to_string()) },
            PositionalToken { offset: 396, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 397, length: 2, token: Token::Word("я".to_string()) },
            PositionalToken { offset: 399, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 400, length: 10, token: Token::Word("лишен".to_string()) },
            PositionalToken { offset: 410, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 411, length: 10, token: Token::Word("былой".to_string()) },
            PositionalToken { offset: 421, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 422, length: 22, token: Token::Word("подвижности".to_string()) },
            PositionalToken { offset: 444, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 445, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 446, length: 8, token: Token::Word("хоть".to_string()) },
            PositionalToken { offset: 454, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 455, length: 2, token: Token::Word("и".to_string()) },
            PositionalToken { offset: 457, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 458, length: 14, token: Token::Word("ковыляю".to_string()) },
            PositionalToken { offset: 472, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 473, length: 4, token: Token::Word("по".to_string()) },
            PositionalToken { offset: 477, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 478, length: 20, token: Token::Word("больничных".to_string()) },
            PositionalToken { offset: 498, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 499, length: 18, token: Token::Word("коридорам".to_string()) },
            PositionalToken { offset: 517, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 518, length: 4, token: Token::Word("по".to_string()) },
            PositionalToken { offset: 522, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 523, length: 12, token: Token::Word("разным".to_string()) },
            PositionalToken { offset: 535, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 536, length: 12, token: Token::Word("нуждам".to_string()) },
            PositionalToken { offset: 548, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 549, length: 2, token: Token::Word("и".to_string()) },
            PositionalToken { offset: 551, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 552, length: 4, token: Token::Word("за".to_string()) },
            PositionalToken { offset: 556, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 557, length: 16, token: Token::Word("кипятком".to_string()) },
            PositionalToken { offset: 573, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 574, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 575, length: 10, token: Token::Word("Врачи".to_string()) },
            PositionalToken { offset: 585, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 586, length: 14, token: Token::Word("обещают".to_string()) },
            PositionalToken { offset: 600, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 601, length: 6, token: Token::Word("мне".to_string()) },
            PositionalToken { offset: 607, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 608, length: 20, token: Token::Word("заживление".to_string()) },
            PositionalToken { offset: 628, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 629, length: 18, token: Token::Word("отверстых".to_string()) },
            PositionalToken { offset: 647, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 648, length: 6, token: Token::Word("ран".to_string()) },
            PositionalToken { offset: 654, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 655, length: 8, token: Token::Word("моих".to_string()) },
            PositionalToken { offset: 663, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 664, length: 2, token: Token::Word("в".to_string()) },
            PositionalToken { offset: 666, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 667, length: 14, token: Token::Word("течение".to_string()) },
            PositionalToken { offset: 681, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 682, length: 16, token: Token::Word("полугода".to_string()) },
            PositionalToken { offset: 698, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 699, length: 2, token: Token::Word("и".to_string()) },
            PositionalToken { offset: 701, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 702, length: 4, token: Token::Word("на".to_string()) },
            PositionalToken { offset: 706, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 707, length: 8, token: Token::Word("этот".to_string()) },
            PositionalToken { offset: 715, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 716, length: 12, token: Token::Word("период".to_string()) },
            PositionalToken { offset: 728, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 729, length: 10, token: Token::Word("можно".to_string()) },
            PositionalToken { offset: 739, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 740, length: 24, token: Token::Word("предполагать".to_string()) },
            PositionalToken { offset: 764, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 765, length: 2, token: Token::Word("с".to_string()) },
            PositionalToken { offset: 767, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 768, length: 24, token: Token::Word("уверенностью".to_string()) },
            PositionalToken { offset: 792, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 793, length: 30, token: Token::Word("преимущественно".to_string()) },
            PositionalToken { offset: 823, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 824, length: 16, token: Token::Word("домашний".to_string()) },
            PositionalToken { offset: 840, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 841, length: 10, token: Token::Word("образ".to_string()) },
            PositionalToken { offset: 851, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 852, length: 10, token: Token::Word("жизни".to_string()) },
            PositionalToken { offset: 862, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 863, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 864, length: 3, token: Token::BBCode { left: vec![
                ], right: vec![
                ] } },
            ];
        let lib_res = uws.complex_tokens().collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        check_results(&result,&lib_res,uws);        
    }


    /*#[test]
    fn html() {
        let uws = "<div class=\"article article_view \" id=\"article_view_-113039156_9551\" data-article-url=\"/@chaibuket-o-chem-ne-zabyt-25-noyabrya\" data-audio-context=\"article:-113039156_9551\"><h1  class=\"article_decoration_first article_decoration_last\" >День Мамы </h1><p  class=\"article_decoration_first article_decoration_last\" >День, когда поздравляют мам, бабушек, сестер и жён — это всемирный праздник, называемый «День Мамы». В настоящее время его отмечают почти в каждой стране, просто везде разные даты и способы празднования. </p><h3  class=\"article_decoration_first article_decoration_last\" ><span class='article_anchor_title'>\n  <span class='article_anchor_button' id='pochemu-my-ego-prazdnuem'></span>\n  <span class='article_anchor_fsymbol'>П</span>\n</span>ПОЧЕМУ МЫ ЕГО ПРАЗДНУЕМ</h3><p  class=\"article_decoration_first article_decoration_last article_decoration_before\" >В 1987 году комитет госдумы по делам женщин, семьи и молодежи выступил с предложением учредить «День мамы», а сам приказ был подписан уже 30 января 1988 года Борисом Ельциным. Было решено, что ежегодно в России празднество дня мамы будет выпадать на последнее воскресенье ноября. </p><figure data-type=\"101\" data-mode=\"\"  class=\"article_decoration_first article_decoration_last\" >\n  <div class=\"article_figure_content\" style=\"width: 1125px\">\n    <div class=\"article_figure_sizer_content\"><div class=\"article_object_sizer_wrap\" data-sizes=\"[{&quot;s&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c0ffd/pcNJaBH3NDo.jpg&quot;,75,50],&quot;m&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c0ffe/ozCLs2kHtRY.jpg&quot;,130,87],&quot;x&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c0fff/E4KtTNDydzE.jpg&quot;,604,403],&quot;y&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1000/1nLxpYKavzU.jpg&quot;,807,538],&quot;z&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1001/IgEODe90yEk.jpg&quot;,1125,750],&quot;o&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1002/01faNwVZ2_E.jpg&quot;,130,87],&quot;p&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1003/baDFzbdRP2s.jpg&quot;,200,133],&quot;q&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1004/CY4khI6KJKA.jpg&quot;,320,213],&quot;r&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1005/NOvAJ6-VltY.jpg&quot;,510,340]}]\">\n  <img class=\"article_object_sizer_inner article_object_photo__image_blur\" src=\"https://pp.userapi.com/c849128/v849128704/c0ffd/pcNJaBH3NDo.jpg\" data-baseurl=\"\"/>\n  \n</div></div>\n    <div class=\"article_figure_sizer\" style=\"padding-bottom: 66.666666666667%\"></div>";
        let result = vec![
            PositionalToken { offset: 236, length: 8, token: Token::Word("День".to_string()) },
            PositionalToken { offset: 244, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 245, length: 8, token: Token::Word("Мамы".to_string()) },
            PositionalToken { offset: 253, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 321, length: 8, token: Token::Word("День".to_string()) },
            PositionalToken { offset: 329, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 330, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 331, length: 10, token: Token::Word("когда".to_string()) },
            PositionalToken { offset: 341, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 342, length: 22, token: Token::Word("поздравляют".to_string()) },
            PositionalToken { offset: 364, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 365, length: 6, token: Token::Word("мам".to_string()) },
            PositionalToken { offset: 371, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 372, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 373, length: 14, token: Token::Word("бабушек".to_string()) },
            PositionalToken { offset: 387, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 388, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 389, length: 12, token: Token::Word("сестер".to_string()) },
            PositionalToken { offset: 401, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 402, length: 2, token: Token::Word("и".to_string()) },
            PositionalToken { offset: 404, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 405, length: 6, token: Token::Word("жён".to_string()) },
            PositionalToken { offset: 411, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 412, length: 3, token: Token::Unicode("u2014".to_string()) },
            PositionalToken { offset: 415, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 416, length: 6, token: Token::Word("это".to_string()) },
            PositionalToken { offset: 422, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 423, length: 18, token: Token::Word("всемирный".to_string()) },
            PositionalToken { offset: 441, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 442, length: 16, token: Token::Word("праздник".to_string()) },
            PositionalToken { offset: 458, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 459, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 460, length: 20, token: Token::Word("называемый".to_string()) },
            PositionalToken { offset: 480, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 481, length: 2, token: Token::Unicode("uab".to_string()) },
            PositionalToken { offset: 483, length: 8, token: Token::Word("День".to_string()) },
            PositionalToken { offset: 491, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 492, length: 8, token: Token::Word("Мамы".to_string()) },
            PositionalToken { offset: 500, length: 2, token: Token::Unicode("ubb".to_string()) },
            PositionalToken { offset: 502, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 503, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 504, length: 2, token: Token::Word("В".to_string()) },
            PositionalToken { offset: 506, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 507, length: 18, token: Token::Word("настоящее".to_string()) },
            PositionalToken { offset: 525, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 526, length: 10, token: Token::Word("время".to_string()) },
            PositionalToken { offset: 536, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 537, length: 6, token: Token::Word("его".to_string()) },
            PositionalToken { offset: 543, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 544, length: 16, token: Token::Word("отмечают".to_string()) },
            PositionalToken { offset: 560, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 561, length: 10, token: Token::Word("почти".to_string()) },
            PositionalToken { offset: 571, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 572, length: 2, token: Token::Word("в".to_string()) },
            PositionalToken { offset: 574, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 575, length: 12, token: Token::Word("каждой".to_string()) },
            PositionalToken { offset: 587, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 588, length: 12, token: Token::Word("стране".to_string()) },
            PositionalToken { offset: 600, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 601, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 602, length: 12, token: Token::Word("просто".to_string()) },
            PositionalToken { offset: 614, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 615, length: 10, token: Token::Word("везде".to_string()) },
            PositionalToken { offset: 625, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 626, length: 12, token: Token::Word("разные".to_string()) },
            PositionalToken { offset: 638, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 639, length: 8, token: Token::Word("даты".to_string()) },
            PositionalToken { offset: 647, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 648, length: 2, token: Token::Word("и".to_string()) },
            PositionalToken { offset: 650, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 651, length: 14, token: Token::Word("способы".to_string()) },
            PositionalToken { offset: 665, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 666, length: 24, token: Token::Word("празднования".to_string()) },
            PositionalToken { offset: 690, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 691, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 794, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 795, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 870, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 871, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 910, length: 2, token: Token::Word("П".to_string()) },
            PositionalToken { offset: 919, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 927, length: 12, token: Token::Word("ПОЧЕМУ".to_string()) },
            PositionalToken { offset: 939, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 940, length: 4, token: Token::Word("МЫ".to_string()) },
            PositionalToken { offset: 944, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 945, length: 6, token: Token::Word("ЕГО".to_string()) },
            PositionalToken { offset: 951, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 952, length: 18, token: Token::Word("ПРАЗДНУЕМ".to_string()) },
            PositionalToken { offset: 1063, length: 2, token: Token::Word("В".to_string()) },
            PositionalToken { offset: 1065, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1066, length: 4, token: Token::Number(Number::Integer(1987)) },
            PositionalToken { offset: 1070, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1071, length: 8, token: Token::Word("году".to_string()) },
            PositionalToken { offset: 1079, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1080, length: 14, token: Token::Word("комитет".to_string()) },
            PositionalToken { offset: 1094, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1095, length: 14, token: Token::Word("госдумы".to_string()) },
            PositionalToken { offset: 1109, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1110, length: 4, token: Token::Word("по".to_string()) },
            PositionalToken { offset: 1114, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1115, length: 10, token: Token::Word("делам".to_string()) },
            PositionalToken { offset: 1125, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1126, length: 12, token: Token::Word("женщин".to_string()) },
            PositionalToken { offset: 1138, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 1139, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1140, length: 10, token: Token::Word("семьи".to_string()) },
            PositionalToken { offset: 1150, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1151, length: 2, token: Token::Word("и".to_string()) },
            PositionalToken { offset: 1153, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1154, length: 16, token: Token::Word("молодежи".to_string()) },
            PositionalToken { offset: 1170, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1171, length: 16, token: Token::Word("выступил".to_string()) },
            PositionalToken { offset: 1187, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1188, length: 2, token: Token::Word("с".to_string()) },
            PositionalToken { offset: 1190, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1191, length: 24, token: Token::Word("предложением".to_string()) },
            PositionalToken { offset: 1215, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1216, length: 16, token: Token::Word("учредить".to_string()) },
            PositionalToken { offset: 1232, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1233, length: 2, token: Token::Unicode("uab".to_string()) },
            PositionalToken { offset: 1235, length: 8, token: Token::Word("День".to_string()) },
            PositionalToken { offset: 1243, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1244, length: 8, token: Token::Word("мамы".to_string()) },
            PositionalToken { offset: 1252, length: 2, token: Token::Unicode("ubb".to_string()) },
            PositionalToken { offset: 1254, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 1255, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1256, length: 2, token: Token::Word("а".to_string()) },
            PositionalToken { offset: 1258, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1259, length: 6, token: Token::Word("сам".to_string()) },
            PositionalToken { offset: 1265, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1266, length: 12, token: Token::Word("приказ".to_string()) },
            PositionalToken { offset: 1278, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1279, length: 6, token: Token::Word("был".to_string()) },
            PositionalToken { offset: 1285, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1286, length: 16, token: Token::Word("подписан".to_string()) },
            PositionalToken { offset: 1302, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1303, length: 6, token: Token::Word("уже".to_string()) },
            PositionalToken { offset: 1309, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1310, length: 2, token: Token::Number(Number::Integer(30)) },
            PositionalToken { offset: 1312, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1313, length: 12, token: Token::Word("января".to_string()) },
            PositionalToken { offset: 1325, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1326, length: 4, token: Token::Number(Number::Integer(1988)) },
            PositionalToken { offset: 1330, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1331, length: 8, token: Token::Word("года".to_string()) },
            PositionalToken { offset: 1339, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1340, length: 14, token: Token::Word("Борисом".to_string()) },
            PositionalToken { offset: 1354, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1355, length: 16, token: Token::Word("Ельциным".to_string()) },
            PositionalToken { offset: 1371, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 1372, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1373, length: 8, token: Token::Word("Было".to_string()) },
            PositionalToken { offset: 1381, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1382, length: 12, token: Token::Word("решено".to_string()) },
            PositionalToken { offset: 1394, length: 1, token: Token::Punctuation(",".to_string()) },
            PositionalToken { offset: 1395, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1396, length: 6, token: Token::Word("что".to_string()) },
            PositionalToken { offset: 1402, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1403, length: 16, token: Token::Word("ежегодно".to_string()) },
            PositionalToken { offset: 1419, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1420, length: 2, token: Token::Word("в".to_string()) },
            PositionalToken { offset: 1422, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1423, length: 12, token: Token::Word("России".to_string()) },
            PositionalToken { offset: 1435, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1436, length: 22, token: Token::Word("празднество".to_string()) },
            PositionalToken { offset: 1458, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1459, length: 6, token: Token::Word("дня".to_string()) },
            PositionalToken { offset: 1465, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1466, length: 8, token: Token::Word("мамы".to_string()) },
            PositionalToken { offset: 1474, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1475, length: 10, token: Token::Word("будет".to_string()) },
            PositionalToken { offset: 1485, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1486, length: 16, token: Token::Word("выпадать".to_string()) },
            PositionalToken { offset: 1502, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1503, length: 4, token: Token::Word("на".to_string()) },
            PositionalToken { offset: 1507, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1508, length: 18, token: Token::Word("последнее".to_string()) },
            PositionalToken { offset: 1526, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1527, length: 22, token: Token::Word("воскресенье".to_string()) },
            PositionalToken { offset: 1549, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1550, length: 12, token: Token::Word("ноября".to_string()) },
            PositionalToken { offset: 1562, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 1563, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1664, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 1665, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 1725, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 1726, length: 4, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 2725, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 2726, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 2888, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 2889, length: 2, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 2891, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 2904, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 2905, length: 4, token: Token::Separator(Separator::Space) },
            ];
        match uws.into_tokens() {
            Err(Untokenizable::Html) => {},
            _ => panic!("Untokenizable::Html"),
        }
        //let lib_res = uws.into_tokens().unwrap().collect::<Vec<_>>();
        //check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!("")
    }*/

    #[test]
    fn vk_bbcode() {
        let uws = "[club113623432|💜💜💜 - для девушек] \n[club113623432|💛💛💛 - для сохраненок]";
        let result = vec![
            PositionalToken { offset: 0, length: 52, token: Token::BBCode { left: vec![
                PositionalToken { offset: 1, length: 13, token: Token::Numerical(Numerical::Alphanumeric("club113623432".to_string())) },
                ], right: vec![
                PositionalToken { offset: 15, length: 4, token: Token::Emoji("purple_heart") },
                PositionalToken { offset: 19, length: 4, token: Token::Emoji("purple_heart") },
                PositionalToken { offset: 23, length: 4, token: Token::Emoji("purple_heart") },
                PositionalToken { offset: 27, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 28, length: 1, token: Token::Punctuation("-".to_string()) },
                PositionalToken { offset: 29, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 30, length: 6, token: Token::Word("для".to_string()) },
                PositionalToken { offset: 36, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 37, length: 14, token: Token::Word("девушек".to_string()) },
                ] } },
            PositionalToken { offset: 52, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 53, length: 1, token: Token::Separator(Separator::Newline) },
            PositionalToken { offset: 54, length: 58, token: Token::BBCode { left: vec![
                PositionalToken { offset: 55, length: 13, token: Token::Numerical(Numerical::Alphanumeric("club113623432".to_string())) },
                ], right: vec![
                PositionalToken { offset: 69, length: 4, token: Token::Emoji("yellow_heart") },
                PositionalToken { offset: 73, length: 4, token: Token::Emoji("yellow_heart") },
                PositionalToken { offset: 77, length: 4, token: Token::Emoji("yellow_heart") },
                PositionalToken { offset: 81, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 82, length: 1, token: Token::Punctuation("-".to_string()) },
                PositionalToken { offset: 83, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 84, length: 6, token: Token::Word("для".to_string()) },
                PositionalToken { offset: 90, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 91, length: 20, token: Token::Word("сохраненок".to_string()) },
                ] } },
            ];
        let lib_res = uws.complex_tokens().collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        check_results(&result,&lib_res,uws);
    }

    /*#[test]
    fn text_href_and_html () {
        let uws = "https://youtu.be/dQErLQZw3qA</a></p><figure data-type=\"102\" data-mode=\"\"  class=\"article_decoration_first article_decoration_last\" >\n";
        let result =  vec![
            PositionalToken { offset: 0, length: 28, token: Token::Url("https://youtu.be/dQErLQZw3qA".to_string()) },
            PositionalToken { offset: 132, length: 1, token: Token::Separator(Separator::Newline) },
            ];
        let lib_res = uws.into_tokens().unwrap().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!("")
    }*/

    #[test]
    fn numerical_no_split() {
        let uws = "12.02.18 31.28.34 23.11.2018 123.568.365.234.578 127.0.0.1 1st 1кг 123123афываыв 12321фвафыов234выалфо 12_123_343.4234_4234";
        let lib_res = uws.into_tokens_with_options(BTreeSet::new()).collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        let result = vec![
            PositionalToken { offset: 0, length: 8, token: Token::Numerical(Numerical::DotSeparated("12.02.18".to_string())) },
            PositionalToken { offset: 8, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 9, length: 8, token: Token::Numerical(Numerical::DotSeparated("31.28.34".to_string())) },
            PositionalToken { offset: 17, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 18, length: 10, token: Token::Numerical(Numerical::DotSeparated("23.11.2018".to_string())) },
            PositionalToken { offset: 28, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 29, length: 19, token: Token::Numerical(Numerical::DotSeparated("123.568.365.234.578".to_string())) },
            PositionalToken { offset: 48, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 49, length: 9, token: Token::Numerical(Numerical::DotSeparated("127.0.0.1".to_string())) },
            PositionalToken { offset: 58, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 59, length: 3, token: Token::Numerical(Numerical::Measures("1st".to_string())) },
            PositionalToken { offset: 62, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 63, length: 5, token: Token::Numerical(Numerical::Measures("1кг".to_string())) },
            PositionalToken { offset: 68, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 69, length: 20, token: Token::Numerical(Numerical::Measures("123123афываыв".to_string())) },
            PositionalToken { offset: 89, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 90, length: 34, token: Token::Numerical(Numerical::Alphanumeric("12321фвафыов234выалфо".to_string())) },
            PositionalToken { offset: 124, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 125, length: 20, token: Token::Numerical(Numerical::Alphanumeric("12_123_343.4234_4234".to_string())) },
            ];
        check_results(&result,&lib_res,uws);
       
    }

    #[test]
    fn numerical_default() {
        let uws = "12.02.18 31.28.34 23.11.2018 123.568.365.234.578 127.0.0.1 1st 1кг 123123афываыв 12321фвафыов234выалфо 12_123_343.4234_4234";
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        let result = vec![
            PositionalToken { offset: 0, length: 2, token: Token::Number(Number::Integer(12)) },
            PositionalToken { offset: 2, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 3, length: 2, token: Token::Number(Number::Integer(2)) },
            PositionalToken { offset: 5, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 6, length: 2, token: Token::Number(Number::Integer(18)) },
            PositionalToken { offset: 8, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 9, length: 2, token: Token::Number(Number::Integer(31)) },
            PositionalToken { offset: 11, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 12, length: 2, token: Token::Number(Number::Integer(28)) },
            PositionalToken { offset: 14, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 15, length: 2, token: Token::Number(Number::Integer(34)) },
            PositionalToken { offset: 17, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 18, length: 2, token: Token::Number(Number::Integer(23)) },
            PositionalToken { offset: 20, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 21, length: 2, token: Token::Number(Number::Integer(11)) },
            PositionalToken { offset: 23, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 24, length: 4, token: Token::Number(Number::Integer(2018)) },
            PositionalToken { offset: 28, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 29, length: 3, token: Token::Number(Number::Integer(123)) },
            PositionalToken { offset: 32, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 33, length: 3, token: Token::Number(Number::Integer(568)) },
            PositionalToken { offset: 36, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 37, length: 3, token: Token::Number(Number::Integer(365)) },
            PositionalToken { offset: 40, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 41, length: 3, token: Token::Number(Number::Integer(234)) },
            PositionalToken { offset: 44, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 45, length: 3, token: Token::Number(Number::Integer(578)) },
            PositionalToken { offset: 48, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 49, length: 3, token: Token::Number(Number::Integer(127)) },
            PositionalToken { offset: 52, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 53, length: 1, token: Token::Number(Number::Integer(0)) },
            PositionalToken { offset: 54, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 55, length: 1, token: Token::Number(Number::Integer(0)) },
            PositionalToken { offset: 56, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 57, length: 1, token: Token::Number(Number::Integer(1)) },
            PositionalToken { offset: 58, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 59, length: 3, token: Token::Numerical(Numerical::Measures("1st".to_string())) },
            PositionalToken { offset: 62, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 63, length: 5, token: Token::Numerical(Numerical::Measures("1кг".to_string())) },
            PositionalToken { offset: 68, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 69, length: 20, token: Token::Numerical(Numerical::Measures("123123афываыв".to_string())) },
            PositionalToken { offset: 89, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 90, length: 34, token: Token::Numerical(Numerical::Alphanumeric("12321фвафыов234выалфо".to_string())) },
            PositionalToken { offset: 124, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { offset: 125, length: 2, token: Token::Number(Number::Integer(12)) },
            PositionalToken { offset: 127, length: 1, token: Token::Punctuation("_".to_string()) },
            PositionalToken { offset: 128, length: 3, token: Token::Number(Number::Integer(123)) },
            PositionalToken { offset: 131, length: 1, token: Token::Punctuation("_".to_string()) },
            PositionalToken { offset: 132, length: 3, token: Token::Number(Number::Integer(343)) },
            PositionalToken { offset: 135, length: 1, token: Token::Punctuation(".".to_string()) },
            PositionalToken { offset: 136, length: 4, token: Token::Number(Number::Integer(4234)) },
            PositionalToken { offset: 140, length: 1, token: Token::Punctuation("_".to_string()) },
            PositionalToken { offset: 141, length: 4, token: Token::Number(Number::Integer(4234)) },
            ];
        check_results(&result,&lib_res,uws);
       
    }

        /*#[test]
    fn new_test() {
        let uws = "";
        let lib_res = uws.into_tokens().unwrap().collect::<Vec<_>>();
        print_result(&lib_res); panic!("");
        let result = vec![];
        check_results(&result,&lib_res,uws);
        
}*/




    /* Language tests */

    enum Lang {
        Zho,
        Jpn,
        Kor,
        Ara,
        Ell,
    }

    #[test]
    fn test_lang_zho() {
        let (uws,result) = get_lang_test(Lang::Zho);
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,&uws);
    }

    #[test]
    fn test_lang_jpn() {
        let (uws,result) = get_lang_test(Lang::Jpn);
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,&uws);
    }

    #[test]
    fn test_lang_kor() {
        let (uws,result) = get_lang_test(Lang::Kor);
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,&uws);
    }

    #[test]
    fn test_lang_ara() {
        let (uws,result) = get_lang_test(Lang::Ara);
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,&uws);
    }

    #[test]
    fn test_lang_ell() {
        let (uws,result) = get_lang_test(Lang::Ell);
        let lib_res = uws.into_tokens().collect::<Vec<_>>();
        check_results(&result,&lib_res,&uws);
    }

    fn get_lang_test(lng: Lang) -> (String, Vec<PositionalToken>) {
        let text = match lng {
            Lang::Zho => "美国电视连续剧《超人前传》的第一集《试播集》于2001年10月16日在電視網首播，剧集主创人阿尔弗雷德·高夫和迈尔斯·米勒編劇，大卫·努特尔执导。这一试播首次向观众引荐了克拉克·肯特一角，他是位拥有超能力的外星孤儿，与家人和朋友一起在堪薩斯州虚构小镇斯莫维尔生活。在这一集里，肯特首度得知自己的来历，同时还需要阻止一位学生试图杀死镇上高中多名学生的报复之举。本集节目里引入了多个之后将贯穿全季甚至整部剧集的主题元素，例如几位主要角色之间的三角恋情。电视剧在加拿大溫哥華取景，旨在选用其“美国中产阶级”景观，主创人花了5个月的时间专门用于为主角物色合适的演员。试播集在所有演员选好4天后正式开拍。由于时间上的限制，剧组无法搭建好实体外景，因此只能使用计算机绘图技术将数字化的外景插入到镜头中。节目一经上映就打破了电视网的多项收视纪录，并且获得了评论员的普遍好评和多个奖项提名，并在其中两项上胜出",
            Lang::Kor =>  "플레이스테이션 은 소니 컴퓨터 엔터테인먼트가 개발한 세 번째 가정용 게임기이다. 마이크로소프트의 엑스박스 360, 닌텐도의 Wii와 경쟁하고 있다. 이전 제품에서 온라인 플레이 기능을 비디오 게임 개발사에 전적으로 의존하던 것과 달리 통합 온라인 게임 서비스인 플레이스테이션 네트워크 서비스를 발매와 함께 시작해 제공하고 있으며, 탄탄한 멀티미디어 재생 기능, 플레이스테이션 포터블과의 연결, 고화질 광학 디스크 포맷인 블루레이 디스크 재생 기능 등의 기능을 갖추고 있다. 2006년 11월 11일에 일본에서 처음으로 출시했으며, 11월 17일에는 북미 지역, 2007년 3월 23일에는 유럽과 오세아니아 지역에서, 대한민국의 경우 6월 5일부터 일주일간 예약판매를 실시해, 매일 준비한 수량이 동이 나는 등 많은 관심을 받았으며 6월 16일에 정식 출시 행사를 열었다",
            Lang::Jpn => "熊野三山本願所は、15世紀末以降における熊野三山（熊野本宮、熊野新宮、熊野那智）の造営・修造のための勧進を担った組織の総称。 熊野三山を含めて、日本における古代から中世前半にかけての寺社の造営は、寺社領経営のような恒常的財源、幕府や朝廷などからの一時的な造営料所の寄進、あるいは公権力からの臨時の保護によって行われていた。しかしながら、熊野三山では、これらの財源はすべて15世紀半ばまでに実効性を失った",
            Lang::Ara => "لشکرکشی‌های روس‌های وارنگی به دریای خزر مجموعه‌ای از حملات نظامی در بین سال‌های ۸۶۴ تا ۱۰۴۱ میلادی به سواحل دریای خزر بوده‌است. روس‌های وارنگی ابتدا در قرن نهم میلادی به عنوان بازرگانان پوست، عسل و برده در سرزمین‌های اسلامی(سرکلند) ظاهر شدند. این بازرگانان در مسیر تجاری ولگا به خرید و فروش می‌پرداختند. نخستین حملهٔ آنان در فاصله سال‌های ۸۶۴ تا ۸۸۴ میلادی در مقیاسی کوچک علیه علویان طبرستان رخ داد. نخستین یورش بزرگ روس‌ها در سال ۹۱۳ رخ داد و آنان با ۵۰۰ فروند درازکشتی شهر گرگان و اطراف آن را غارت کردند. آن‌ها در این حمله مقداری کالا و برده را به تاراج بردند و در راه بازگشتن به سمت شمال، در دلتای ولگا، مورد حملهٔ خزرهای مسلمان قرار گرفتند و بعضی از آنان موفق به فرار شدند، ولی در میانهٔ ولگا به قتل رسیدند. دومین هجوم بزرگ روس‌ها به دریای خزر در سال ۹۴۳ به وقوع پیوست. در این دوره ایگور یکم، حاکم روس کیف، رهبری روس‌ها را در دست داشت. روس‌ها پس از توافق با دولت خزرها برای عبور امن از منطقه، تا رود کورا و اعماق قفقاز پیش رفتند و در سال ۹۴۳ موفق شدند بندر بردعه، پایتخت اران (جمهوری آذربایجان کنونی)، را تصرف کنند. روس‌ها در آنجا به مدت چند ماه ماندند و بسیاری از ساکنان شهر را کشتند و از راه غارت‌گری اموالی را به تاراج بردند. تنها دلیل بازگشت آنان ",
            Lang::Ell => "Το Πρόγραμμα υλοποιείται εξ ολοκλήρου από απόσταση και μπορεί να συμμετέχει κάθε εμπλεκόμενος στη ή/και ενδιαφερόμενος για τη διδασκαλία της Ελληνικής ως δεύτερης/ξένης γλώσσας στην Ελλάδα και στο εξωτερικό, αρκεί να είναι απόφοιτος ελληνικής φιλολογίας, ξένων φιλολογιών, παιδαγωγικών τμημάτων, θεολογικών σχολών ή άλλων πανεπιστημιακών τμημάτων ελληνικών ή ισότιμων ξένων πανεπιστημίων. Υπό όρους γίνονται δεκτοί υποψήφιοι που δεν έχουν ολοκληρώσει σπουδές τριτοβάθμιας εκπαίδευσης.",
        }.chars().take(100).fold(String::new(),|acc,c| acc + &format!("{}",c));
        let tokens = match lng {
            Lang::Zho => vec![
                PositionalToken { offset: 0, length: 3, token: Token::Word("美".to_string()) },
                PositionalToken { offset: 3, length: 3, token: Token::Word("国".to_string()) },
                PositionalToken { offset: 6, length: 3, token: Token::Word("电".to_string()) },
                PositionalToken { offset: 9, length: 3, token: Token::Word("视".to_string()) },
                PositionalToken { offset: 12, length: 3, token: Token::Word("连".to_string()) },
                PositionalToken { offset: 15, length: 3, token: Token::Word("续".to_string()) },
                PositionalToken { offset: 18, length: 3, token: Token::Word("剧".to_string()) },
                PositionalToken { offset: 21, length: 3, token: Token::Punctuation("《".to_string()) },
                PositionalToken { offset: 24, length: 3, token: Token::Word("超".to_string()) },
                PositionalToken { offset: 27, length: 3, token: Token::Word("人".to_string()) },
                PositionalToken { offset: 30, length: 3, token: Token::Word("前".to_string()) },
                PositionalToken { offset: 33, length: 3, token: Token::Word("传".to_string()) },
                PositionalToken { offset: 36, length: 3, token: Token::Punctuation("》".to_string()) },
                PositionalToken { offset: 39, length: 3, token: Token::Word("的".to_string()) },
                PositionalToken { offset: 42, length: 3, token: Token::Word("第".to_string()) },
                PositionalToken { offset: 45, length: 3, token: Token::Word("一".to_string()) },
                PositionalToken { offset: 48, length: 3, token: Token::Word("集".to_string()) },
                PositionalToken { offset: 51, length: 3, token: Token::Punctuation("《".to_string()) },
                PositionalToken { offset: 54, length: 3, token: Token::Word("试".to_string()) },
                PositionalToken { offset: 57, length: 3, token: Token::Word("播".to_string()) },
                PositionalToken { offset: 60, length: 3, token: Token::Word("集".to_string()) },
                PositionalToken { offset: 63, length: 3, token: Token::Punctuation("》".to_string()) },
                PositionalToken { offset: 66, length: 3, token: Token::Word("于".to_string()) },
                PositionalToken { offset: 69, length: 4, token: Token::Number(Number::Integer(2001)) },
                PositionalToken { offset: 73, length: 3, token: Token::Word("年".to_string()) },
                PositionalToken { offset: 76, length: 2, token: Token::Number(Number::Integer(10)) },
                PositionalToken { offset: 78, length: 3, token: Token::Word("月".to_string()) },
                PositionalToken { offset: 81, length: 2, token: Token::Number(Number::Integer(16)) },
                PositionalToken { offset: 83, length: 3, token: Token::Word("日".to_string()) },
                PositionalToken { offset: 86, length: 3, token: Token::Word("在".to_string()) },
                PositionalToken { offset: 89, length: 3, token: Token::Word("電".to_string()) },
                PositionalToken { offset: 92, length: 3, token: Token::Word("視".to_string()) },
                PositionalToken { offset: 95, length: 3, token: Token::Word("網".to_string()) },
                PositionalToken { offset: 98, length: 3, token: Token::Word("首".to_string()) },
                PositionalToken { offset: 101, length: 3, token: Token::Word("播".to_string()) },
                PositionalToken { offset: 104, length: 3, token: Token::Punctuation("，".to_string()) },
                PositionalToken { offset: 107, length: 3, token: Token::Word("剧".to_string()) },
                PositionalToken { offset: 110, length: 3, token: Token::Word("集".to_string()) },
                PositionalToken { offset: 113, length: 3, token: Token::Word("主".to_string()) },
                PositionalToken { offset: 116, length: 3, token: Token::Word("创".to_string()) },
                PositionalToken { offset: 119, length: 3, token: Token::Word("人".to_string()) },
                PositionalToken { offset: 122, length: 3, token: Token::Word("阿".to_string()) },
                PositionalToken { offset: 125, length: 3, token: Token::Word("尔".to_string()) },
                PositionalToken { offset: 128, length: 3, token: Token::Word("弗".to_string()) },
                PositionalToken { offset: 131, length: 3, token: Token::Word("雷".to_string()) },
                PositionalToken { offset: 134, length: 3, token: Token::Word("德".to_string()) },
                PositionalToken { offset: 137, length: 2, token: Token::Punctuation("·".to_string()) },
                PositionalToken { offset: 139, length: 3, token: Token::Word("高".to_string()) },
                PositionalToken { offset: 142, length: 3, token: Token::Word("夫".to_string()) },
                PositionalToken { offset: 145, length: 3, token: Token::Word("和".to_string()) },
                PositionalToken { offset: 148, length: 3, token: Token::Word("迈".to_string()) },
                PositionalToken { offset: 151, length: 3, token: Token::Word("尔".to_string()) },
                PositionalToken { offset: 154, length: 3, token: Token::Word("斯".to_string()) },
                PositionalToken { offset: 157, length: 2, token: Token::Punctuation("·".to_string()) },
                PositionalToken { offset: 159, length: 3, token: Token::Word("米".to_string()) },
                PositionalToken { offset: 162, length: 3, token: Token::Word("勒".to_string()) },
                PositionalToken { offset: 165, length: 3, token: Token::Word("編".to_string()) },
                PositionalToken { offset: 168, length: 3, token: Token::Word("劇".to_string()) },
                PositionalToken { offset: 171, length: 3, token: Token::Punctuation("，".to_string()) },
                PositionalToken { offset: 174, length: 3, token: Token::Word("大".to_string()) },
                PositionalToken { offset: 177, length: 3, token: Token::Word("卫".to_string()) },
                PositionalToken { offset: 180, length: 2, token: Token::Punctuation("·".to_string()) },
                PositionalToken { offset: 182, length: 3, token: Token::Word("努".to_string()) },
                PositionalToken { offset: 185, length: 3, token: Token::Word("特".to_string()) },
                PositionalToken { offset: 188, length: 3, token: Token::Word("尔".to_string()) },
                PositionalToken { offset: 191, length: 3, token: Token::Word("执".to_string()) },
                PositionalToken { offset: 194, length: 3, token: Token::Word("导".to_string()) },
                PositionalToken { offset: 197, length: 3, token: Token::Punctuation("。".to_string()) },
                PositionalToken { offset: 200, length: 3, token: Token::Word("这".to_string()) },
                PositionalToken { offset: 203, length: 3, token: Token::Word("一".to_string()) },
                PositionalToken { offset: 206, length: 3, token: Token::Word("试".to_string()) },
                PositionalToken { offset: 209, length: 3, token: Token::Word("播".to_string()) },
                PositionalToken { offset: 212, length: 3, token: Token::Word("首".to_string()) },
                PositionalToken { offset: 215, length: 3, token: Token::Word("次".to_string()) },
                PositionalToken { offset: 218, length: 3, token: Token::Word("向".to_string()) },
                PositionalToken { offset: 221, length: 3, token: Token::Word("观".to_string()) },
                PositionalToken { offset: 224, length: 3, token: Token::Word("众".to_string()) },
                PositionalToken { offset: 227, length: 3, token: Token::Word("引".to_string()) },
                PositionalToken { offset: 230, length: 3, token: Token::Word("荐".to_string()) },
                PositionalToken { offset: 233, length: 3, token: Token::Word("了".to_string()) },
                PositionalToken { offset: 236, length: 3, token: Token::Word("克".to_string()) },
                PositionalToken { offset: 239, length: 3, token: Token::Word("拉".to_string()) },
                PositionalToken { offset: 242, length: 3, token: Token::Word("克".to_string()) },
                PositionalToken { offset: 245, length: 2, token: Token::Punctuation("·".to_string()) },
                PositionalToken { offset: 247, length: 3, token: Token::Word("肯".to_string()) },
                PositionalToken { offset: 250, length: 3, token: Token::Word("特".to_string()) },
                PositionalToken { offset: 253, length: 3, token: Token::Word("一".to_string()) },
                PositionalToken { offset: 256, length: 3, token: Token::Word("角".to_string()) },
                PositionalToken { offset: 259, length: 3, token: Token::Punctuation("，".to_string()) },
                PositionalToken { offset: 262, length: 3, token: Token::Word("他".to_string()) },
                PositionalToken { offset: 265, length: 3, token: Token::Word("是".to_string()) },
                PositionalToken { offset: 268, length: 3, token: Token::Word("位".to_string()) },
                PositionalToken { offset: 271, length: 3, token: Token::Word("拥".to_string()) },
                PositionalToken { offset: 274, length: 3, token: Token::Word("有".to_string()) },
                PositionalToken { offset: 277, length: 3, token: Token::Word("超".to_string()) },
                ],
            Lang::Jpn => vec![
                PositionalToken { offset: 0, length: 3, token: Token::Word("熊".to_string()) },
                PositionalToken { offset: 3, length: 3, token: Token::Word("野".to_string()) },
                PositionalToken { offset: 6, length: 3, token: Token::Word("三".to_string()) },
                PositionalToken { offset: 9, length: 3, token: Token::Word("山".to_string()) },
                PositionalToken { offset: 12, length: 3, token: Token::Word("本".to_string()) },
                PositionalToken { offset: 15, length: 3, token: Token::Word("願".to_string()) },
                PositionalToken { offset: 18, length: 3, token: Token::Word("所".to_string()) },
                PositionalToken { offset: 21, length: 3, token: Token::Word("は".to_string()) },
                PositionalToken { offset: 24, length: 3, token: Token::Punctuation("、".to_string()) },
                PositionalToken { offset: 27, length: 2, token: Token::Number(Number::Integer(15)) },
                PositionalToken { offset: 29, length: 3, token: Token::Word("世".to_string()) },
                PositionalToken { offset: 32, length: 3, token: Token::Word("紀".to_string()) },
                PositionalToken { offset: 35, length: 3, token: Token::Word("末".to_string()) },
                PositionalToken { offset: 38, length: 3, token: Token::Word("以".to_string()) },
                PositionalToken { offset: 41, length: 3, token: Token::Word("降".to_string()) },
                PositionalToken { offset: 44, length: 3, token: Token::Word("に".to_string()) },
                PositionalToken { offset: 47, length: 3, token: Token::Word("お".to_string()) },
                PositionalToken { offset: 50, length: 3, token: Token::Word("け".to_string()) },
                PositionalToken { offset: 53, length: 3, token: Token::Word("る".to_string()) },
                PositionalToken { offset: 56, length: 3, token: Token::Word("熊".to_string()) },
                PositionalToken { offset: 59, length: 3, token: Token::Word("野".to_string()) },
                PositionalToken { offset: 62, length: 3, token: Token::Word("三".to_string()) },
                PositionalToken { offset: 65, length: 3, token: Token::Word("山".to_string()) },
                PositionalToken { offset: 68, length: 3, token: Token::Punctuation("（".to_string()) },
                PositionalToken { offset: 71, length: 3, token: Token::Word("熊".to_string()) },
                PositionalToken { offset: 74, length: 3, token: Token::Word("野".to_string()) },
                PositionalToken { offset: 77, length: 3, token: Token::Word("本".to_string()) },
                PositionalToken { offset: 80, length: 3, token: Token::Word("宮".to_string()) },
                PositionalToken { offset: 83, length: 3, token: Token::Punctuation("、".to_string()) },
                PositionalToken { offset: 86, length: 3, token: Token::Word("熊".to_string()) },
                PositionalToken { offset: 89, length: 3, token: Token::Word("野".to_string()) },
                PositionalToken { offset: 92, length: 3, token: Token::Word("新".to_string()) },
                PositionalToken { offset: 95, length: 3, token: Token::Word("宮".to_string()) },
                PositionalToken { offset: 98, length: 3, token: Token::Punctuation("、".to_string()) },
                PositionalToken { offset: 101, length: 3, token: Token::Word("熊".to_string()) },
                PositionalToken { offset: 104, length: 3, token: Token::Word("野".to_string()) },
                PositionalToken { offset: 107, length: 3, token: Token::Word("那".to_string()) },
                PositionalToken { offset: 110, length: 3, token: Token::Word("智".to_string()) },
                PositionalToken { offset: 113, length: 3, token: Token::Punctuation("）".to_string()) },
                PositionalToken { offset: 116, length: 3, token: Token::Word("の".to_string()) },
                PositionalToken { offset: 119, length: 3, token: Token::Word("造".to_string()) },
                PositionalToken { offset: 122, length: 3, token: Token::Word("営".to_string()) },
                PositionalToken { offset: 125, length: 3, token: Token::Punctuation("・".to_string()) },
                PositionalToken { offset: 128, length: 3, token: Token::Word("修".to_string()) },
                PositionalToken { offset: 131, length: 3, token: Token::Word("造".to_string()) },
                PositionalToken { offset: 134, length: 3, token: Token::Word("の".to_string()) },
                PositionalToken { offset: 137, length: 3, token: Token::Word("た".to_string()) },
                PositionalToken { offset: 140, length: 3, token: Token::Word("め".to_string()) },
                PositionalToken { offset: 143, length: 3, token: Token::Word("の".to_string()) },
                PositionalToken { offset: 146, length: 3, token: Token::Word("勧".to_string()) },
                PositionalToken { offset: 149, length: 3, token: Token::Word("進".to_string()) },
                PositionalToken { offset: 152, length: 3, token: Token::Word("を".to_string()) },
                PositionalToken { offset: 155, length: 3, token: Token::Word("担".to_string()) },
                PositionalToken { offset: 158, length: 3, token: Token::Word("っ".to_string()) },
                PositionalToken { offset: 161, length: 3, token: Token::Word("た".to_string()) },
                PositionalToken { offset: 164, length: 3, token: Token::Word("組".to_string()) },
                PositionalToken { offset: 167, length: 3, token: Token::Word("織".to_string()) },
                PositionalToken { offset: 170, length: 3, token: Token::Word("の".to_string()) },
                PositionalToken { offset: 173, length: 3, token: Token::Word("総".to_string()) },
                PositionalToken { offset: 176, length: 3, token: Token::Word("称".to_string()) },
                PositionalToken { offset: 179, length: 3, token: Token::Punctuation("。".to_string()) },
                PositionalToken { offset: 182, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 183, length: 3, token: Token::Word("熊".to_string()) },
                PositionalToken { offset: 186, length: 3, token: Token::Word("野".to_string()) },
                PositionalToken { offset: 189, length: 3, token: Token::Word("三".to_string()) },
                PositionalToken { offset: 192, length: 3, token: Token::Word("山".to_string()) },
                PositionalToken { offset: 195, length: 3, token: Token::Word("を".to_string()) },
                PositionalToken { offset: 198, length: 3, token: Token::Word("含".to_string()) },
                PositionalToken { offset: 201, length: 3, token: Token::Word("め".to_string()) },
                PositionalToken { offset: 204, length: 3, token: Token::Word("て".to_string()) },
                PositionalToken { offset: 207, length: 3, token: Token::Punctuation("、".to_string()) },
                PositionalToken { offset: 210, length: 3, token: Token::Word("日".to_string()) },
                PositionalToken { offset: 213, length: 3, token: Token::Word("本".to_string()) },
                PositionalToken { offset: 216, length: 3, token: Token::Word("に".to_string()) },
                PositionalToken { offset: 219, length: 3, token: Token::Word("お".to_string()) },
                PositionalToken { offset: 222, length: 3, token: Token::Word("け".to_string()) },
                PositionalToken { offset: 225, length: 3, token: Token::Word("る".to_string()) },
                PositionalToken { offset: 228, length: 3, token: Token::Word("古".to_string()) },
                PositionalToken { offset: 231, length: 3, token: Token::Word("代".to_string()) },
                PositionalToken { offset: 234, length: 3, token: Token::Word("か".to_string()) },
                PositionalToken { offset: 237, length: 3, token: Token::Word("ら".to_string()) },
                PositionalToken { offset: 240, length: 3, token: Token::Word("中".to_string()) },
                PositionalToken { offset: 243, length: 3, token: Token::Word("世".to_string()) },
                PositionalToken { offset: 246, length: 3, token: Token::Word("前".to_string()) },
                PositionalToken { offset: 249, length: 3, token: Token::Word("半".to_string()) },
                PositionalToken { offset: 252, length: 3, token: Token::Word("に".to_string()) },
                PositionalToken { offset: 255, length: 3, token: Token::Word("か".to_string()) },
                PositionalToken { offset: 258, length: 3, token: Token::Word("け".to_string()) },
                PositionalToken { offset: 261, length: 3, token: Token::Word("て".to_string()) },
                PositionalToken { offset: 264, length: 3, token: Token::Word("の".to_string()) },
                PositionalToken { offset: 267, length: 3, token: Token::Word("寺".to_string()) },
                PositionalToken { offset: 270, length: 3, token: Token::Word("社".to_string()) },
                PositionalToken { offset: 273, length: 3, token: Token::Word("の".to_string()) },
                PositionalToken { offset: 276, length: 3, token: Token::Word("造".to_string()) },
                PositionalToken { offset: 279, length: 3, token: Token::Word("営".to_string()) },
                PositionalToken { offset: 282, length: 3, token: Token::Word("は".to_string()) },
                PositionalToken { offset: 285, length: 3, token: Token::Punctuation("、".to_string()) },
                PositionalToken { offset: 288, length: 3, token: Token::Word("寺".to_string()) },
                PositionalToken { offset: 291, length: 3, token: Token::Word("社".to_string()) },
                ],
            Lang::Kor => vec![
                PositionalToken { offset: 0, length: 21, token: Token::Word("플레이스테이션".to_string()) },
                PositionalToken { offset: 21, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 22, length: 3, token: Token::Word("은".to_string()) },
                PositionalToken { offset: 25, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 26, length: 6, token: Token::Word("소니".to_string()) },
                PositionalToken { offset: 32, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 33, length: 9, token: Token::Word("컴퓨터".to_string()) },
                PositionalToken { offset: 42, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 43, length: 21, token: Token::Word("엔터테인먼트가".to_string()) },
                PositionalToken { offset: 64, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 65, length: 9, token: Token::Word("개발한".to_string()) },
                PositionalToken { offset: 74, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 75, length: 3, token: Token::Word("세".to_string()) },
                PositionalToken { offset: 78, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 79, length: 6, token: Token::Word("번째".to_string()) },
                PositionalToken { offset: 85, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 86, length: 9, token: Token::Word("가정용".to_string()) },
                PositionalToken { offset: 95, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 96, length: 15, token: Token::Word("게임기이다".to_string()) },
                PositionalToken { offset: 111, length: 1, token: Token::Punctuation(".".to_string()) },
                PositionalToken { offset: 112, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 113, length: 24, token: Token::Word("마이크로소프트의".to_string()) },
                PositionalToken { offset: 137, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 138, length: 12, token: Token::Word("엑스박스".to_string()) },
                PositionalToken { offset: 150, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 151, length: 3, token: Token::Number(Number::Integer(360)) },
                PositionalToken { offset: 154, length: 1, token: Token::Punctuation(",".to_string()) },
                PositionalToken { offset: 155, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 156, length: 12, token: Token::Word("닌텐도의".to_string()) },
                PositionalToken { offset: 168, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 169, length: 6, token: Token::Word("Wii와".to_string()) },
                PositionalToken { offset: 175, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 176, length: 12, token: Token::Word("경쟁하고".to_string()) },
                PositionalToken { offset: 188, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 189, length: 6, token: Token::Word("있다".to_string()) },
                PositionalToken { offset: 195, length: 1, token: Token::Punctuation(".".to_string()) },
                PositionalToken { offset: 196, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 197, length: 6, token: Token::Word("이전".to_string()) },
                PositionalToken { offset: 203, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 204, length: 12, token: Token::Word("제품에서".to_string()) },
                PositionalToken { offset: 216, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 217, length: 9, token: Token::Word("온라인".to_string()) },
                PositionalToken { offset: 226, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 227, length: 9, token: Token::Word("플레이".to_string()) },
                PositionalToken { offset: 236, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 237, length: 3, token: Token::Word("기".to_string()) },
                ],
            Lang::Ara => vec![
                PositionalToken { offset: 0, length: 14, token: Token::Word("لشکرکشی".to_string()) },
                PositionalToken { offset: 14, length: 3, token: Token::UnicodeFormatter(Formatter::Char('\u{200c}')) },
                PositionalToken { offset: 17, length: 6, token: Token::Word("های".to_string()) },
                PositionalToken { offset: 23, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 24, length: 6, token: Token::Word("روس".to_string()) },
                PositionalToken { offset: 30, length: 3, token: Token::UnicodeFormatter(Formatter::Char('\u{200c}')) },
                PositionalToken { offset: 33, length: 6, token: Token::Word("های".to_string()) },
                PositionalToken { offset: 39, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 40, length: 12, token: Token::Word("وارنگی".to_string()) },
                PositionalToken { offset: 52, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 53, length: 4, token: Token::Word("به".to_string()) },
                PositionalToken { offset: 57, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 58, length: 10, token: Token::Word("دریای".to_string()) },
                PositionalToken { offset: 68, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 69, length: 6, token: Token::Word("خزر".to_string()) },
                PositionalToken { offset: 75, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 76, length: 12, token: Token::Word("مجموعه".to_string()) },
                PositionalToken { offset: 88, length: 3, token: Token::UnicodeFormatter(Formatter::Char('\u{200c}')) },
                PositionalToken { offset: 91, length: 4, token: Token::Word("ای".to_string()) },
                PositionalToken { offset: 95, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 96, length: 4, token: Token::Word("از".to_string()) },
                PositionalToken { offset: 100, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 101, length: 10, token: Token::Word("حملات".to_string()) },
                PositionalToken { offset: 111, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 112, length: 10, token: Token::Word("نظامی".to_string()) },
                PositionalToken { offset: 122, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 123, length: 4, token: Token::Word("در".to_string()) },
                PositionalToken { offset: 127, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 128, length: 6, token: Token::Word("بین".to_string()) },
                PositionalToken { offset: 134, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 135, length: 6, token: Token::Word("سال".to_string()) },
                PositionalToken { offset: 141, length: 3, token: Token::UnicodeFormatter(Formatter::Char('\u{200c}')) },
                PositionalToken { offset: 144, length: 6, token: Token::Word("های".to_string()) },
                PositionalToken { offset: 150, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 151, length: 6, token: Token::StrangeWord("۸۶۴".to_string()) },
                PositionalToken { offset: 157, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 158, length: 4, token: Token::Word("تا".to_string()) },
                PositionalToken { offset: 162, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 163, length: 8, token: Token::StrangeWord("۱۰۴۱".to_string()) },
                PositionalToken { offset: 171, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 172, length: 12, token: Token::Word("میلادی".to_string()) },
                PositionalToken { offset: 184, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 185, length: 2, token: Token::Word("ب".to_string()) },
                ],
            Lang::Ell => vec![
                PositionalToken { offset: 0, length: 4, token: Token::Word("Το".to_string()) },
                PositionalToken { offset: 4, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 5, length: 18, token: Token::Word("Πρόγραμμα".to_string()) },
                PositionalToken { offset: 23, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 24, length: 22, token: Token::Word("υλοποιείται".to_string()) },
                PositionalToken { offset: 46, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 47, length: 4, token: Token::Word("εξ".to_string()) },
                PositionalToken { offset: 51, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 52, length: 18, token: Token::Word("ολοκλήρου".to_string()) },
                PositionalToken { offset: 70, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 71, length: 6, token: Token::Word("από".to_string()) },
                PositionalToken { offset: 77, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 78, length: 16, token: Token::Word("απόσταση".to_string()) },
                PositionalToken { offset: 94, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 95, length: 6, token: Token::Word("και".to_string()) },
                PositionalToken { offset: 101, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 102, length: 12, token: Token::Word("μπορεί".to_string()) },
                PositionalToken { offset: 114, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 115, length: 4, token: Token::Word("να".to_string()) },
                PositionalToken { offset: 119, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 120, length: 20, token: Token::Word("συμμετέχει".to_string()) },
                PositionalToken { offset: 140, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 141, length: 8, token: Token::Word("κάθε".to_string()) },
                PositionalToken { offset: 149, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 150, length: 24, token: Token::Word("εμπλεκόμενος".to_string()) },
                PositionalToken { offset: 174, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 175, length: 6, token: Token::Word("στη".to_string()) },
                PositionalToken { offset: 181, length: 1, token: Token::Separator(Separator::Space) },
                PositionalToken { offset: 182, length: 2, token: Token::Word("ή".to_string()) },
                PositionalToken { offset: 184, length: 1, token: Token::Punctuation("/".to_string()) },
                ],
        };
        (text,tokens)
    }
}
 
