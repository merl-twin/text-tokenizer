use unicode_properties::{
    GeneralCategoryGroup,
    UnicodeGeneralCategory,
    GeneralCategory,
};
use std::str::FromStr;
use std::collections::{VecDeque,BTreeSet};

use text_parsing::Local;

use crate::{
    EMOJIMAP,
    Token, Numerical, Number, Formatter, Separator, Struct, Word, Special, Unicode,
    TokenizerOptions, TokenizerParams, IntoTokenizer, SentenceBreaker,
    wordbreaker::{
        BasicToken,
        WordBreaker,
        one_char_word,
    },
};


impl<'t> IntoTokenizer for &'t str {
    type IntoTokens = Tokens<'t>;
    
    fn into_tokenizer<S: SentenceBreaker>(self, params: TokenizerParams<S>) -> Self::IntoTokens {
        Tokens::new(self, &params.options)
    }
}

impl<'t> Iterator for Tokens<'t> {
    type Item = Local<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.buffer.len()>0 {                
                return self.next_from_buffer();
            } else {
                loop {
                    match self.bounds.next() {
                        Some(local_bt) => {
                            let sep = if let BasicToken::Separator(_) = local_bt.data() { true } else { false };
                            self.buffer.push_back(local_bt);
                            if sep {
                                return self.next();
                            }
                        },
                        None if self.buffer.len()>0 => return self.next(),
                        None => return None,
                    }
                }
            }
        }
    }
}

//#[derive(Debug)]
pub struct Tokens<'t> {
    bounds: WordBreaker<'t>,
    buffer: VecDeque<Local<BasicToken<'t>>>,
    allow_structs: bool,
}
impl<'t> Tokens<'t> {
    pub(crate) fn new<'a>(s: &'a str, options: &BTreeSet<TokenizerOptions>) -> Tokens<'a> {
        Tokens {
            bounds: WordBreaker::new(s, &options),
            buffer: VecDeque::new(),
            allow_structs: if options.contains(&TokenizerOptions::StructTokens) { true } else { false },
        }
    }
    fn basic_separator_to_pt(&mut self, s: &str) -> Token {
        Token::Special(Special::Separator(match s.chars().next() {
            Some(' ') => Separator::Space,
            Some('\n') => Separator::Newline,
            Some('\t') => Separator::Tab,
            Some(c) => Separator::Char(c),
            None => Separator::Unknown,
        }))
    }
    fn basic_formater_to_pt(&mut self, s: &str) -> Token {
        Token::Unicode(Unicode::Formatter(match s.chars().next() {
            Some('\u{200d}') => Formatter::Joiner,
            Some(c) => Formatter::Char(c),
            None => Formatter::Unknown,
        }))
    }   
    fn basic_number_to_pt(&mut self, s: &str) -> Token {
        Token::Word(match i64::from_str(s) {
            Ok(n) => Word::Number(Number::Integer(n)),
            Err(_) => {
                match f64::from_str(s) {
                    Ok(n) => Word::Number(Number::Float(n)),
                    Err(..) => {
                        #[cfg(feature = "strings")]
                        { Word::Word(s.to_string()) }
                        #[cfg(not(feature = "strings"))]
                        { Word::Word }  
                    },
                }
            }
        })
    }
    fn basic_mixed_to_pt(&mut self, s: &str) -> Token {
        let mut word = true;
        let mut has_word_parts = false;
        let mut first = true;
        let mut same = false;
        let mut one_c = ' ';
        for c in s.chars() {
            match c.is_alphanumeric() || c.is_digit(10) || (c.general_category_group() == GeneralCategoryGroup::Punctuation) || (c == '\u{0060}') {
                true => { has_word_parts = true; },
                false => { word = false; },
            }
            match first {
                true => { one_c = c; first = false; same = true; },
                false => if one_c != c { same = false; }
            }
        }
        if !first && same && (one_c.is_whitespace() || (one_c.general_category() == GeneralCategory::Format)) {
            if one_c.is_whitespace() {
                return self.basic_separator_to_pt(s);                       
            } else {
                return self.basic_formater_to_pt(s)
            }
        }
        if word {
            #[cfg(feature = "strings")]
            { Token::Word(Word::StrangeWord(s.to_string())) }
            #[cfg(not(feature = "strings"))]
            { Token::Word(Word::StrangeWord) }  
        } else {
            let rs = s.replace("\u{fe0f}","");
            match EMOJIMAP.get(&rs as &str) {
                Some(em) => Token::Word(Word::Emoji(em)),
                None => match one_char_word(&rs) {
                    //Some(c) if c.general_category() == GeneralCategory::ModifierSymbol => Token::UnicodeModifier(c),
                    Some(c) if c.general_category_group() == GeneralCategoryGroup::Symbol => Token::Special(Special::Symbol(c)),
                    Some(_) | None => match has_word_parts {
                        true => {
                            #[cfg(feature = "strings")]
                            { Token::Word(Word::StrangeWord(s.to_string())) }
                            #[cfg(not(feature = "strings"))]
                            { Token::Word(Word::StrangeWord) }  
                        },
                        false => Token::Unicode(Unicode::String({
                            let mut us = "".to_string();
                            for c in rs.chars() {
                                if us!="" { us += "_"; }
                                us += "u";
                                let ns = format!("{}",c.escape_unicode());
                                us += &ns[3 .. ns.len()-1];
                            }
                            us
                        })),
                    }
                },
            }
        }        
    }
    fn basic_alphanumeric_to_pt(&mut self, s: &str) -> Token {
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
        Token::Word(match (digits,digits_begin_only,dots,alphas_and_apos,other) {
            (true,false,true,false,false) => {
                // TODO: Date, Ip, DotSeparated
                Word::Numerical(Numerical::DotSeparated(s.to_string()))
            },
            (true,true,_,true,false) => {
                // TODO: Countable or Measures
                Word::Numerical(Numerical::Measures(s.to_string()))
            },
            (true, _, _, _, _) => {
                // Numerical trash, ids, etc.
                Word::Numerical(Numerical::Alphanumeric(s.to_string()))
            }
            (false,false,_,true,false) => {
                // Word
                #[cfg(feature = "strings")]
                { Word::Word(s.to_string()) }
                #[cfg(not(feature = "strings"))]
                { Word::Word }  
            },
            (false,false,_,_,_) => {
                // Strange                    
                #[cfg(feature = "strings")]
                { Word::StrangeWord(s.to_string()) }
                #[cfg(not(feature = "strings"))]
                { Word::StrangeWord }   
            },
            (false,true,_,_,_) => unreachable!(),
        })
    }
    fn basic_punctuation_to_pt(&mut self, s: &str) -> Token {
        #[cfg(feature = "strings")]
        { Token::Special(Special::Punctuation(s.to_string())) }
        #[cfg(not(feature = "strings"))]
        { Token::Special(Special::Punctuation) }
    }
    /*fn check_url(&mut self) -> Option<PositionalToken> {
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
            let tag_bound = None;
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
    }*/
    fn check_hashtag(&mut self) -> Option<Local<Token>> {
        if !self.allow_structs || (self.buffer.len() < 2) { return None; }

        let (loc1,s1) = self.buffer[0].into_inner();
        let (loc2,s2) = self.buffer[1].into_inner();
        match (s1,s2) {
            (BasicToken::Punctuation("#"),BasicToken::Alphanumeric(s)) |
            (BasicToken::Punctuation("#"),BasicToken::Number(s)) => match Local::from_segment(loc1,loc2) {
                Ok(local) => {
                    self.buffer.pop_front();
                    self.buffer.pop_front();
                    
                    Some(local.local(Token::Struct({
                        #[cfg(feature = "strings")]
                        { Struct::Hashtag(s.to_string()) }
                        #[cfg(not(feature = "strings"))]
                        { Struct::Hashtag }
                    })))
                },
                Err(_) => None,                    
            },
            _ => None,
        }
    }
    fn check_mention(&mut self) -> Option<Local<Token>> {
        if !self.allow_structs || (self.buffer.len() < 2) { return None; }

        let (loc1,s1) = self.buffer[0].into_inner();
        let (loc2,s2) = self.buffer[1].into_inner();
        match (s1,s2) {
            (BasicToken::Punctuation("@"),BasicToken::Alphanumeric(s)) |
            (BasicToken::Punctuation("@"),BasicToken::Number(s)) => match Local::from_segment(loc1,loc2) {
                Ok(local) => {
                    self.buffer.pop_front();
                    self.buffer.pop_front();
                    
                    Some(local.local(Token::Struct({
                        #[cfg(feature = "strings")]
                        { Struct::Mention(s.to_string()) }
                        #[cfg(not(feature = "strings"))]
                        { Struct::Mention }
                    })))
                },
                Err(_) => None,                    
            },
            _ => None,
        }
    }
    fn next_from_buffer(&mut self) -> Option<Local<Token>> {
        //if let Some(t) = self.check_url() { return Some(t); }
        if let Some(t) = self.check_hashtag() { return Some(t); }
        if let Some(t) = self.check_mention() { return Some(t); }
        match self.buffer.pop_front() {
            Some(local_tok) => {
                let (local,tok) = local_tok.into_inner();
                Some(local.local(match tok {
                    BasicToken::Alphanumeric(s) => self.basic_alphanumeric_to_pt(s),
                    BasicToken::Number(s) => self.basic_number_to_pt(s),
                    BasicToken::Punctuation(s) => self.basic_punctuation_to_pt(s),
                    BasicToken::Mixed(s) => self.basic_mixed_to_pt(s),
                    BasicToken::Separator(s) => self.basic_separator_to_pt(s),
                    BasicToken::Formatter(s) => self.basic_formater_to_pt(s),
                }))
            },
            None => None,
        }
    }
}

