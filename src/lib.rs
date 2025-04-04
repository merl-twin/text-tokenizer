use std::sync::Arc;
use text_parsing::{Breaker, IntoSource, Local, Snip, Source, SourceEvent};

mod emoji;
pub use emoji::EMOJIMAP;

mod breakers;
pub use breakers::{SentenceBreaker, UnicodeSentenceBreaker};

mod wordbreaker;

mod options;
pub use options::{IntoTokenizer, TokenizerOptions, TokenizerParams};

mod tokens;
pub use tokens::Tokens;

mod text_tokens;
use text_tokens::InnerBound;
pub use text_tokens::TextTokens;

#[derive(Debug)]
pub enum Error {
    TextParser(text_parsing::Error),
}

const EPS: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Number {
    Integer(i64),
    Float(f64),
}
impl Number {
    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Integer(i) => *i as f64,
            Number::Float(f) => *f,
        }
    }
}
impl Ord for Number {
    fn cmp(&self, other: &Number) -> std::cmp::Ordering {
        let s = self.as_f64();
        let o = other.as_f64();
        let d = s - o;
        match d.abs() < EPS {
            true => std::cmp::Ordering::Equal,
            false => {
                if d > 0.0 {
                    return std::cmp::Ordering::Greater;
                }
                if d < 0.0 {
                    return std::cmp::Ordering::Less;
                }
                std::cmp::Ordering::Equal
            }
        }
    }
}
impl Eq for Number {}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum Separator {
    Space,
    Tab,
    Newline,
    Char(char),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum Formatter {
    Char(char),
    Joiner, // u{200d}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum Special {
    Punctuation(char),
    Symbol(char),
    Separator(Separator),
}

#[cfg(feature = "strings")]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Word {
    Word(String),
    StrangeWord(String),
    Numerical(Numerical),
    Number(Number),
    Emoji(&'static str),
}

#[cfg(feature = "strings")]
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum Numerical {
    //Date(String),
    //Ip(String),
    //Countable(String),
    DotSeparated(String),
    Measures(String),
    Alphanumeric(String),
}

#[cfg(feature = "strings")]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Struct {
    Hashtag(String),
    Mention(String),
    //Url(String),
}

#[cfg(feature = "strings")]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Unicode {
    String(String),
    Formatter(Formatter),
}

#[cfg(not(feature = "strings"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Word {
    Word,
    StrangeWord,
    Numerical(Numerical),
    Number(Number),
    Emoji(&'static str),
}

#[cfg(not(feature = "strings"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Numerical {
    //Date,
    //Ip,
    //Countable,
    DotSeparated,
    Measures,
    Alphanumeric,
}

#[cfg(not(feature = "strings"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Struct {
    Hashtag,
    Mention,
    //Url,
}

#[cfg(not(feature = "strings"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Unicode {
    String,
    Formatter(Formatter),
}

#[cfg(feature = "strings")]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Token {
    Word(Word),
    Struct(Struct),
    Special(Special),
    Unicode(Unicode),
}

#[cfg(not(feature = "strings"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Token {
    Word(Word),
    Struct(Struct),
    Special(Special),
    Unicode(Unicode),
}

/*pub trait IntoTokens<T> {
    type IntoTokens: IntoTokenizer<IntoTokens = T>;
}

impl<'s> IntoTokenSource<Token2> for &'s str {
    type IntoTokens = TextStr<'s>;

    fn (self) -> Result<TextStr<'s>,Error> {
        TextStr::new(self)
    }

}*/

#[derive(Debug)]
pub struct TextStr<'s> {
    buffer: &'s str,
    originals: Vec<Local<()>>,
    breakers: Vec<InnerBound>,
}
impl<'s> TextStr<'s> {
    pub fn new<'a>(s: &'a str) -> Result<TextStr<'a>, Error> {
        let text = inner_new(s.into_source(), false)?;
        Ok(TextStr {
            buffer: s,
            originals: text.originals,
            breakers: text.breakers,
        })
    }
}

fn inner_new<S: Source>(mut source: S, with_buffer: bool) -> Result<Text, Error> {
    let mut buffer = String::new();
    let mut originals = Vec::new();
    let mut breakers = Vec::new();

    while let Some(local_se) = source.next_char().map_err(Error::TextParser)? {
        let (local, se) = local_se.into_inner();
        let c = match se {
            SourceEvent::Char(c) => match c {
                '\u{0060}' => '\u{0027}',
                _ => c,
            },
            SourceEvent::Breaker(b) => {
                let (c, opt_b) = match b {
                    Breaker::None => continue,
                    Breaker::Space => (' ', None),
                    Breaker::Line => ('\n', None),
                    Breaker::Word => ('\u{200B}', Some(b)), // zero width space
                    Breaker::Sentence | Breaker::Paragraph | Breaker::Section => ('\n', Some(b)),
                };
                if let Some(b) = opt_b {
                    let br = InnerBound {
                        bytes: Snip {
                            offset: buffer.len(),
                            length: c.len_utf8(),
                        },
                        chars: Snip {
                            offset: originals.len(),
                            length: 1,
                        },
                        breaker: b,
                        original: Some(local),
                    };
                    //println!("BR: {:?}",br);
                    breakers.push(br);
                }
                c
            }
        };
        if with_buffer {
            buffer.push(c);
        }
        originals.push(local);
    }
    Ok(Text {
        buffer: Arc::new(buffer),
        originals,
        breakers,
    })
}

#[derive(Debug)]
pub struct Text {
    buffer: Arc<String>,
    originals: Vec<Local<()>>,
    breakers: Vec<InnerBound>,
}
impl Text {
    pub fn new<S: Source>(source: S) -> Result<Text, Error> {
        inner_new(source, true)
    }
    pub fn token_text<'s>(&'s self, token: &TextToken) -> &'s str {
        let Snip {
            offset: begin,
            length: len,
        } = token.locality.bytes();
        let end = begin + len;
        &self.buffer[begin..end]
    }
    pub fn text(&self) -> &str {
        self.buffer.as_ref()
    }
    pub fn original_locality(&self, idx: usize) -> Option<Local<()>> {
        self.originals.get(idx).copied()
    }
    pub fn originals(&self) -> &Vec<Local<()>> {
        &self.originals
    }
    pub fn shared_text(&self) -> Arc<String> {
        self.buffer.clone()
    }
}

impl TryFrom<String> for Text {
    type Error = Error;

    fn try_from(s: String) -> Result<Text, Error> {
        let mut text = inner_new((&s).into_source(), false)?;
        text.buffer = Arc::new(s);
        Ok(text)
    }
}

impl TryFrom<&str> for Text {
    type Error = Error;

    fn try_from(s: &str) -> Result<Text, Error> {
        Text::new(s.into_source())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Bound {
    Sentence,
    Paragraph,
    Section,
}

#[cfg(feature = "strings")]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct TextToken {
    locality: Local<()>,
    original: Option<Local<()>>,
    pub token: Token2,
}

#[cfg(not(feature = "strings"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub struct TextToken {
    locality: Local<()>,
    original: Option<Local<()>>,
    pub token: Token2,
}

#[cfg(test)]
impl TextToken {
    fn into_original_token_1(self) -> Option<Local<Token>> {
        match self.original {
            Some(original) => self.token.into_token().map(|t| original.local(t)),
            None => None,
        }
    }
}

impl TextToken {
    pub fn local(&self) -> Local<()> {
        self.locality
    }
    pub fn original(&self) -> Option<Local<()>> {
        self.original
    }
    pub fn into_position(mut self) -> TextToken {
        self.locality = self.locality.into_position();
        self.original = self.original.map(|or| or.into_position());
        self
    }
    pub fn try_as_token(&self) -> Result<Token, Bound> {
        self.token.try_as_token()
    }
    pub fn as_original_token(&self) -> Option<Local<&Token2>> {
        self.original.map(|original| original.local(&self.token))
    }
    pub fn into_original_token(self) -> Option<Local<Token2>> {
        self.original.map(|original| original.local(self.token))
    }
    pub fn original_str<'s>(&self, original: &'s str) -> Result<&'s str, OriginalError> {
        match self.original {
            Some(local) => {
                let Snip {
                    offset: begin,
                    length: len,
                } = local.bytes();
                let end = begin + len;
                match original.get(begin..end) {
                    Some(s) => Ok(s),
                    None => Err(OriginalError::InvalidSnip),
                }
            }
            None => Err(OriginalError::NoOriginal),
        }
    }

    pub fn test_token(lt: Local<Token2>) -> TextToken {
        let (local, token) = lt.into_inner();
        TextToken {
            locality: local,
            original: Some(local.local(())),
            token,
        }
    }
    pub fn test_new(token: Token2, local: Local<()>, original: Option<Local<()>>) -> TextToken {
        TextToken {
            locality: local,
            original,
            token,
        }
    }
}

/*pub trait TokenExt: Iterator<Item = TextToken> + Sized {
    fn merge_separators(self) -> Merger<Self>;
}

impl<T> TokenExt for T where T: Iterator<Item = TextToken> {
    fn merge_separators(self) -> Merger<Self> {
        Merger {
            tokens: self,
        }
    }
}

pub struct Merger<T>
where T: Iterator<Item = TextToken>
{
    tokens: T,
}
impl<T> Iterator for Merger<T>
where T: Iterator<Item = TextToken>
{
    type Item = TextToken;
    fn next(&mut self) -> Option<Self::Item> {
        self.tokens.next()
    }
}*/

#[derive(Debug)]
pub enum OriginalError {
    NoOriginal,
    InvalidSnip,
}

/*#[derive(Debug,Clone,PartialEq)]
pub enum ExtToken {
    Token(Local<Token>),
    Breaker(Local<Bound>),
    Bound(Bound),
}*/

#[cfg(feature = "strings")]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Token2 {
    Word(Word),
    Struct(Struct),
    Special(Special),
    Unicode(Unicode),

    Bound(Bound),
}
#[cfg(not(feature = "strings"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum Token2 {
    Word(Word),
    Struct(Struct),
    Special(Special),
    Unicode(Unicode),

    Bound(Bound),
}
impl From<Token> for Token2 {
    fn from(t: Token) -> Token2 {
        match t {
            Token::Word(w) => Token2::Word(w),
            Token::Struct(s) => Token2::Struct(s),
            Token::Special(s) => Token2::Special(s),
            Token::Unicode(u) => Token2::Unicode(u),
        }
    }
}
impl Token2 {
    #[cfg(not(feature = "strings"))]
    fn try_as_token(&self) -> Result<Token, Bound> {
        (*self).try_into_token()
    }

    #[cfg(feature = "strings")]
    fn try_as_token(&self) -> Result<Token, Bound> {
        self.clone().try_into_token()
    }

    fn try_into_token(self) -> Result<Token, Bound> {
        match self {
            Token2::Word(w) => Ok(Token::Word(w)),
            Token2::Struct(s) => Ok(Token::Struct(s)),
            Token2::Special(s) => Ok(Token::Special(s)),
            Token2::Unicode(u) => Ok(Token::Unicode(u)),
            Token2::Bound(b) => Err(b),
        }
    }
}
#[cfg(test)]
impl Token2 {
    fn into_token(self) -> Option<Token> {
        match self {
            Token2::Word(w) => Some(Token::Word(w)),
            Token2::Struct(s) => Some(Token::Struct(s)),
            Token2::Special(s) => Some(Token::Special(s)),
            Token2::Unicode(u) => Some(Token::Unicode(u)),
            Token2::Bound(_) => None,
        }
    }
}

#[cfg(test)]
mod test_v0_5 {
    use super::*;
    use text_parsing::{entities, tagger, IntoPipeParser, IntoSource, ParserExt, SourceExt};

    //#[test]
    fn basic() {
        /*let uws = "Oxana Putan shared the quick (\"brown\") fox can't jump 32.3 feet, right?4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n+Done! Готово";

        /*let result = vec![
            PositionalToken { source: uws, offset: 0, length: 7, token: Token::Word("l'oreal".to_string()) },
            PositionalToken { source: uws, offset: 7, length: 1, token: Token::Punctuation(";".to_string()) },
            PositionalToken { source: uws, offset: 8, length: 1, token: Token::Separator(Separator::Space) },
            PositionalToken { source: uws, offset: 9, length: 7, token: Token::Word("l'oreal".to_string()) },
        ];*/
        let text = Text::new({
            uws.into_source()
                .into_separator()
                .merge_separators()
        }).unwrap();*/

        let uws = "<p>Oxana Putan shared the quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc.</p><p> qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n+Done! Готово</p>";
        let text = Text::new({
            uws.into_source()
                .pipe(tagger::Builder::new().create().into_breaker())
                .pipe(entities::Builder::new().create().into_piped())
                .into_separator()
        })
        .unwrap();
        let lib_res = text
            .into_tokenizer({
                TokenizerParams::default()
                    .add_option(TokenizerOptions::SplitDot)
                    .add_option(TokenizerOptions::SplitUnderscore)
                    .add_option(TokenizerOptions::SplitColon)
                    .with_default_sentences()
            })
            .collect::<Vec<_>>();

        for tok in lib_res {
            println!(
                "C{:?}, B{:?}, {:?} -> {:?}",
                tok.original.map(|loc| loc.chars()),
                tok.original.map(|loc| loc.bytes()),
                tok.token,
                tok.original_str(uws)
            );
        }

        panic!()
    }
}

#[cfg(test)]
#[cfg(feature = "strings")]
mod test {
    use super::*;
    use text_parsing::{
        entities, tagger, IntoPipeParser, IntoSource, Localize, ParserExt, SourceExt,
    };

    /*
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
        let mut r = format!("CharToken {{ byte_offset: {}, byte_length: {}, char_offset: {}, char_length: {}, token: Token::{:?} }},",tok.byte_offset,tok.byte_length,tok.char_offset,tok.char_length,tok.token);
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
    }*/

    #[derive(Debug, Clone)]
    struct CharToken {
        byte_offset: usize,
        byte_length: usize,
        char_offset: usize,
        char_length: usize,
        token: Token,
    }
    impl Into<Local<Token>> for CharToken {
        fn into(self) -> Local<Token> {
            self.token.localize(
                Snip {
                    offset: self.char_offset,
                    length: self.char_length,
                },
                Snip {
                    offset: self.byte_offset,
                    length: self.byte_length,
                },
            )
        }
    }

    #[derive(Debug, Clone)]
    struct PositionalToken {
        source: &'static str,
        offset: usize,
        length: usize,
        token: Token,
    }
    impl Into<Local<Token>> for PositionalToken {
        fn into(self) -> Local<Token> {
            self.token.localize(
                Snip {
                    offset: self.source[..self.offset].chars().count(),
                    length: self.source[self.offset..self.offset + self.length]
                        .chars()
                        .count(),
                },
                Snip {
                    offset: self.offset,
                    length: self.length,
                },
            )
        }
    }

    fn check_results(result: &Vec<PositionalToken>, lib_res: &Vec<Local<Token>>, _uws: &str) {
        assert_eq!(result.len(), lib_res.len());
        for i in 0..result.len() {
            let res: Local<Token> = result[i].clone().into();
            assert_eq!(res, lib_res[i]);
        }
    }

    fn check_cresults(result: &Vec<CharToken>, lib_res: &Vec<Local<Token>>, _uws: &str) {
        assert_eq!(result.len(), lib_res.len());
        for i in 0..result.len() {
            let res: Local<Token> = result[i].clone().into();
            assert_eq!(res, lib_res[i]);
        }
    }

    fn check<T: Clone + std::fmt::Debug + Into<Local<Token>>>(
        res: &Vec<T>,
        lib: &Vec<Local<Token>>,
        _uws: &str,
    ) {
        let mut lib = lib.iter();
        let mut res = res.iter().map(|r| {
            let res: Local<Token> = r.clone().into();
            res
        });
        let mut diff = Vec::new();
        loop {
            match (lib.next(), res.next()) {
                (Some(lw), Some(rw)) => {
                    if *lw != rw {
                        diff.push(format!("LIB:  {:?}", lw));
                        diff.push(format!("TEST: {:?}", rw));
                        diff.push("".to_string())
                    }
                }
                (Some(lw), None) => {
                    diff.push(format!("LIB:  {:?}", lw));
                    diff.push("TEST: ----".to_string());
                    diff.push("".to_string())
                }
                (None, Some(rw)) => {
                    diff.push("LIB:  ----".to_string());
                    diff.push(format!("TEST: {:?}", rw));
                    diff.push("".to_string())
                }
                (None, None) => break,
            }
        }
        if diff.len() > 0 {
            for ln in &diff {
                println!("{}", ln);
            }
            panic!("Diff count: {}", diff.len() / 3);
        }
    }

    #[test]
    fn spaces() {
        let uws = "    spaces    too   many   apces   ";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 4,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 4,
                length: 6,
                token: Token::Word(Word::Word("spaces".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 10,
                length: 4,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 14,
                length: 3,
                token: Token::Word(Word::Word("too".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 3,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 20,
                length: 4,
                token: Token::Word(Word::Word("many".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 24,
                length: 3,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 27,
                length: 5,
                token: Token::Word(Word::Word("apces".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 32,
                length: 3,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
        ];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
        //panic!()
    }

    #[test]
    fn numbers() {
        let uws = "(() -2\n()  -2";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 1,
                token: Token::Special(Special::Punctuation('(')),
            },
            PositionalToken {
                source: uws,
                offset: 1,
                length: 1,
                token: Token::Special(Special::Punctuation('(')),
            },
            PositionalToken {
                source: uws,
                offset: 2,
                length: 1,
                token: Token::Special(Special::Punctuation(')')),
            },
            PositionalToken {
                source: uws,
                offset: 3,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 4,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(-2))),
            },
            PositionalToken {
                source: uws,
                offset: 6,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 7,
                length: 1,
                token: Token::Special(Special::Punctuation('(')),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Punctuation(')')),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 11,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(-2))),
            },
        ];
        let lib_res = uws
            .into_tokenizer({
                TokenizerParams::default()
                    .add_option(TokenizerOptions::SplitDot)
                    .add_option(TokenizerOptions::SplitUnderscore)
                    .add_option(TokenizerOptions::SplitColon)
                    .add_option(TokenizerOptions::MergeWhites)
            })
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn word_with_inner_hyphens() {
        let uws = "Опро­сы по­ка­зы­ва­ют";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 14,
                token: Token::Word(Word::StrangeWord("Опро­сы".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 14,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 15,
                length: 28,
                token: Token::Word(Word::StrangeWord("по­ка­зы­ва­ют".to_string())),
            },
        ];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn mixed_but_word() {
        let uws = "L’Oreal";
        let result = vec![PositionalToken {
            source: uws,
            offset: 0,
            length: 9,
            token: Token::Word(Word::StrangeWord("L’Oreal".to_string())),
        }];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn hashtags() {
        let uws = "#hashtag#hashtag2";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 1,
                token: Token::Special(Special::Punctuation('#')),
            },
            PositionalToken {
                source: uws,
                offset: 1,
                length: 7,
                token: Token::Word(Word::Word("hashtag".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Punctuation('#')),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 8,
                token: Token::Word(Word::Numerical(Numerical::Alphanumeric(
                    "hashtag2".to_string(),
                ))),
            },
        ];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn apostrophe() {
        let uws = "l'oreal; l\u{0060}oreal";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 7,
                token: Token::Word(Word::Word("l'oreal".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 7,
                length: 1,
                token: Token::Special(Special::Punctuation(';')),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 7,
                token: Token::Word(Word::Word("l'oreal".to_string())),
            },
        ];
        let text = Text::new(uws.into_source()).unwrap();
        let lib_res = text
            .into_tokenizer(TokenizerParams::v1())
            .filter_map(|tt| tt.into_original_token_1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn char_tokens() {
        let uws = "[Oxana Putan|1712640565] shared the quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n+Done! Готово";
        let result = vec![
            CharToken {
                byte_offset: 0,
                byte_length: 1,
                char_offset: 0,
                char_length: 1,
                token: Token::Special(Special::Punctuation('[')),
            },
            CharToken {
                byte_offset: 1,
                byte_length: 5,
                char_offset: 1,
                char_length: 5,
                token: Token::Word(Word::Word("Oxana".to_string())),
            },
            CharToken {
                byte_offset: 6,
                byte_length: 1,
                char_offset: 6,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 7,
                byte_length: 5,
                char_offset: 7,
                char_length: 5,
                token: Token::Word(Word::Word("Putan".to_string())),
            },
            CharToken {
                byte_offset: 12,
                byte_length: 1,
                char_offset: 12,
                char_length: 1,
                token: Token::Special(Special::Punctuation('|')),
            },
            CharToken {
                byte_offset: 13,
                byte_length: 10,
                char_offset: 13,
                char_length: 10,
                token: Token::Word(Word::Number(Number::Integer(1712640565))),
            },
            CharToken {
                byte_offset: 23,
                byte_length: 1,
                char_offset: 23,
                char_length: 1,
                token: Token::Special(Special::Punctuation(']')),
            },
            /*CharToken { byte_offset: 0, byte_length: 24, char_offset: 0, char_length: 24, token: Token::BBCode { left: vec![
            CharToken { byte_offset: 1, byte_length: 5, char_offset: 1, char_length: 5, token: Token::Word(Word::Word("Oxana".to_string())) },
            CharToken { byte_offset: 6, byte_length: 1, char_offset: 6, char_length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            CharToken { byte_offset: 7, byte_length: 5, char_offset: 7, char_length: 5, token: Token::Word(Word::Word("Putan".to_string())) },
            ], right: vec![
            CharToken { byte_offset: 13, byte_length: 10, char_offset: 13, char_length: 10, token: Token::Word(Word::Number(Number::Integer(1712640565))) },
            ] } },*/
            CharToken {
                byte_offset: 24,
                byte_length: 1,
                char_offset: 24,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 25,
                byte_length: 6,
                char_offset: 25,
                char_length: 6,
                token: Token::Word(Word::Word("shared".to_string())),
            },
            CharToken {
                byte_offset: 31,
                byte_length: 1,
                char_offset: 31,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 32,
                byte_length: 3,
                char_offset: 32,
                char_length: 3,
                token: Token::Word(Word::Word("the".to_string())),
            },
            CharToken {
                byte_offset: 35,
                byte_length: 1,
                char_offset: 35,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 36,
                byte_length: 5,
                char_offset: 36,
                char_length: 5,
                token: Token::Word(Word::Word("quick".to_string())),
            },
            CharToken {
                byte_offset: 41,
                byte_length: 1,
                char_offset: 41,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 42,
                byte_length: 1,
                char_offset: 42,
                char_length: 1,
                token: Token::Special(Special::Punctuation('(')),
            },
            CharToken {
                byte_offset: 43,
                byte_length: 1,
                char_offset: 43,
                char_length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            CharToken {
                byte_offset: 44,
                byte_length: 5,
                char_offset: 44,
                char_length: 5,
                token: Token::Word(Word::Word("brown".to_string())),
            },
            CharToken {
                byte_offset: 49,
                byte_length: 1,
                char_offset: 49,
                char_length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            CharToken {
                byte_offset: 50,
                byte_length: 1,
                char_offset: 50,
                char_length: 1,
                token: Token::Special(Special::Punctuation(')')),
            },
            CharToken {
                byte_offset: 51,
                byte_length: 1,
                char_offset: 51,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 52,
                byte_length: 3,
                char_offset: 52,
                char_length: 3,
                token: Token::Word(Word::Word("fox".to_string())),
            },
            CharToken {
                byte_offset: 55,
                byte_length: 1,
                char_offset: 55,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 56,
                byte_length: 5,
                char_offset: 56,
                char_length: 5,
                token: Token::Word(Word::Word("can\'t".to_string())),
            },
            CharToken {
                byte_offset: 61,
                byte_length: 1,
                char_offset: 61,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 62,
                byte_length: 4,
                char_offset: 62,
                char_length: 4,
                token: Token::Word(Word::Word("jump".to_string())),
            },
            CharToken {
                byte_offset: 66,
                byte_length: 1,
                char_offset: 66,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 67,
                byte_length: 4,
                char_offset: 67,
                char_length: 4,
                token: Token::Word(Word::Number(Number::Float(32.3))),
            },
            CharToken {
                byte_offset: 71,
                byte_length: 1,
                char_offset: 71,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 72,
                byte_length: 4,
                char_offset: 72,
                char_length: 4,
                token: Token::Word(Word::Word("feet".to_string())),
            },
            CharToken {
                byte_offset: 76,
                byte_length: 1,
                char_offset: 76,
                char_length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            CharToken {
                byte_offset: 77,
                byte_length: 1,
                char_offset: 77,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 78,
                byte_length: 5,
                char_offset: 78,
                char_length: 5,
                token: Token::Word(Word::Word("right".to_string())),
            },
            CharToken {
                byte_offset: 83,
                byte_length: 1,
                char_offset: 83,
                char_length: 1,
                token: Token::Special(Special::Punctuation('?')),
            },
            CharToken {
                byte_offset: 84,
                byte_length: 1,
                char_offset: 84,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 85,
                byte_length: 4,
                char_offset: 85,
                char_length: 4,
                token: Token::Word(Word::Numerical(Numerical::Measures("4pda".to_string()))),
            },
            CharToken {
                byte_offset: 89,
                byte_length: 1,
                char_offset: 89,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 90,
                byte_length: 3,
                char_offset: 90,
                char_length: 3,
                token: Token::Word(Word::Word("etc".to_string())),
            },
            CharToken {
                byte_offset: 93,
                byte_length: 1,
                char_offset: 93,
                char_length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            CharToken {
                byte_offset: 94,
                byte_length: 1,
                char_offset: 94,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 95,
                byte_length: 3,
                char_offset: 95,
                char_length: 3,
                token: Token::Word(Word::Word("qeq".to_string())),
            },
            CharToken {
                byte_offset: 98,
                byte_length: 1,
                char_offset: 98,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 99,
                byte_length: 5,
                char_offset: 99,
                char_length: 5,
                token: Token::Word(Word::Word("U.S.A".to_string())),
            },
            CharToken {
                byte_offset: 104,
                byte_length: 2,
                char_offset: 104,
                char_length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 106,
                byte_length: 3,
                char_offset: 106,
                char_length: 3,
                token: Token::Word(Word::Word("asd".to_string())),
            },
            CharToken {
                byte_offset: 109,
                byte_length: 3,
                char_offset: 109,
                char_length: 3,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            CharToken {
                byte_offset: 112,
                byte_length: 3,
                char_offset: 112,
                char_length: 3,
                token: Token::Word(Word::Word("Brr".to_string())),
            },
            CharToken {
                byte_offset: 115,
                byte_length: 1,
                char_offset: 115,
                char_length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            CharToken {
                byte_offset: 116,
                byte_length: 1,
                char_offset: 116,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 117,
                byte_length: 4,
                char_offset: 117,
                char_length: 4,
                token: Token::Word(Word::Word("it\'s".to_string())),
            },
            CharToken {
                byte_offset: 121,
                byte_length: 1,
                char_offset: 121,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 122,
                byte_length: 4,
                char_offset: 122,
                char_length: 4,
                token: Token::Word(Word::Number(Number::Float(29.3))),
            },
            CharToken {
                byte_offset: 126,
                byte_length: 2,
                char_offset: 126,
                char_length: 1,
                token: Token::Special(Special::Symbol('°')),
            },
            CharToken {
                byte_offset: 128,
                byte_length: 1,
                char_offset: 127,
                char_length: 1,
                token: Token::Word(Word::Word("F".to_string())),
            },
            CharToken {
                byte_offset: 129,
                byte_length: 1,
                char_offset: 128,
                char_length: 1,
                token: Token::Special(Special::Punctuation('!')),
            },
            CharToken {
                byte_offset: 130,
                byte_length: 1,
                char_offset: 129,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            CharToken {
                byte_offset: 131,
                byte_length: 1,
                char_offset: 130,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 132,
                byte_length: 14,
                char_offset: 131,
                char_length: 7,
                token: Token::Word(Word::Word("Русское".to_string())),
            },
            CharToken {
                byte_offset: 146,
                byte_length: 1,
                char_offset: 138,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 147,
                byte_length: 22,
                char_offset: 139,
                char_length: 11,
                token: Token::Word(Word::Word("предложение".to_string())),
            },
            CharToken {
                byte_offset: 169,
                byte_length: 1,
                char_offset: 150,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 170,
                byte_length: 5,
                char_offset: 151,
                char_length: 5,
                token: Token::Struct(Struct::Hashtag("36.6".to_string())),
            },
            CharToken {
                byte_offset: 175,
                byte_length: 1,
                char_offset: 156,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 176,
                byte_length: 6,
                char_offset: 157,
                char_length: 3,
                token: Token::Word(Word::Word("для".to_string())),
            },
            CharToken {
                byte_offset: 182,
                byte_length: 1,
                char_offset: 160,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 183,
                byte_length: 24,
                char_offset: 161,
                char_length: 12,
                token: Token::Word(Word::Word("тестирования".to_string())),
            },
            CharToken {
                byte_offset: 207,
                byte_length: 1,
                char_offset: 173,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 208,
                byte_length: 14,
                char_offset: 174,
                char_length: 7,
                token: Token::Word(Word::Word("деления".to_string())),
            },
            CharToken {
                byte_offset: 222,
                byte_length: 1,
                char_offset: 181,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 223,
                byte_length: 4,
                char_offset: 182,
                char_length: 2,
                token: Token::Word(Word::Word("по".to_string())),
            },
            CharToken {
                byte_offset: 227,
                byte_length: 1,
                char_offset: 184,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 228,
                byte_length: 12,
                char_offset: 185,
                char_length: 6,
                token: Token::Word(Word::Word("юникод".to_string())),
            },
            CharToken {
                byte_offset: 240,
                byte_length: 1,
                char_offset: 191,
                char_length: 1,
                token: Token::Special(Special::Punctuation('-')),
            },
            CharToken {
                byte_offset: 241,
                byte_length: 12,
                char_offset: 192,
                char_length: 6,
                token: Token::Word(Word::Word("словам".to_string())),
            },
            CharToken {
                byte_offset: 253,
                byte_length: 3,
                char_offset: 198,
                char_length: 3,
                token: Token::Special(Special::Punctuation('.')),
            },
            CharToken {
                byte_offset: 256,
                byte_length: 1,
                char_offset: 201,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            CharToken {
                byte_offset: 257,
                byte_length: 8,
                char_offset: 202,
                char_length: 2,
                token: Token::Word(Word::Emoji("russia")),
            },
            CharToken {
                byte_offset: 265,
                byte_length: 1,
                char_offset: 204,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 266,
                byte_length: 8,
                char_offset: 205,
                char_length: 2,
                token: Token::Word(Word::Emoji("sao_tome_and_principe")),
            },
            CharToken {
                byte_offset: 274,
                byte_length: 1,
                char_offset: 207,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            CharToken {
                byte_offset: 275,
                byte_length: 8,
                char_offset: 208,
                char_length: 2,
                token: Token::Word(Word::Emoji("blond_haired_person_dark_skin_tone")),
            },
            CharToken {
                byte_offset: 283,
                byte_length: 8,
                char_offset: 210,
                char_length: 2,
                token: Token::Word(Word::Emoji("baby_medium_skin_tone")),
            },
            CharToken {
                byte_offset: 291,
                byte_length: 8,
                char_offset: 212,
                char_length: 2,
                token: Token::Word(Word::Emoji("man_medium_skin_tone")),
            },
            CharToken {
                byte_offset: 299,
                byte_length: 1,
                char_offset: 214,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            CharToken {
                byte_offset: 300,
                byte_length: 1,
                char_offset: 215,
                char_length: 1,
                token: Token::Special(Special::Punctuation('+')),
            },
            CharToken {
                byte_offset: 301,
                byte_length: 4,
                char_offset: 216,
                char_length: 4,
                token: Token::Word(Word::Word("Done".to_string())),
            },
            CharToken {
                byte_offset: 305,
                byte_length: 1,
                char_offset: 220,
                char_length: 1,
                token: Token::Special(Special::Punctuation('!')),
            },
            CharToken {
                byte_offset: 306,
                byte_length: 1,
                char_offset: 221,
                char_length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            CharToken {
                byte_offset: 307,
                byte_length: 12,
                char_offset: 222,
                char_length: 6,
                token: Token::Word(Word::Word("Готово".to_string())),
            },
        ];

        let lib_res = uws
            .into_tokenizer(TokenizerParams::complex())
            .collect::<Vec<_>>();

        //print_cresult(); panic!();
        check_cresults(&result, &lib_res, uws);
    }

    #[test]
    fn general_default() {
        let uws = "The quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 3,
                token: Token::Word(Word::Word("The".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 3,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 4,
                length: 5,
                token: Token::Word(Word::Word("quick".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 10,
                length: 1,
                token: Token::Special(Special::Punctuation('(')),
            },
            PositionalToken {
                source: uws,
                offset: 11,
                length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            PositionalToken {
                source: uws,
                offset: 12,
                length: 5,
                token: Token::Word(Word::Word("brown".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 1,
                token: Token::Special(Special::Punctuation(')')),
            },
            PositionalToken {
                source: uws,
                offset: 19,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 20,
                length: 3,
                token: Token::Word(Word::Word("fox".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 23,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 24,
                length: 5,
                token: Token::Word(Word::Word("can\'t".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 29,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 30,
                length: 4,
                token: Token::Word(Word::Word("jump".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 34,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 35,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(32.3))),
            },
            PositionalToken {
                source: uws,
                offset: 39,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 40,
                length: 4,
                token: Token::Word(Word::Word("feet".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 44,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 45,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 46,
                length: 5,
                token: Token::Word(Word::Word("right".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 51,
                length: 1,
                token: Token::Special(Special::Punctuation('?')),
            },
            PositionalToken {
                source: uws,
                offset: 52,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 53,
                length: 4,
                token: Token::Word(Word::Numerical(Numerical::Measures("4pda".to_string()))),
            }, // TODO
            PositionalToken {
                source: uws,
                offset: 57,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 58,
                length: 3,
                token: Token::Word(Word::Word("etc".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 61,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 62,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 63,
                length: 3,
                token: Token::Word(Word::Word("qeq".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 66,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 67,
                length: 1,
                token: Token::Word(Word::Word("U".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 68,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 69,
                length: 1,
                token: Token::Word(Word::Word("S".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 70,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 71,
                length: 1,
                token: Token::Word(Word::Word("A".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 72,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 74,
                length: 3,
                token: Token::Word(Word::Word("asd".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 77,
                length: 3,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 80,
                length: 3,
                token: Token::Word(Word::Word("Brr".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 83,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 84,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 85,
                length: 4,
                token: Token::Word(Word::Word("it\'s".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 89,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 90,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(29.3))),
            },
            PositionalToken {
                source: uws,
                offset: 94,
                length: 2,
                token: Token::Special(Special::Symbol('°')),
            },
            PositionalToken {
                source: uws,
                offset: 96,
                length: 1,
                token: Token::Word(Word::Word("F".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 97,
                length: 1,
                token: Token::Special(Special::Punctuation('!')),
            },
            PositionalToken {
                source: uws,
                offset: 98,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 99,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 100,
                length: 14,
                token: Token::Word(Word::Word("Русское".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 114,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 115,
                length: 22,
                token: Token::Word(Word::Word("предложение".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 137,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 138,
                length: 1,
                token: Token::Special(Special::Punctuation('#')),
            },
            PositionalToken {
                source: uws,
                offset: 139,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(36.6))),
            },
            PositionalToken {
                source: uws,
                offset: 143,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 144,
                length: 6,
                token: Token::Word(Word::Word("для".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 150,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 151,
                length: 24,
                token: Token::Word(Word::Word("тестирования".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 175,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 176,
                length: 14,
                token: Token::Word(Word::Word("деления".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 190,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 191,
                length: 4,
                token: Token::Word(Word::Word("по".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 195,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 196,
                length: 12,
                token: Token::Word(Word::Word("юникод".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 208,
                length: 1,
                token: Token::Special(Special::Punctuation('-')),
            },
            PositionalToken {
                source: uws,
                offset: 209,
                length: 12,
                token: Token::Word(Word::Word("словам".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 221,
                length: 3,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 224,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
        ];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn general_no_split() {
        let uws = "The quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 3,
                token: Token::Word(Word::Word("The".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 3,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 4,
                length: 5,
                token: Token::Word(Word::Word("quick".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 10,
                length: 1,
                token: Token::Special(Special::Punctuation('(')),
            },
            PositionalToken {
                source: uws,
                offset: 11,
                length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            PositionalToken {
                source: uws,
                offset: 12,
                length: 5,
                token: Token::Word(Word::Word("brown".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 1,
                token: Token::Special(Special::Punctuation(')')),
            },
            PositionalToken {
                source: uws,
                offset: 19,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 20,
                length: 3,
                token: Token::Word(Word::Word("fox".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 23,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 24,
                length: 5,
                token: Token::Word(Word::Word("can\'t".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 29,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 30,
                length: 4,
                token: Token::Word(Word::Word("jump".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 34,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 35,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(32.3))),
            },
            PositionalToken {
                source: uws,
                offset: 39,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 40,
                length: 4,
                token: Token::Word(Word::Word("feet".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 44,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 45,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 46,
                length: 5,
                token: Token::Word(Word::Word("right".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 51,
                length: 1,
                token: Token::Special(Special::Punctuation('?')),
            },
            PositionalToken {
                source: uws,
                offset: 52,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 53,
                length: 4,
                token: Token::Word(Word::Numerical(Numerical::Measures("4pda".to_string()))),
            }, // TODO
            PositionalToken {
                source: uws,
                offset: 57,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 58,
                length: 3,
                token: Token::Word(Word::Word("etc".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 61,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 62,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 63,
                length: 3,
                token: Token::Word(Word::Word("qeq".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 66,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 67,
                length: 5,
                token: Token::Word(Word::Word("U.S.A".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 72,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 73,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 74,
                length: 3,
                token: Token::Word(Word::Word("asd".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 77,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 78,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 79,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 80,
                length: 3,
                token: Token::Word(Word::Word("Brr".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 83,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 84,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 85,
                length: 4,
                token: Token::Word(Word::Word("it\'s".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 89,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 90,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(29.3))),
            },
            PositionalToken {
                source: uws,
                offset: 94,
                length: 2,
                token: Token::Special(Special::Symbol('°')),
            },
            PositionalToken {
                source: uws,
                offset: 96,
                length: 1,
                token: Token::Word(Word::Word("F".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 97,
                length: 1,
                token: Token::Special(Special::Punctuation('!')),
            },
            PositionalToken {
                source: uws,
                offset: 98,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 99,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 100,
                length: 14,
                token: Token::Word(Word::Word("Русское".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 114,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 115,
                length: 22,
                token: Token::Word(Word::Word("предложение".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 137,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 138,
                length: 1,
                token: Token::Special(Special::Punctuation('#')),
            },
            PositionalToken {
                source: uws,
                offset: 139,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(36.6))),
            },
            PositionalToken {
                source: uws,
                offset: 143,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 144,
                length: 6,
                token: Token::Word(Word::Word("для".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 150,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 151,
                length: 24,
                token: Token::Word(Word::Word("тестирования".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 175,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 176,
                length: 14,
                token: Token::Word(Word::Word("деления".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 190,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 191,
                length: 4,
                token: Token::Word(Word::Word("по".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 195,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 196,
                length: 12,
                token: Token::Word(Word::Word("юникод".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 208,
                length: 1,
                token: Token::Special(Special::Punctuation('-')),
            },
            PositionalToken {
                source: uws,
                offset: 209,
                length: 12,
                token: Token::Word(Word::Word("словам".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 221,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 222,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 223,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 224,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
        ];
        let lib_res = uws.into_tokenizer(Default::default()).collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn general_complex() {
        let uws = "The quick (\"brown\") fox can't jump 32.3 feet, right? 4pda etc. qeq U.S.A  asd\n\n\nBrr, it's 29.3°F!\n Русское предложение #36.6 для тестирования деления по юникод-словам...\n";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 3,
                token: Token::Word(Word::Word("The".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 3,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 4,
                length: 5,
                token: Token::Word(Word::Word("quick".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 10,
                length: 1,
                token: Token::Special(Special::Punctuation('(')),
            },
            PositionalToken {
                source: uws,
                offset: 11,
                length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            PositionalToken {
                source: uws,
                offset: 12,
                length: 5,
                token: Token::Word(Word::Word("brown".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 1,
                token: Token::Special(Special::Punctuation('"')),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 1,
                token: Token::Special(Special::Punctuation(')')),
            },
            PositionalToken {
                source: uws,
                offset: 19,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 20,
                length: 3,
                token: Token::Word(Word::Word("fox".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 23,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 24,
                length: 5,
                token: Token::Word(Word::Word("can\'t".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 29,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 30,
                length: 4,
                token: Token::Word(Word::Word("jump".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 34,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 35,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(32.3))),
            },
            PositionalToken {
                source: uws,
                offset: 39,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 40,
                length: 4,
                token: Token::Word(Word::Word("feet".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 44,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 45,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 46,
                length: 5,
                token: Token::Word(Word::Word("right".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 51,
                length: 1,
                token: Token::Special(Special::Punctuation('?')),
            },
            PositionalToken {
                source: uws,
                offset: 52,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 53,
                length: 4,
                token: Token::Word(Word::Numerical(Numerical::Measures("4pda".to_string()))),
            }, // TODO
            PositionalToken {
                source: uws,
                offset: 57,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 58,
                length: 3,
                token: Token::Word(Word::Word("etc".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 61,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 62,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 63,
                length: 3,
                token: Token::Word(Word::Word("qeq".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 66,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 67,
                length: 5,
                token: Token::Word(Word::Word("U.S.A".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 72,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 74,
                length: 3,
                token: Token::Word(Word::Word("asd".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 77,
                length: 3,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 80,
                length: 3,
                token: Token::Word(Word::Word("Brr".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 83,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 84,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 85,
                length: 4,
                token: Token::Word(Word::Word("it\'s".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 89,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 90,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(29.3))),
            },
            PositionalToken {
                source: uws,
                offset: 94,
                length: 2,
                token: Token::Special(Special::Symbol('°')),
            },
            PositionalToken {
                source: uws,
                offset: 96,
                length: 1,
                token: Token::Word(Word::Word("F".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 97,
                length: 1,
                token: Token::Special(Special::Punctuation('!')),
            },
            PositionalToken {
                source: uws,
                offset: 98,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 99,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 100,
                length: 14,
                token: Token::Word(Word::Word("Русское".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 114,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 115,
                length: 22,
                token: Token::Word(Word::Word("предложение".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 137,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 138,
                length: 5,
                token: Token::Struct(Struct::Hashtag("36.6".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 143,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 144,
                length: 6,
                token: Token::Word(Word::Word("для".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 150,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 151,
                length: 24,
                token: Token::Word(Word::Word("тестирования".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 175,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 176,
                length: 14,
                token: Token::Word(Word::Word("деления".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 190,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 191,
                length: 4,
                token: Token::Word(Word::Word("по".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 195,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 196,
                length: 12,
                token: Token::Word(Word::Word("юникод".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 208,
                length: 1,
                token: Token::Special(Special::Punctuation('-')),
            },
            PositionalToken {
                source: uws,
                offset: 209,
                length: 12,
                token: Token::Word(Word::Word("словам".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 221,
                length: 3,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 224,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
        ];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::complex())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn plus_minus() {
        let uws = "+23 -4.5 -34 +25.7 - 2 + 5.6";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(23))),
            },
            PositionalToken {
                source: uws,
                offset: 3,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 4,
                length: 4,
                token: Token::Word(Word::Number(Number::Float(-4.5))),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(-34))),
            },
            PositionalToken {
                source: uws,
                offset: 12,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 13,
                length: 5,
                token: Token::Word(Word::Number(Number::Float(25.7))),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 19,
                length: 1,
                token: Token::Special(Special::Punctuation('-')),
            },
            PositionalToken {
                source: uws,
                offset: 20,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 21,
                length: 1,
                token: Token::Word(Word::Number(Number::Integer(2))),
            },
            PositionalToken {
                source: uws,
                offset: 22,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 23,
                length: 1,
                token: Token::Special(Special::Punctuation('+')),
            },
            PositionalToken {
                source: uws,
                offset: 24,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 25,
                length: 3,
                token: Token::Word(Word::Number(Number::Float(5.6))),
            },
        ];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check(&result, &lib_res, uws);
        //print_result(&lib_res); panic!("")
    }

    #[test]
    #[ignore]
    fn woman_bouncing_ball() {
        let uws = "\u{26f9}\u{200d}\u{2640}";
        let result = vec![PositionalToken {
            source: uws,
            offset: 0,
            length: 9,
            token: Token::Word(Word::Emoji("woman_bouncing_ball")),
        }];
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
        //print_result(&lib_res); panic!("")
    }

    #[test]
    fn emoji_and_rusabbr_default() {
        let uws = "🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n👱\nС.С.С.Р.\n👨‍👩‍👦‍👦\n🧠\n";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 8,
                token: Token::Word(Word::Emoji("russia")),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 8,
                token: Token::Word(Word::Emoji("sao_tome_and_principe")),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 8,
                token: Token::Word(Word::Emoji("blond_haired_person_dark_skin_tone")),
            },
            PositionalToken {
                source: uws,
                offset: 26,
                length: 8,
                token: Token::Word(Word::Emoji("baby_medium_skin_tone")),
            },
            PositionalToken {
                source: uws,
                offset: 34,
                length: 8,
                token: Token::Word(Word::Emoji("man_medium_skin_tone")),
            },
            PositionalToken {
                source: uws,
                offset: 42,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 43,
                length: 4,
                token: Token::Word(Word::Emoji("blond_haired_person")),
            },
            PositionalToken {
                source: uws,
                offset: 47,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 48,
                length: 2,
                token: Token::Word(Word::Word("С".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 50,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 51,
                length: 2,
                token: Token::Word(Word::Word("С".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 53,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 54,
                length: 2,
                token: Token::Word(Word::Word("С".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 56,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 57,
                length: 2,
                token: Token::Word(Word::Word("Р".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 59,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 60,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 61,
                length: 25,
                token: Token::Word(Word::Emoji("family_man_woman_boy_boy")),
            },
            PositionalToken {
                source: uws,
                offset: 86,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 87,
                length: 4,
                token: Token::Word(Word::Emoji("brain")),
            },
            PositionalToken {
                source: uws,
                offset: 91,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
        ];

        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
        //print_result(&lib_res); panic!();
    }

    #[test]
    fn emoji_and_rusabbr_no_split() {
        let uws = "🇷🇺 🇸🇹\n👱🏿👶🏽👨🏽\n👱\nС.С.С.Р.\n👨‍👩‍👦‍👦\n🧠\n";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 8,
                token: Token::Word(Word::Emoji("russia")),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 8,
                token: Token::Word(Word::Emoji("sao_tome_and_principe")),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 8,
                token: Token::Word(Word::Emoji("blond_haired_person_dark_skin_tone")),
            },
            PositionalToken {
                source: uws,
                offset: 26,
                length: 8,
                token: Token::Word(Word::Emoji("baby_medium_skin_tone")),
            },
            PositionalToken {
                source: uws,
                offset: 34,
                length: 8,
                token: Token::Word(Word::Emoji("man_medium_skin_tone")),
            },
            PositionalToken {
                source: uws,
                offset: 42,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 43,
                length: 4,
                token: Token::Word(Word::Emoji("blond_haired_person")),
            },
            PositionalToken {
                source: uws,
                offset: 47,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 48,
                length: 11,
                token: Token::Word(Word::Word("С.С.С.Р".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 59,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 60,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 61,
                length: 25,
                token: Token::Word(Word::Emoji("family_man_woman_boy_boy")),
            },
            PositionalToken {
                source: uws,
                offset: 86,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 87,
                length: 4,
                token: Token::Word(Word::Emoji("brain")),
            },
            PositionalToken {
                source: uws,
                offset: 91,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
        ];

        let lib_res = uws.into_tokenizer(Default::default()).collect::<Vec<_>>();
        check_results(&result, &lib_res, uws);
        //print_result(&lib_res); panic!();
    }

    /*#[test]
    fn hashtags_mentions_urls() {
        let uws = "\nSome ##text with #hashtags and @other components\nadfa wdsfdf asdf asd http://asdfasdfsd.com/fasdfd/sadfsadf/sdfas/12312_12414/asdf?fascvx=fsfwer&dsdfasdf=fasdf#fasdf asdfa sdfa sdf\nasdfas df asd who@bla-bla.com asdfas df asdfsd\n";
        let result = vec![
            PositionalToken { source: uws, offset: 0, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { source: uws, offset: 1, length: 4, token: Token::Word(Word::Word("Some".to_string())) },
            PositionalToken { source: uws, offset: 5, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 6, length: 2, token: Token::Special(Special::Punctuation("##".to_string())) },
            PositionalToken { source: uws, offset: 8, length: 4, token: Token::Word(Word::Word("text".to_string())) },
            PositionalToken { source: uws, offset: 12, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 13, length: 4, token: Token::Word(Word::Word("with".to_string())) },
            PositionalToken { source: uws, offset: 17, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 18, length: 9, token: Token::Struct(Struct::Hashtag("hashtags".to_string())) },
            PositionalToken { source: uws, offset: 27, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 28, length: 3, token: Token::Word(Word::Word("and".to_string())) },
            PositionalToken { source: uws, offset: 31, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 32, length: 6, token: Token::Struct(Struct::Mention("other".to_string())) },
            PositionalToken { source: uws, offset: 38, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 39, length: 10, token: Token::Word(Word::Word("components".to_string())) },
            PositionalToken { source: uws, offset: 49, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { source: uws, offset: 50, length: 4, token: Token::Word(Word::Word("adfa".to_string())) },
            PositionalToken { source: uws, offset: 54, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 55, length: 6, token: Token::Word(Word::Word("wdsfdf".to_string())) },
            PositionalToken { source: uws, offset: 61, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 62, length: 4, token: Token::Word(Word::Word("asdf".to_string())) },
            PositionalToken { source: uws, offset: 66, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 67, length: 3, token: Token::Word(Word::Word("asd".to_string())) },
            PositionalToken { source: uws, offset: 70, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 71, length: 95, token: Token::Struct(Struct::Url("http://asdfasdfsd.com/fasdfd/sadfsadf/sdfas/12312_12414/asdf?fascvx=fsfwer&dsdfasdf=fasdf#fasdf".to_string())) },
            PositionalToken { source: uws, offset: 166, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 167, length: 5, token: Token::Word(Word::Word("asdfa".to_string())) },
            PositionalToken { source: uws, offset: 172, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 173, length: 4, token: Token::Word(Word::Word("sdfa".to_string())) },
            PositionalToken { source: uws, offset: 177, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 178, length: 3, token: Token::Word(Word::Word("sdf".to_string())) },
            PositionalToken { source: uws, offset: 181, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { source: uws, offset: 182, length: 6, token: Token::Word(Word::Word("asdfas".to_string())) },
            PositionalToken { source: uws, offset: 188, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 189, length: 2, token: Token::Word(Word::Word("df".to_string())) },
            PositionalToken { source: uws, offset: 191, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 192, length: 3, token: Token::Word(Word::Word("asd".to_string())) },
            PositionalToken { source: uws, offset: 195, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 196, length: 3, token: Token::Word(Word::Word("who".to_string())) },
            PositionalToken { source: uws, offset: 199, length: 4, token: Token::Struct(Struct::Mention("bla".to_string())) },
            PositionalToken { source: uws, offset: 203, length: 1, token: Token::Special(Special::Punctuation('-')) },
            PositionalToken { source: uws, offset: 204, length: 7, token: Token::Word(Word::Word("bla.com".to_string())) },
            PositionalToken { source: uws, offset: 211, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 212, length: 6, token: Token::Word(Word::Word("asdfas".to_string())) },
            PositionalToken { source: uws, offset: 218, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 219, length: 2, token: Token::Word(Word::Word("df".to_string())) },
            PositionalToken { source: uws, offset: 221, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { source: uws, offset: 222, length: 6, token: Token::Word(Word::Word("asdfsd".to_string())) },
            PositionalToken { source: uws, offset: 228, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            ];
        let lib_res = uws.into_tokenizer(TokenizerParams::complex()).collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!("")
    }*/

    /*#[test]
    fn bb_code() {
        let uws = "[Oxana Putan|1712640565] shared a [post|100001150683379_1873048549410150]. \nAndrew\n[link|https://www.facebook.com/100001150683379/posts/1873048549410150]\nДрузья мои, издатели, редакторы, просветители, культуртрегеры, субъекты мирового рынка и ту хум ит ещё мей консёрн.\nНа текущий момент я лишен былой подвижности, хоть и ковыляю по больничных коридорам по разным нуждам и за кипятком.\nВрачи обещают мне заживление отверстых ран моих в течение полугода и на этот период можно предполагать с уверенностью преимущественно домашний образ жизни.\n[|]";
        let result = vec![
            PositionalToken { offset: 0, length: 24, token: Token::BBCode { left: vec![
                PositionalToken { offset: 1, length: 5, token: Token::Word(Word::Word("Oxana".to_string())) },
                PositionalToken { offset: 6, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
                PositionalToken { offset: 7, length: 5, token: Token::Word(Word::Word("Putan".to_string())) },
                ], right: vec![
                PositionalToken { offset: 13, length: 10, token: Token::Word(Word::Number(Number::Integer(1712640565))) },
                ] } },
            PositionalToken { offset: 24, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 25, length: 6, token: Token::Word(Word::Word("shared".to_string())) },
            PositionalToken { offset: 31, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 32, length: 1, token: Token::Word(Word::Word("a".to_string())) },
            PositionalToken { offset: 33, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 34, length: 39, token: Token::BBCode { left: vec![
                PositionalToken { offset: 35, length: 4, token: Token::Word(Word::Word("post".to_string())) },
                ], right: vec![
                PositionalToken { offset: 40, length: 32, token: Token::Word(Word::Numerical(Numerical::Alphanumeric("100001150683379_1873048549410150".to_string()))) },
                ] } },
            PositionalToken { offset: 73, length: 1, token: Token::Special(Special::Punctuation('.')) },
            PositionalToken { offset: 74, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 75, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { offset: 76, length: 6, token: Token::Word(Word::Word("Andrew".to_string())) },
            PositionalToken { offset: 82, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { offset: 83, length: 70, token: Token::BBCode { left: vec![
                PositionalToken { offset: 84, length: 4, token: Token::Word(Word::Word("link".to_string())) },
                ], right: vec![
                PositionalToken { offset: 89, length: 63, token: Token::Struct(Struct::Url("https://www.facebook.com/100001150683379/posts/1873048549410150".to_string())) },
                ] } },
            PositionalToken { offset: 153, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { offset: 154, length: 12, token: Token::Word(Word::Word("Друзья".to_string())) },
            PositionalToken { offset: 166, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 167, length: 6, token: Token::Word(Word::Word("мои".to_string())) },
            PositionalToken { offset: 173, length: 1, token: Token::Special(Special::Punctuation(',')) },
            PositionalToken { offset: 174, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 175, length: 16, token: Token::Word(Word::Word("издатели".to_string())) },
            PositionalToken { offset: 191, length: 1, token: Token::Special(Special::Punctuation(',')) },
            PositionalToken { offset: 192, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 193, length: 18, token: Token::Word(Word::Word("редакторы".to_string())) },
            PositionalToken { offset: 211, length: 1, token: Token::Special(Special::Punctuation(',')) },
            PositionalToken { offset: 212, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 213, length: 24, token: Token::Word(Word::Word("просветители".to_string())) },
            PositionalToken { offset: 237, length: 1, token: Token::Special(Special::Punctuation(',')) },
            PositionalToken { offset: 238, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 239, length: 28, token: Token::Word(Word::Word("культуртрегеры".to_string())) },
            PositionalToken { offset: 267, length: 1, token: Token::Special(Special::Punctuation(',')) },
            PositionalToken { offset: 268, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 269, length: 16, token: Token::Word(Word::Word("субъекты".to_string())) },
            PositionalToken { offset: 285, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 286, length: 16, token: Token::Word(Word::Word("мирового".to_string())) },
            PositionalToken { offset: 302, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 303, length: 10, token: Token::Word(Word::Word("рынка".to_string())) },
            PositionalToken { offset: 313, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 314, length: 2, token: Token::Word(Word::Word("и".to_string())) },
            PositionalToken { offset: 316, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 317, length: 4, token: Token::Word(Word::Word("ту".to_string())) },
            PositionalToken { offset: 321, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 322, length: 6, token: Token::Word(Word::Word("хум".to_string())) },
            PositionalToken { offset: 328, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 329, length: 4, token: Token::Word(Word::Word("ит".to_string())) },
            PositionalToken { offset: 333, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 334, length: 6, token: Token::Word(Word::Word("ещё".to_string())) },
            PositionalToken { offset: 340, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 341, length: 6, token: Token::Word(Word::Word("мей".to_string())) },
            PositionalToken { offset: 347, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 348, length: 14, token: Token::Word(Word::Word("консёрн".to_string())) },
            PositionalToken { offset: 362, length: 1, token: Token::Special(Special::Punctuation('.')) },
            PositionalToken { offset: 363, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { offset: 364, length: 4, token: Token::Word(Word::Word("На".to_string())) },
            PositionalToken { offset: 368, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 369, length: 14, token: Token::Word(Word::Word("текущий".to_string())) },
            PositionalToken { offset: 383, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 384, length: 12, token: Token::Word(Word::Word("момент".to_string())) },
            PositionalToken { offset: 396, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 397, length: 2, token: Token::Word(Word::Word("я".to_string())) },
            PositionalToken { offset: 399, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 400, length: 10, token: Token::Word(Word::Word("лишен".to_string())) },
            PositionalToken { offset: 410, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 411, length: 10, token: Token::Word(Word::Word("былой".to_string())) },
            PositionalToken { offset: 421, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 422, length: 22, token: Token::Word(Word::Word("подвижности".to_string())) },
            PositionalToken { offset: 444, length: 1, token: Token::Special(Special::Punctuation(',')) },
            PositionalToken { offset: 445, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 446, length: 8, token: Token::Word(Word::Word("хоть".to_string())) },
            PositionalToken { offset: 454, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 455, length: 2, token: Token::Word(Word::Word("и".to_string())) },
            PositionalToken { offset: 457, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 458, length: 14, token: Token::Word(Word::Word("ковыляю".to_string())) },
            PositionalToken { offset: 472, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 473, length: 4, token: Token::Word(Word::Word("по".to_string())) },
            PositionalToken { offset: 477, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 478, length: 20, token: Token::Word(Word::Word("больничных".to_string())) },
            PositionalToken { offset: 498, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 499, length: 18, token: Token::Word(Word::Word("коридорам".to_string())) },
            PositionalToken { offset: 517, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 518, length: 4, token: Token::Word(Word::Word("по".to_string())) },
            PositionalToken { offset: 522, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 523, length: 12, token: Token::Word(Word::Word("разным".to_string())) },
            PositionalToken { offset: 535, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 536, length: 12, token: Token::Word(Word::Word("нуждам".to_string())) },
            PositionalToken { offset: 548, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 549, length: 2, token: Token::Word(Word::Word("и".to_string())) },
            PositionalToken { offset: 551, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 552, length: 4, token: Token::Word(Word::Word("за".to_string())) },
            PositionalToken { offset: 556, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 557, length: 16, token: Token::Word(Word::Word("кипятком".to_string())) },
            PositionalToken { offset: 573, length: 1, token: Token::Special(Special::Punctuation('.')) },
            PositionalToken { offset: 574, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { offset: 575, length: 10, token: Token::Word(Word::Word("Врачи".to_string())) },
            PositionalToken { offset: 585, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 586, length: 14, token: Token::Word(Word::Word("обещают".to_string())) },
            PositionalToken { offset: 600, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 601, length: 6, token: Token::Word(Word::Word("мне".to_string())) },
            PositionalToken { offset: 607, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 608, length: 20, token: Token::Word(Word::Word("заживление".to_string())) },
            PositionalToken { offset: 628, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 629, length: 18, token: Token::Word(Word::Word("отверстых".to_string())) },
            PositionalToken { offset: 647, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 648, length: 6, token: Token::Word(Word::Word("ран".to_string())) },
            PositionalToken { offset: 654, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 655, length: 8, token: Token::Word(Word::Word("моих".to_string())) },
            PositionalToken { offset: 663, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 664, length: 2, token: Token::Word(Word::Word("в".to_string())) },
            PositionalToken { offset: 666, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 667, length: 14, token: Token::Word(Word::Word("течение".to_string())) },
            PositionalToken { offset: 681, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 682, length: 16, token: Token::Word(Word::Word("полугода".to_string())) },
            PositionalToken { offset: 698, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 699, length: 2, token: Token::Word(Word::Word("и".to_string())) },
            PositionalToken { offset: 701, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 702, length: 4, token: Token::Word(Word::Word("на".to_string())) },
            PositionalToken { offset: 706, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 707, length: 8, token: Token::Word(Word::Word("этот".to_string())) },
            PositionalToken { offset: 715, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 716, length: 12, token: Token::Word(Word::Word("период".to_string())) },
            PositionalToken { offset: 728, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 729, length: 10, token: Token::Word(Word::Word("можно".to_string())) },
            PositionalToken { offset: 739, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 740, length: 24, token: Token::Word(Word::Word("предполагать".to_string())) },
            PositionalToken { offset: 764, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 765, length: 2, token: Token::Word(Word::Word("с".to_string())) },
            PositionalToken { offset: 767, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 768, length: 24, token: Token::Word(Word::Word("уверенностью".to_string())) },
            PositionalToken { offset: 792, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 793, length: 30, token: Token::Word(Word::Word("преимущественно".to_string())) },
            PositionalToken { offset: 823, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 824, length: 16, token: Token::Word(Word::Word("домашний".to_string())) },
            PositionalToken { offset: 840, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 841, length: 10, token: Token::Word(Word::Word("образ".to_string())) },
            PositionalToken { offset: 851, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 852, length: 10, token: Token::Word(Word::Word("жизни".to_string())) },
            PositionalToken { offset: 862, length: 1, token: Token::Special(Special::Punctuation('.')) },
            PositionalToken { offset: 863, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { offset: 864, length: 3, token: Token::BBCode { left: vec![
                ], right: vec![
                ] } },
            ];
        let lib_res = uws.into_tokenizer(TokenizerParams::complex()).collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        check_results(&result,&lib_res,uws);
    }*/

    #[test]
    fn html() {
        let uws = "<div class=\"article article_view \" id=\"article_view_-113039156_9551\" data-article-url=\"/@chaibuket-o-chem-ne-zabyt-25-noyabrya\" data-audio-context=\"article:-113039156_9551\"><h1  class=\"article_decoration_first article_decoration_last\" >День Мамы </h1><p  class=\"article_decoration_first article_decoration_last\" >День, когда поздравляют мам, бабушек, сестер и жён — это всемирный праздник, называемый «День Мамы». В настоящее время его отмечают почти в каждой стране, просто везде разные даты и способы празднования. </p><h3  class=\"article_decoration_first article_decoration_last\" ><span class='article_anchor_title'>\n  <span class='article_anchor_button' id='pochemu-my-ego-prazdnuem'></span>\n  <span class='article_anchor_fsymbol'>П</span>\n</span>ПОЧЕМУ МЫ ЕГО ПРАЗДНУЕМ</h3><p  class=\"article_decoration_first article_decoration_last article_decoration_before\" >В 1987 году комитет госдумы по делам женщин, семьи и молодежи выступил с предложением учредить «День мамы», а сам приказ был подписан уже 30 января 1988 года Борисом Ельциным. Было решено, что ежегодно в России празднество дня мамы будет выпадать на последнее воскресенье ноября. </p><figure data-type=\"101\" data-mode=\"\"  class=\"article_decoration_first article_decoration_last\" >\n  <div class=\"article_figure_content\" style=\"width: 1125px\">\n    <div class=\"article_figure_sizer_content\"><div class=\"article_object_sizer_wrap\" data-sizes=\"[{&quot;s&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c0ffd/pcNJaBH3NDo.jpg&quot;,75,50],&quot;m&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c0ffe/ozCLs2kHtRY.jpg&quot;,130,87],&quot;x&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c0fff/E4KtTNDydzE.jpg&quot;,604,403],&quot;y&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1000/1nLxpYKavzU.jpg&quot;,807,538],&quot;z&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1001/IgEODe90yEk.jpg&quot;,1125,750],&quot;o&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1002/01faNwVZ2_E.jpg&quot;,130,87],&quot;p&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1003/baDFzbdRP2s.jpg&quot;,200,133],&quot;q&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1004/CY4khI6KJKA.jpg&quot;,320,213],&quot;r&quot;:[&quot;https://pp.userapi.com/c849128/v849128704/c1005/NOvAJ6-VltY.jpg&quot;,510,340]}]\">\n  <img class=\"article_object_sizer_inner article_object_photo__image_blur\" src=\"https://pp.userapi.com/c849128/v849128704/c0ffd/pcNJaBH3NDo.jpg\" data-baseurl=\"\"/>\n  \n</div></div>\n    <div class=\"article_figure_sizer\" style=\"padding-bottom: 66.666666666667%\"></div>";
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 236,
                length: 8,
                token: Token::Word(Word::Word("День".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 244,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 245,
                length: 8,
                token: Token::Word(Word::Word("Мамы".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 253,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 321,
                length: 8,
                token: Token::Word(Word::Word("День".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 329,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 330,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 331,
                length: 10,
                token: Token::Word(Word::Word("когда".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 341,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 342,
                length: 22,
                token: Token::Word(Word::Word("поздравляют".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 364,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 365,
                length: 6,
                token: Token::Word(Word::Word("мам".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 371,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 372,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 373,
                length: 14,
                token: Token::Word(Word::Word("бабушек".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 387,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 388,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 389,
                length: 12,
                token: Token::Word(Word::Word("сестер".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 401,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 402,
                length: 2,
                token: Token::Word(Word::Word("и".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 404,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 405,
                length: 6,
                token: Token::Word(Word::Word("жён".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 411,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 412,
                length: 3,
                token: Token::Special(Special::Punctuation('—')),
            },
            PositionalToken {
                source: uws,
                offset: 415,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 416,
                length: 6,
                token: Token::Word(Word::Word("это".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 422,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 423,
                length: 18,
                token: Token::Word(Word::Word("всемирный".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 441,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 442,
                length: 16,
                token: Token::Word(Word::Word("праздник".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 458,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 459,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 460,
                length: 20,
                token: Token::Word(Word::Word("называемый".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 480,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 481,
                length: 2,
                token: Token::Special(Special::Punctuation('«')),
            },
            PositionalToken {
                source: uws,
                offset: 483,
                length: 8,
                token: Token::Word(Word::Word("День".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 491,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 492,
                length: 8,
                token: Token::Word(Word::Word("Мамы".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 500,
                length: 2,
                token: Token::Special(Special::Punctuation('»')),
            },
            PositionalToken {
                source: uws,
                offset: 502,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 503,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 504,
                length: 2,
                token: Token::Word(Word::Word("В".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 506,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 507,
                length: 18,
                token: Token::Word(Word::Word("настоящее".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 525,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 526,
                length: 10,
                token: Token::Word(Word::Word("время".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 536,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 537,
                length: 6,
                token: Token::Word(Word::Word("его".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 543,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 544,
                length: 16,
                token: Token::Word(Word::Word("отмечают".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 560,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 561,
                length: 10,
                token: Token::Word(Word::Word("почти".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 571,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 572,
                length: 2,
                token: Token::Word(Word::Word("в".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 574,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 575,
                length: 12,
                token: Token::Word(Word::Word("каждой".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 587,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 588,
                length: 12,
                token: Token::Word(Word::Word("стране".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 600,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 601,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 602,
                length: 12,
                token: Token::Word(Word::Word("просто".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 614,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 615,
                length: 10,
                token: Token::Word(Word::Word("везде".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 625,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 626,
                length: 12,
                token: Token::Word(Word::Word("разные".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 638,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 639,
                length: 8,
                token: Token::Word(Word::Word("даты".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 647,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 648,
                length: 2,
                token: Token::Word(Word::Word("и".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 650,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 651,
                length: 14,
                token: Token::Word(Word::Word("способы".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 665,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 666,
                length: 24,
                token: Token::Word(Word::Word("празднования".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 690,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 691,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 794,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 795,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 870,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 871,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 910,
                length: 2,
                token: Token::Word(Word::Word("П".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 919,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 927,
                length: 12,
                token: Token::Word(Word::Word("ПОЧЕМУ".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 939,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 940,
                length: 4,
                token: Token::Word(Word::Word("МЫ".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 944,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 945,
                length: 6,
                token: Token::Word(Word::Word("ЕГО".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 951,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 952,
                length: 18,
                token: Token::Word(Word::Word("ПРАЗДНУЕМ".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1063,
                length: 2,
                token: Token::Word(Word::Word("В".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1065,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1066,
                length: 4,
                token: Token::Word(Word::Number(Number::Integer(1987))),
            },
            PositionalToken {
                source: uws,
                offset: 1070,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1071,
                length: 8,
                token: Token::Word(Word::Word("году".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1079,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1080,
                length: 14,
                token: Token::Word(Word::Word("комитет".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1094,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1095,
                length: 14,
                token: Token::Word(Word::Word("госдумы".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1109,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1110,
                length: 4,
                token: Token::Word(Word::Word("по".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1114,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1115,
                length: 10,
                token: Token::Word(Word::Word("делам".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1125,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1126,
                length: 12,
                token: Token::Word(Word::Word("женщин".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1138,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 1139,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1140,
                length: 10,
                token: Token::Word(Word::Word("семьи".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1150,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1151,
                length: 2,
                token: Token::Word(Word::Word("и".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1153,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1154,
                length: 16,
                token: Token::Word(Word::Word("молодежи".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1170,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1171,
                length: 16,
                token: Token::Word(Word::Word("выступил".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1187,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1188,
                length: 2,
                token: Token::Word(Word::Word("с".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1190,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1191,
                length: 24,
                token: Token::Word(Word::Word("предложением".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1215,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1216,
                length: 16,
                token: Token::Word(Word::Word("учредить".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1232,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1233,
                length: 2,
                token: Token::Special(Special::Punctuation('«')),
            },
            PositionalToken {
                source: uws,
                offset: 1235,
                length: 8,
                token: Token::Word(Word::Word("День".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1243,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1244,
                length: 8,
                token: Token::Word(Word::Word("мамы".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1252,
                length: 2,
                token: Token::Special(Special::Punctuation('»')),
            },
            PositionalToken {
                source: uws,
                offset: 1254,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 1255,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1256,
                length: 2,
                token: Token::Word(Word::Word("а".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1258,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1259,
                length: 6,
                token: Token::Word(Word::Word("сам".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1265,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1266,
                length: 12,
                token: Token::Word(Word::Word("приказ".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1278,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1279,
                length: 6,
                token: Token::Word(Word::Word("был".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1285,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1286,
                length: 16,
                token: Token::Word(Word::Word("подписан".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1302,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1303,
                length: 6,
                token: Token::Word(Word::Word("уже".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1309,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1310,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(30))),
            },
            PositionalToken {
                source: uws,
                offset: 1312,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1313,
                length: 12,
                token: Token::Word(Word::Word("января".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1325,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1326,
                length: 4,
                token: Token::Word(Word::Number(Number::Integer(1988))),
            },
            PositionalToken {
                source: uws,
                offset: 1330,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1331,
                length: 8,
                token: Token::Word(Word::Word("года".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1339,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1340,
                length: 14,
                token: Token::Word(Word::Word("Борисом".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1354,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1355,
                length: 16,
                token: Token::Word(Word::Word("Ельциным".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1371,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 1372,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1373,
                length: 8,
                token: Token::Word(Word::Word("Было".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1381,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1382,
                length: 12,
                token: Token::Word(Word::Word("решено".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1394,
                length: 1,
                token: Token::Special(Special::Punctuation(',')),
            },
            PositionalToken {
                source: uws,
                offset: 1395,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1396,
                length: 6,
                token: Token::Word(Word::Word("что".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1402,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1403,
                length: 16,
                token: Token::Word(Word::Word("ежегодно".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1419,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1420,
                length: 2,
                token: Token::Word(Word::Word("в".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1422,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1423,
                length: 12,
                token: Token::Word(Word::Word("России".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1435,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1436,
                length: 22,
                token: Token::Word(Word::Word("празднество".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1458,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1459,
                length: 6,
                token: Token::Word(Word::Word("дня".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1465,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1466,
                length: 8,
                token: Token::Word(Word::Word("мамы".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1474,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1475,
                length: 10,
                token: Token::Word(Word::Word("будет".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1485,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1486,
                length: 16,
                token: Token::Word(Word::Word("выпадать".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1502,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1503,
                length: 4,
                token: Token::Word(Word::Word("на".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1507,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1508,
                length: 18,
                token: Token::Word(Word::Word("последнее".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1526,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1527,
                length: 22,
                token: Token::Word(Word::Word("воскресенье".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1549,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1550,
                length: 12,
                token: Token::Word(Word::Word("ноября".to_string())),
            },
            PositionalToken {
                source: uws,
                offset: 1562,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 1563,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1664,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 1665,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 1725,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 1726,
                length: 4,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 2725,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 2726,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 2888,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 2889,
                length: 2,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 2891,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 2904,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Newline)),
            },
            PositionalToken {
                source: uws,
                offset: 2905,
                length: 4,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
        ];

        let text = Text::new({
            uws.into_source()
                .pipe(tagger::Builder::new().create().into_breaker())
                .pipe(entities::Builder::new().create().into_piped())
                .into_separator()
        })
        .unwrap();

        let lib_res = text
            .into_tokenizer(TokenizerParams::v1())
            .filter_map(|tt| tt.into_original_token_1())
            .collect::<Vec<_>>();

        check_results(&result, &lib_res, uws);
    }

    /*#[test]
    fn vk_bbcode() {
        let uws = "[club113623432|💜💜💜 - для девушек] \n[club113623432|💛💛💛 - для сохраненок]";
        let result = vec![
            PositionalToken { offset: 0, length: 52, token: Token::BBCode { left: vec![
                PositionalToken { offset: 1, length: 13, token: Token::Word(Word::Numerical(Numerical::Alphanumeric("club113623432".to_string()))) },
                ], right: vec![
                PositionalToken { offset: 15, length: 4, token: Token::Word(Word::Emoji("purple_heart")) },
                PositionalToken { offset: 19, length: 4, token: Token::Word(Word::Emoji("purple_heart")) },
                PositionalToken { offset: 23, length: 4, token: Token::Word(Word::Emoji("purple_heart")) },
                PositionalToken { offset: 27, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
                PositionalToken { offset: 28, length: 1, token: Token::Special(Special::Punctuation('-')) },
                PositionalToken { offset: 29, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
                PositionalToken { offset: 30, length: 6, token: Token::Word(Word::Word("для".to_string())) },
                PositionalToken { offset: 36, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
                PositionalToken { offset: 37, length: 14, token: Token::Word(Word::Word("девушек".to_string())) },
                ] } },
            PositionalToken { offset: 52, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
            PositionalToken { offset: 53, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            PositionalToken { offset: 54, length: 58, token: Token::BBCode { left: vec![
                PositionalToken { offset: 55, length: 13, token: Token::Word(Word::Numerical(Numerical::Alphanumeric("club113623432".to_string()))) },
                ], right: vec![
                PositionalToken { offset: 69, length: 4, token: Token::Word(Word::Emoji("yellow_heart")) },
                PositionalToken { offset: 73, length: 4, token: Token::Word(Word::Emoji("yellow_heart")) },
                PositionalToken { offset: 77, length: 4, token: Token::Word(Word::Emoji("yellow_heart")) },
                PositionalToken { offset: 81, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
                PositionalToken { offset: 82, length: 1, token: Token::Special(Special::Punctuation('-')) },
                PositionalToken { offset: 83, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
                PositionalToken { offset: 84, length: 6, token: Token::Word(Word::Word("для".to_string())) },
                PositionalToken { offset: 90, length: 1, token: Token::Special(Special::Separator(Separator::Space)) },
                PositionalToken { offset: 91, length: 20, token: Token::Word(Word::Word("сохраненок".to_string())) },
                ] } },
            ];
        let lib_res = uws.into_tokenizer(TokenizerParams::complex()).collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        check_results(&result,&lib_res,uws);
    }*/

    /*#[test]
    fn text_href_and_html () {
        let uws = "https://youtu.be/dQErLQZw3qA</a></p><figure data-type=\"102\" data-mode=\"\"  class=\"article_decoration_first article_decoration_last\" >\n";
        let result =  vec![
            PositionalToken { offset: 0, length: 28, token: Token::Struct(Struct::Url("https://youtu.be/dQErLQZw3qA".to_string())) },
            PositionalToken { offset: 132, length: 1, token: Token::Special(Special::Separator(Separator::Newline)) },
            ];
        let lib_res = uws.into_tokenizer(TokenizerParams::v1()).unwrap().collect::<Vec<_>>();
        check_results(&result,&lib_res,uws);
        //print_result(&lib_res); panic!("")
    }*/

    #[test]
    fn numerical_no_split() {
        let uws = "12.02.18 31.28.34 23.11.2018 123.568.365.234.578 127.0.0.1 1st 1кг 123123афываыв 12321фвафыов234выалфо 12_123_343.4234_4234";
        let lib_res = uws.into_tokenizer(Default::default()).collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 8,
                token: Token::Word(Word::Numerical(Numerical::DotSeparated(
                    "12.02.18".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 8,
                token: Token::Word(Word::Numerical(Numerical::DotSeparated(
                    "31.28.34".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 10,
                token: Token::Word(Word::Numerical(Numerical::DotSeparated(
                    "23.11.2018".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 28,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 29,
                length: 19,
                token: Token::Word(Word::Numerical(Numerical::DotSeparated(
                    "123.568.365.234.578".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 48,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 49,
                length: 9,
                token: Token::Word(Word::Numerical(Numerical::DotSeparated(
                    "127.0.0.1".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 58,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 59,
                length: 3,
                token: Token::Word(Word::Numerical(Numerical::Measures("1st".to_string()))),
            },
            PositionalToken {
                source: uws,
                offset: 62,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 63,
                length: 5,
                token: Token::Word(Word::Numerical(Numerical::Measures("1кг".to_string()))),
            },
            PositionalToken {
                source: uws,
                offset: 68,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 69,
                length: 20,
                token: Token::Word(Word::Numerical(Numerical::Measures(
                    "123123афываыв".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 89,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 90,
                length: 34,
                token: Token::Word(Word::Numerical(Numerical::Alphanumeric(
                    "12321фвафыов234выалфо".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 124,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 125,
                length: 20,
                token: Token::Word(Word::Numerical(Numerical::Alphanumeric(
                    "12_123_343.4234_4234".to_string(),
                ))),
            },
        ];
        check_results(&result, &lib_res, uws);
    }

    #[test]
    fn numerical_default() {
        let uws = "12.02.18 31.28.34 23.11.2018 123.568.365.234.578 127.0.0.1 1st 1кг 123123афываыв 12321фвафыов234выалфо 12_123_343.4234_4234";
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        //print_result(&lib_res); panic!("");
        let result = vec![
            PositionalToken {
                source: uws,
                offset: 0,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(12))),
            },
            PositionalToken {
                source: uws,
                offset: 2,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 3,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(2))),
            },
            PositionalToken {
                source: uws,
                offset: 5,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 6,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(18))),
            },
            PositionalToken {
                source: uws,
                offset: 8,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 9,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(31))),
            },
            PositionalToken {
                source: uws,
                offset: 11,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 12,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(28))),
            },
            PositionalToken {
                source: uws,
                offset: 14,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 15,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(34))),
            },
            PositionalToken {
                source: uws,
                offset: 17,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 18,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(23))),
            },
            PositionalToken {
                source: uws,
                offset: 20,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 21,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(11))),
            },
            PositionalToken {
                source: uws,
                offset: 23,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 24,
                length: 4,
                token: Token::Word(Word::Number(Number::Integer(2018))),
            },
            PositionalToken {
                source: uws,
                offset: 28,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 29,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(123))),
            },
            PositionalToken {
                source: uws,
                offset: 32,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 33,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(568))),
            },
            PositionalToken {
                source: uws,
                offset: 36,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 37,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(365))),
            },
            PositionalToken {
                source: uws,
                offset: 40,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 41,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(234))),
            },
            PositionalToken {
                source: uws,
                offset: 44,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 45,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(578))),
            },
            PositionalToken {
                source: uws,
                offset: 48,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 49,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(127))),
            },
            PositionalToken {
                source: uws,
                offset: 52,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 53,
                length: 1,
                token: Token::Word(Word::Number(Number::Integer(0))),
            },
            PositionalToken {
                source: uws,
                offset: 54,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 55,
                length: 1,
                token: Token::Word(Word::Number(Number::Integer(0))),
            },
            PositionalToken {
                source: uws,
                offset: 56,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 57,
                length: 1,
                token: Token::Word(Word::Number(Number::Integer(1))),
            },
            PositionalToken {
                source: uws,
                offset: 58,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 59,
                length: 3,
                token: Token::Word(Word::Numerical(Numerical::Measures("1st".to_string()))),
            },
            PositionalToken {
                source: uws,
                offset: 62,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 63,
                length: 5,
                token: Token::Word(Word::Numerical(Numerical::Measures("1кг".to_string()))),
            },
            PositionalToken {
                source: uws,
                offset: 68,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 69,
                length: 20,
                token: Token::Word(Word::Numerical(Numerical::Measures(
                    "123123афываыв".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 89,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 90,
                length: 34,
                token: Token::Word(Word::Numerical(Numerical::Alphanumeric(
                    "12321фвафыов234выалфо".to_string(),
                ))),
            },
            PositionalToken {
                source: uws,
                offset: 124,
                length: 1,
                token: Token::Special(Special::Separator(Separator::Space)),
            },
            PositionalToken {
                source: uws,
                offset: 125,
                length: 2,
                token: Token::Word(Word::Number(Number::Integer(12))),
            },
            PositionalToken {
                source: uws,
                offset: 127,
                length: 1,
                token: Token::Special(Special::Punctuation('_')),
            },
            PositionalToken {
                source: uws,
                offset: 128,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(123))),
            },
            PositionalToken {
                source: uws,
                offset: 131,
                length: 1,
                token: Token::Special(Special::Punctuation('_')),
            },
            PositionalToken {
                source: uws,
                offset: 132,
                length: 3,
                token: Token::Word(Word::Number(Number::Integer(343))),
            },
            PositionalToken {
                source: uws,
                offset: 135,
                length: 1,
                token: Token::Special(Special::Punctuation('.')),
            },
            PositionalToken {
                source: uws,
                offset: 136,
                length: 4,
                token: Token::Word(Word::Number(Number::Integer(4234))),
            },
            PositionalToken {
                source: uws,
                offset: 140,
                length: 1,
                token: Token::Special(Special::Punctuation('_')),
            },
            PositionalToken {
                source: uws,
                offset: 141,
                length: 4,
                token: Token::Word(Word::Number(Number::Integer(4234))),
            },
        ];
        check_results(&result, &lib_res, uws);
    }

    /*#[test]
        fn new_test() {
            let uws = "";
            let lib_res = uws.into_tokenizer(TokenizerParams::v1()).unwrap().collect::<Vec<_>>();
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
        let (uws, result) = get_lang_test(Lang::Zho);
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, &uws);
    }

    #[test]
    fn test_lang_jpn() {
        let (uws, result) = get_lang_test(Lang::Jpn);
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, &uws);
    }

    #[test]
    fn test_lang_kor() {
        let (uws, result) = get_lang_test(Lang::Kor);
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, &uws);
    }

    #[test]
    fn test_lang_ara() {
        let (uws, result) = get_lang_test(Lang::Ara);
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, &uws);
    }

    #[test]
    fn test_lang_ell() {
        let (uws, result) = get_lang_test(Lang::Ell);
        let lib_res = uws
            .into_tokenizer(TokenizerParams::v1())
            .collect::<Vec<_>>();
        check_results(&result, &lib_res, &uws);
    }

    fn get_lang_test(lng: Lang) -> (String, Vec<PositionalToken>) {
        let uws = match lng {
            Lang::Zho => "美国电视连续剧《超人前传》的第一集《试播集》于2001年10月16日在電視網首播，剧集主创人阿尔弗雷德·高夫和迈尔斯·米勒編劇，大卫·努特尔执导。这一试播首次向观众引荐了克拉克·肯特一角，他是位拥有超能力的外星孤儿，与家人和朋友一起在堪薩斯州虚构小镇斯莫维尔生活。在这一集里，肯特首度得知自己的来历，同时还需要阻止一位学生试图杀死镇上高中多名学生的报复之举。本集节目里引入了多个之后将贯穿全季甚至整部剧集的主题元素，例如几位主要角色之间的三角恋情。电视剧在加拿大溫哥華取景，旨在选用其“美国中产阶级”景观，主创人花了5个月的时间专门用于为主角物色合适的演员。试播集在所有演员选好4天后正式开拍。由于时间上的限制，剧组无法搭建好实体外景，因此只能使用计算机绘图技术将数字化的外景插入到镜头中。节目一经上映就打破了电视网的多项收视纪录，并且获得了评论员的普遍好评和多个奖项提名，并在其中两项上胜出",
            Lang::Kor =>  "플레이스테이션 은 소니 컴퓨터 엔터테인먼트가 개발한 세 번째 가정용 게임기이다. 마이크로소프트의 엑스박스 360, 닌텐도의 Wii와 경쟁하고 있다. 이전 제품에서 온라인 플레이 기능을 비디오 게임 개발사에 전적으로 의존하던 것과 달리 통합 온라인 게임 서비스인 플레이스테이션 네트워크 서비스를 발매와 함께 시작해 제공하고 있으며, 탄탄한 멀티미디어 재생 기능, 플레이스테이션 포터블과의 연결, 고화질 광학 디스크 포맷인 블루레이 디스크 재생 기능 등의 기능을 갖추고 있다. 2006년 11월 11일에 일본에서 처음으로 출시했으며, 11월 17일에는 북미 지역, 2007년 3월 23일에는 유럽과 오세아니아 지역에서, 대한민국의 경우 6월 5일부터 일주일간 예약판매를 실시해, 매일 준비한 수량이 동이 나는 등 많은 관심을 받았으며 6월 16일에 정식 출시 행사를 열었다",
            Lang::Jpn => "熊野三山本願所は、15世紀末以降における熊野三山（熊野本宮、熊野新宮、熊野那智）の造営・修造のための勧進を担った組織の総称。 熊野三山を含めて、日本における古代から中世前半にかけての寺社の造営は、寺社領経営のような恒常的財源、幕府や朝廷などからの一時的な造営料所の寄進、あるいは公権力からの臨時の保護によって行われていた。しかしながら、熊野三山では、これらの財源はすべて15世紀半ばまでに実効性を失った",
            Lang::Ara => "لشکرکشی‌های روس‌های وارنگی به دریای خزر مجموعه‌ای از حملات نظامی در بین سال‌های ۸۶۴ تا ۱۰۴۱ میلادی به سواحل دریای خزر بوده‌است. روس‌های وارنگی ابتدا در قرن نهم میلادی به عنوان بازرگانان پوست، عسل و برده در سرزمین‌های اسلامی(سرکلند) ظاهر شدند. این بازرگانان در مسیر تجاری ولگا به خرید و فروش می‌پرداختند. نخستین حملهٔ آنان در فاصله سال‌های ۸۶۴ تا ۸۸۴ میلادی در مقیاسی کوچک علیه علویان طبرستان رخ داد. نخستین یورش بزرگ روس‌ها در سال ۹۱۳ رخ داد و آنان با ۵۰۰ فروند درازکشتی شهر گرگان و اطراف آن را غارت کردند. آن‌ها در این حمله مقداری کالا و برده را به تاراج بردند و در راه بازگشتن به سمت شمال، در دلتای ولگا، مورد حملهٔ خزرهای مسلمان قرار گرفتند و بعضی از آنان موفق به فرار شدند، ولی در میانهٔ ولگا به قتل رسیدند. دومین هجوم بزرگ روس‌ها به دریای خزر در سال ۹۴۳ به وقوع پیوست. در این دوره ایگور یکم، حاکم روس کیف، رهبری روس‌ها را در دست داشت. روس‌ها پس از توافق با دولت خزرها برای عبور امن از منطقه، تا رود کورا و اعماق قفقاز پیش رفتند و در سال ۹۴۳ موفق شدند بندر بردعه، پایتخت اران (جمهوری آذربایجان کنونی)، را تصرف کنند. روس‌ها در آنجا به مدت چند ماه ماندند و بسیاری از ساکنان شهر را کشتند و از راه غارت‌گری اموالی را به تاراج بردند. تنها دلیل بازگشت آنان ",
            Lang::Ell => "Το Πρόγραμμα υλοποιείται εξ ολοκλήρου από απόσταση και μπορεί να συμμετέχει κάθε εμπλεκόμενος στη ή/και ενδιαφερόμενος για τη διδασκαλία της Ελληνικής ως δεύτερης/ξένης γλώσσας στην Ελλάδα και στο εξωτερικό, αρκεί να είναι απόφοιτος ελληνικής φιλολογίας, ξένων φιλολογιών, παιδαγωγικών τμημάτων, θεολογικών σχολών ή άλλων πανεπιστημιακών τμημάτων ελληνικών ή ισότιμων ξένων πανεπιστημίων. Υπό όρους γίνονται δεκτοί υποψήφιοι που δεν έχουν ολοκληρώσει σπουδές τριτοβάθμιας εκπαίδευσης.",
        };
        let tokens = match lng {
            Lang::Zho => vec![
                PositionalToken {
                    source: uws,
                    offset: 0,
                    length: 3,
                    token: Token::Word(Word::Word("美".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 3,
                    length: 3,
                    token: Token::Word(Word::Word("国".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 6,
                    length: 3,
                    token: Token::Word(Word::Word("电".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 9,
                    length: 3,
                    token: Token::Word(Word::Word("视".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 12,
                    length: 3,
                    token: Token::Word(Word::Word("连".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 15,
                    length: 3,
                    token: Token::Word(Word::Word("续".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 18,
                    length: 3,
                    token: Token::Word(Word::Word("剧".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 21,
                    length: 3,
                    token: Token::Special(Special::Punctuation('《')),
                },
                PositionalToken {
                    source: uws,
                    offset: 24,
                    length: 3,
                    token: Token::Word(Word::Word("超".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 27,
                    length: 3,
                    token: Token::Word(Word::Word("人".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 30,
                    length: 3,
                    token: Token::Word(Word::Word("前".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 33,
                    length: 3,
                    token: Token::Word(Word::Word("传".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 36,
                    length: 3,
                    token: Token::Special(Special::Punctuation('》')),
                },
                PositionalToken {
                    source: uws,
                    offset: 39,
                    length: 3,
                    token: Token::Word(Word::Word("的".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 42,
                    length: 3,
                    token: Token::Word(Word::Word("第".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 45,
                    length: 3,
                    token: Token::Word(Word::Word("一".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 48,
                    length: 3,
                    token: Token::Word(Word::Word("集".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 51,
                    length: 3,
                    token: Token::Special(Special::Punctuation('《')),
                },
                PositionalToken {
                    source: uws,
                    offset: 54,
                    length: 3,
                    token: Token::Word(Word::Word("试".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 57,
                    length: 3,
                    token: Token::Word(Word::Word("播".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 60,
                    length: 3,
                    token: Token::Word(Word::Word("集".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 63,
                    length: 3,
                    token: Token::Special(Special::Punctuation('》')),
                },
                PositionalToken {
                    source: uws,
                    offset: 66,
                    length: 3,
                    token: Token::Word(Word::Word("于".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 69,
                    length: 4,
                    token: Token::Word(Word::Number(Number::Integer(2001))),
                },
                PositionalToken {
                    source: uws,
                    offset: 73,
                    length: 3,
                    token: Token::Word(Word::Word("年".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 76,
                    length: 2,
                    token: Token::Word(Word::Number(Number::Integer(10))),
                },
                PositionalToken {
                    source: uws,
                    offset: 78,
                    length: 3,
                    token: Token::Word(Word::Word("月".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 81,
                    length: 2,
                    token: Token::Word(Word::Number(Number::Integer(16))),
                },
                PositionalToken {
                    source: uws,
                    offset: 83,
                    length: 3,
                    token: Token::Word(Word::Word("日".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 86,
                    length: 3,
                    token: Token::Word(Word::Word("在".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 89,
                    length: 3,
                    token: Token::Word(Word::Word("電".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 92,
                    length: 3,
                    token: Token::Word(Word::Word("視".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 95,
                    length: 3,
                    token: Token::Word(Word::Word("網".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 98,
                    length: 3,
                    token: Token::Word(Word::Word("首".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 101,
                    length: 3,
                    token: Token::Word(Word::Word("播".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 104,
                    length: 3,
                    token: Token::Special(Special::Punctuation('，')),
                },
                PositionalToken {
                    source: uws,
                    offset: 107,
                    length: 3,
                    token: Token::Word(Word::Word("剧".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 110,
                    length: 3,
                    token: Token::Word(Word::Word("集".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 113,
                    length: 3,
                    token: Token::Word(Word::Word("主".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 116,
                    length: 3,
                    token: Token::Word(Word::Word("创".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 119,
                    length: 3,
                    token: Token::Word(Word::Word("人".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 122,
                    length: 3,
                    token: Token::Word(Word::Word("阿".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 125,
                    length: 3,
                    token: Token::Word(Word::Word("尔".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 128,
                    length: 3,
                    token: Token::Word(Word::Word("弗".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 131,
                    length: 3,
                    token: Token::Word(Word::Word("雷".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 134,
                    length: 3,
                    token: Token::Word(Word::Word("德".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 137,
                    length: 2,
                    token: Token::Special(Special::Punctuation('·')),
                },
                PositionalToken {
                    source: uws,
                    offset: 139,
                    length: 3,
                    token: Token::Word(Word::Word("高".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 142,
                    length: 3,
                    token: Token::Word(Word::Word("夫".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 145,
                    length: 3,
                    token: Token::Word(Word::Word("和".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 148,
                    length: 3,
                    token: Token::Word(Word::Word("迈".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 151,
                    length: 3,
                    token: Token::Word(Word::Word("尔".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 154,
                    length: 3,
                    token: Token::Word(Word::Word("斯".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 157,
                    length: 2,
                    token: Token::Special(Special::Punctuation('·')),
                },
                PositionalToken {
                    source: uws,
                    offset: 159,
                    length: 3,
                    token: Token::Word(Word::Word("米".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 162,
                    length: 3,
                    token: Token::Word(Word::Word("勒".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 165,
                    length: 3,
                    token: Token::Word(Word::Word("編".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 168,
                    length: 3,
                    token: Token::Word(Word::Word("劇".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 171,
                    length: 3,
                    token: Token::Special(Special::Punctuation('，')),
                },
                PositionalToken {
                    source: uws,
                    offset: 174,
                    length: 3,
                    token: Token::Word(Word::Word("大".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 177,
                    length: 3,
                    token: Token::Word(Word::Word("卫".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 180,
                    length: 2,
                    token: Token::Special(Special::Punctuation('·')),
                },
                PositionalToken {
                    source: uws,
                    offset: 182,
                    length: 3,
                    token: Token::Word(Word::Word("努".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 185,
                    length: 3,
                    token: Token::Word(Word::Word("特".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 188,
                    length: 3,
                    token: Token::Word(Word::Word("尔".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 191,
                    length: 3,
                    token: Token::Word(Word::Word("执".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 194,
                    length: 3,
                    token: Token::Word(Word::Word("导".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 197,
                    length: 3,
                    token: Token::Special(Special::Punctuation('。')),
                },
                PositionalToken {
                    source: uws,
                    offset: 200,
                    length: 3,
                    token: Token::Word(Word::Word("这".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 203,
                    length: 3,
                    token: Token::Word(Word::Word("一".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 206,
                    length: 3,
                    token: Token::Word(Word::Word("试".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 209,
                    length: 3,
                    token: Token::Word(Word::Word("播".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 212,
                    length: 3,
                    token: Token::Word(Word::Word("首".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 215,
                    length: 3,
                    token: Token::Word(Word::Word("次".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 218,
                    length: 3,
                    token: Token::Word(Word::Word("向".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 221,
                    length: 3,
                    token: Token::Word(Word::Word("观".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 224,
                    length: 3,
                    token: Token::Word(Word::Word("众".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 227,
                    length: 3,
                    token: Token::Word(Word::Word("引".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 230,
                    length: 3,
                    token: Token::Word(Word::Word("荐".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 233,
                    length: 3,
                    token: Token::Word(Word::Word("了".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 236,
                    length: 3,
                    token: Token::Word(Word::Word("克".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 239,
                    length: 3,
                    token: Token::Word(Word::Word("拉".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 242,
                    length: 3,
                    token: Token::Word(Word::Word("克".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 245,
                    length: 2,
                    token: Token::Special(Special::Punctuation('·')),
                },
                PositionalToken {
                    source: uws,
                    offset: 247,
                    length: 3,
                    token: Token::Word(Word::Word("肯".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 250,
                    length: 3,
                    token: Token::Word(Word::Word("特".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 253,
                    length: 3,
                    token: Token::Word(Word::Word("一".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 256,
                    length: 3,
                    token: Token::Word(Word::Word("角".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 259,
                    length: 3,
                    token: Token::Special(Special::Punctuation('，')),
                },
                PositionalToken {
                    source: uws,
                    offset: 262,
                    length: 3,
                    token: Token::Word(Word::Word("他".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 265,
                    length: 3,
                    token: Token::Word(Word::Word("是".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 268,
                    length: 3,
                    token: Token::Word(Word::Word("位".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 271,
                    length: 3,
                    token: Token::Word(Word::Word("拥".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 274,
                    length: 3,
                    token: Token::Word(Word::Word("有".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 277,
                    length: 3,
                    token: Token::Word(Word::Word("超".to_string())),
                },
            ],
            Lang::Jpn => vec![
                PositionalToken {
                    source: uws,
                    offset: 0,
                    length: 3,
                    token: Token::Word(Word::Word("熊".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 3,
                    length: 3,
                    token: Token::Word(Word::Word("野".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 6,
                    length: 3,
                    token: Token::Word(Word::Word("三".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 9,
                    length: 3,
                    token: Token::Word(Word::Word("山".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 12,
                    length: 3,
                    token: Token::Word(Word::Word("本".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 15,
                    length: 3,
                    token: Token::Word(Word::Word("願".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 18,
                    length: 3,
                    token: Token::Word(Word::Word("所".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 21,
                    length: 3,
                    token: Token::Word(Word::Word("は".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 24,
                    length: 3,
                    token: Token::Special(Special::Punctuation('、')),
                },
                PositionalToken {
                    source: uws,
                    offset: 27,
                    length: 2,
                    token: Token::Word(Word::Number(Number::Integer(15))),
                },
                PositionalToken {
                    source: uws,
                    offset: 29,
                    length: 3,
                    token: Token::Word(Word::Word("世".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 32,
                    length: 3,
                    token: Token::Word(Word::Word("紀".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 35,
                    length: 3,
                    token: Token::Word(Word::Word("末".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 38,
                    length: 3,
                    token: Token::Word(Word::Word("以".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 41,
                    length: 3,
                    token: Token::Word(Word::Word("降".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 44,
                    length: 3,
                    token: Token::Word(Word::Word("に".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 47,
                    length: 3,
                    token: Token::Word(Word::Word("お".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 50,
                    length: 3,
                    token: Token::Word(Word::Word("け".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 53,
                    length: 3,
                    token: Token::Word(Word::Word("る".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 56,
                    length: 3,
                    token: Token::Word(Word::Word("熊".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 59,
                    length: 3,
                    token: Token::Word(Word::Word("野".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 62,
                    length: 3,
                    token: Token::Word(Word::Word("三".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 65,
                    length: 3,
                    token: Token::Word(Word::Word("山".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 68,
                    length: 3,
                    token: Token::Special(Special::Punctuation('（')),
                },
                PositionalToken {
                    source: uws,
                    offset: 71,
                    length: 3,
                    token: Token::Word(Word::Word("熊".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 74,
                    length: 3,
                    token: Token::Word(Word::Word("野".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 77,
                    length: 3,
                    token: Token::Word(Word::Word("本".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 80,
                    length: 3,
                    token: Token::Word(Word::Word("宮".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 83,
                    length: 3,
                    token: Token::Special(Special::Punctuation('、')),
                },
                PositionalToken {
                    source: uws,
                    offset: 86,
                    length: 3,
                    token: Token::Word(Word::Word("熊".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 89,
                    length: 3,
                    token: Token::Word(Word::Word("野".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 92,
                    length: 3,
                    token: Token::Word(Word::Word("新".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 95,
                    length: 3,
                    token: Token::Word(Word::Word("宮".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 98,
                    length: 3,
                    token: Token::Special(Special::Punctuation('、')),
                },
                PositionalToken {
                    source: uws,
                    offset: 101,
                    length: 3,
                    token: Token::Word(Word::Word("熊".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 104,
                    length: 3,
                    token: Token::Word(Word::Word("野".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 107,
                    length: 3,
                    token: Token::Word(Word::Word("那".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 110,
                    length: 3,
                    token: Token::Word(Word::Word("智".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 113,
                    length: 3,
                    token: Token::Special(Special::Punctuation('）')),
                },
                PositionalToken {
                    source: uws,
                    offset: 116,
                    length: 3,
                    token: Token::Word(Word::Word("の".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 119,
                    length: 3,
                    token: Token::Word(Word::Word("造".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 122,
                    length: 3,
                    token: Token::Word(Word::Word("営".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 125,
                    length: 3,
                    token: Token::Special(Special::Punctuation('・')),
                },
                PositionalToken {
                    source: uws,
                    offset: 128,
                    length: 3,
                    token: Token::Word(Word::Word("修".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 131,
                    length: 3,
                    token: Token::Word(Word::Word("造".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 134,
                    length: 3,
                    token: Token::Word(Word::Word("の".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 137,
                    length: 3,
                    token: Token::Word(Word::Word("た".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 140,
                    length: 3,
                    token: Token::Word(Word::Word("め".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 143,
                    length: 3,
                    token: Token::Word(Word::Word("の".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 146,
                    length: 3,
                    token: Token::Word(Word::Word("勧".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 149,
                    length: 3,
                    token: Token::Word(Word::Word("進".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 152,
                    length: 3,
                    token: Token::Word(Word::Word("を".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 155,
                    length: 3,
                    token: Token::Word(Word::Word("担".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 158,
                    length: 3,
                    token: Token::Word(Word::Word("っ".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 161,
                    length: 3,
                    token: Token::Word(Word::Word("た".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 164,
                    length: 3,
                    token: Token::Word(Word::Word("組".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 167,
                    length: 3,
                    token: Token::Word(Word::Word("織".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 170,
                    length: 3,
                    token: Token::Word(Word::Word("の".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 173,
                    length: 3,
                    token: Token::Word(Word::Word("総".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 176,
                    length: 3,
                    token: Token::Word(Word::Word("称".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 179,
                    length: 3,
                    token: Token::Special(Special::Punctuation('。')),
                },
                PositionalToken {
                    source: uws,
                    offset: 182,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 183,
                    length: 3,
                    token: Token::Word(Word::Word("熊".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 186,
                    length: 3,
                    token: Token::Word(Word::Word("野".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 189,
                    length: 3,
                    token: Token::Word(Word::Word("三".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 192,
                    length: 3,
                    token: Token::Word(Word::Word("山".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 195,
                    length: 3,
                    token: Token::Word(Word::Word("を".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 198,
                    length: 3,
                    token: Token::Word(Word::Word("含".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 201,
                    length: 3,
                    token: Token::Word(Word::Word("め".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 204,
                    length: 3,
                    token: Token::Word(Word::Word("て".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 207,
                    length: 3,
                    token: Token::Special(Special::Punctuation('、')),
                },
                PositionalToken {
                    source: uws,
                    offset: 210,
                    length: 3,
                    token: Token::Word(Word::Word("日".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 213,
                    length: 3,
                    token: Token::Word(Word::Word("本".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 216,
                    length: 3,
                    token: Token::Word(Word::Word("に".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 219,
                    length: 3,
                    token: Token::Word(Word::Word("お".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 222,
                    length: 3,
                    token: Token::Word(Word::Word("け".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 225,
                    length: 3,
                    token: Token::Word(Word::Word("る".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 228,
                    length: 3,
                    token: Token::Word(Word::Word("古".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 231,
                    length: 3,
                    token: Token::Word(Word::Word("代".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 234,
                    length: 3,
                    token: Token::Word(Word::Word("か".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 237,
                    length: 3,
                    token: Token::Word(Word::Word("ら".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 240,
                    length: 3,
                    token: Token::Word(Word::Word("中".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 243,
                    length: 3,
                    token: Token::Word(Word::Word("世".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 246,
                    length: 3,
                    token: Token::Word(Word::Word("前".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 249,
                    length: 3,
                    token: Token::Word(Word::Word("半".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 252,
                    length: 3,
                    token: Token::Word(Word::Word("に".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 255,
                    length: 3,
                    token: Token::Word(Word::Word("か".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 258,
                    length: 3,
                    token: Token::Word(Word::Word("け".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 261,
                    length: 3,
                    token: Token::Word(Word::Word("て".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 264,
                    length: 3,
                    token: Token::Word(Word::Word("の".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 267,
                    length: 3,
                    token: Token::Word(Word::Word("寺".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 270,
                    length: 3,
                    token: Token::Word(Word::Word("社".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 273,
                    length: 3,
                    token: Token::Word(Word::Word("の".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 276,
                    length: 3,
                    token: Token::Word(Word::Word("造".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 279,
                    length: 3,
                    token: Token::Word(Word::Word("営".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 282,
                    length: 3,
                    token: Token::Word(Word::Word("は".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 285,
                    length: 3,
                    token: Token::Special(Special::Punctuation('、')),
                },
                PositionalToken {
                    source: uws,
                    offset: 288,
                    length: 3,
                    token: Token::Word(Word::Word("寺".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 291,
                    length: 3,
                    token: Token::Word(Word::Word("社".to_string())),
                },
            ],
            Lang::Kor => vec![
                PositionalToken {
                    source: uws,
                    offset: 0,
                    length: 21,
                    token: Token::Word(Word::Word("플레이스테이션".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 21,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 22,
                    length: 3,
                    token: Token::Word(Word::Word("은".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 25,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 26,
                    length: 6,
                    token: Token::Word(Word::Word("소니".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 32,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 33,
                    length: 9,
                    token: Token::Word(Word::Word("컴퓨터".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 42,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 43,
                    length: 21,
                    token: Token::Word(Word::Word("엔터테인먼트가".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 64,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 65,
                    length: 9,
                    token: Token::Word(Word::Word("개발한".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 74,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 75,
                    length: 3,
                    token: Token::Word(Word::Word("세".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 78,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 79,
                    length: 6,
                    token: Token::Word(Word::Word("번째".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 85,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 86,
                    length: 9,
                    token: Token::Word(Word::Word("가정용".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 95,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 96,
                    length: 15,
                    token: Token::Word(Word::Word("게임기이다".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 111,
                    length: 1,
                    token: Token::Special(Special::Punctuation('.')),
                },
                PositionalToken {
                    source: uws,
                    offset: 112,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 113,
                    length: 24,
                    token: Token::Word(Word::Word("마이크로소프트의".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 137,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 138,
                    length: 12,
                    token: Token::Word(Word::Word("엑스박스".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 150,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 151,
                    length: 3,
                    token: Token::Word(Word::Number(Number::Integer(360))),
                },
                PositionalToken {
                    source: uws,
                    offset: 154,
                    length: 1,
                    token: Token::Special(Special::Punctuation(',')),
                },
                PositionalToken {
                    source: uws,
                    offset: 155,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 156,
                    length: 12,
                    token: Token::Word(Word::Word("닌텐도의".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 168,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 169,
                    length: 6,
                    token: Token::Word(Word::Word("Wii와".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 175,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 176,
                    length: 12,
                    token: Token::Word(Word::Word("경쟁하고".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 188,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 189,
                    length: 6,
                    token: Token::Word(Word::Word("있다".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 195,
                    length: 1,
                    token: Token::Special(Special::Punctuation('.')),
                },
                PositionalToken {
                    source: uws,
                    offset: 196,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 197,
                    length: 6,
                    token: Token::Word(Word::Word("이전".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 203,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 204,
                    length: 12,
                    token: Token::Word(Word::Word("제품에서".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 216,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 217,
                    length: 9,
                    token: Token::Word(Word::Word("온라인".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 226,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 227,
                    length: 9,
                    token: Token::Word(Word::Word("플레이".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 236,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 237,
                    length: 3,
                    token: Token::Word(Word::Word("기".to_string())),
                },
            ],
            Lang::Ara => vec![
                PositionalToken {
                    source: uws,
                    offset: 0,
                    length: 14,
                    token: Token::Word(Word::Word("لشکرکشی".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 14,
                    length: 3,
                    token: Token::Unicode(Unicode::Formatter(Formatter::Char('\u{200c}'))),
                },
                PositionalToken {
                    source: uws,
                    offset: 17,
                    length: 6,
                    token: Token::Word(Word::Word("های".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 23,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 24,
                    length: 6,
                    token: Token::Word(Word::Word("روس".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 30,
                    length: 3,
                    token: Token::Unicode(Unicode::Formatter(Formatter::Char('\u{200c}'))),
                },
                PositionalToken {
                    source: uws,
                    offset: 33,
                    length: 6,
                    token: Token::Word(Word::Word("های".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 39,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 40,
                    length: 12,
                    token: Token::Word(Word::Word("وارنگی".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 52,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 53,
                    length: 4,
                    token: Token::Word(Word::Word("به".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 57,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 58,
                    length: 10,
                    token: Token::Word(Word::Word("دریای".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 68,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 69,
                    length: 6,
                    token: Token::Word(Word::Word("خزر".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 75,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 76,
                    length: 12,
                    token: Token::Word(Word::Word("مجموعه".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 88,
                    length: 3,
                    token: Token::Unicode(Unicode::Formatter(Formatter::Char('\u{200c}'))),
                },
                PositionalToken {
                    source: uws,
                    offset: 91,
                    length: 4,
                    token: Token::Word(Word::Word("ای".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 95,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 96,
                    length: 4,
                    token: Token::Word(Word::Word("از".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 100,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 101,
                    length: 10,
                    token: Token::Word(Word::Word("حملات".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 111,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 112,
                    length: 10,
                    token: Token::Word(Word::Word("نظامی".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 122,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 123,
                    length: 4,
                    token: Token::Word(Word::Word("در".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 127,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 128,
                    length: 6,
                    token: Token::Word(Word::Word("بین".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 134,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 135,
                    length: 6,
                    token: Token::Word(Word::Word("سال".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 141,
                    length: 3,
                    token: Token::Unicode(Unicode::Formatter(Formatter::Char('\u{200c}'))),
                },
                PositionalToken {
                    source: uws,
                    offset: 144,
                    length: 6,
                    token: Token::Word(Word::Word("های".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 150,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 151,
                    length: 6,
                    token: Token::Word(Word::StrangeWord("۸۶۴".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 157,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 158,
                    length: 4,
                    token: Token::Word(Word::Word("تا".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 162,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 163,
                    length: 8,
                    token: Token::Word(Word::StrangeWord("۱۰۴۱".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 171,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 172,
                    length: 12,
                    token: Token::Word(Word::Word("میلادی".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 184,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 185,
                    length: 2,
                    token: Token::Word(Word::Word("ب".to_string())),
                },
            ],
            Lang::Ell => vec![
                PositionalToken {
                    source: uws,
                    offset: 0,
                    length: 4,
                    token: Token::Word(Word::Word("Το".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 4,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 5,
                    length: 18,
                    token: Token::Word(Word::Word("Πρόγραμμα".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 23,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 24,
                    length: 22,
                    token: Token::Word(Word::Word("υλοποιείται".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 46,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 47,
                    length: 4,
                    token: Token::Word(Word::Word("εξ".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 51,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 52,
                    length: 18,
                    token: Token::Word(Word::Word("ολοκλήρου".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 70,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 71,
                    length: 6,
                    token: Token::Word(Word::Word("από".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 77,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 78,
                    length: 16,
                    token: Token::Word(Word::Word("απόσταση".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 94,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 95,
                    length: 6,
                    token: Token::Word(Word::Word("και".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 101,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 102,
                    length: 12,
                    token: Token::Word(Word::Word("μπορεί".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 114,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 115,
                    length: 4,
                    token: Token::Word(Word::Word("να".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 119,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 120,
                    length: 20,
                    token: Token::Word(Word::Word("συμμετέχει".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 140,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 141,
                    length: 8,
                    token: Token::Word(Word::Word("κάθε".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 149,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 150,
                    length: 24,
                    token: Token::Word(Word::Word("εμπλεκόμενος".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 174,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 175,
                    length: 6,
                    token: Token::Word(Word::Word("στη".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 181,
                    length: 1,
                    token: Token::Special(Special::Separator(Separator::Space)),
                },
                PositionalToken {
                    source: uws,
                    offset: 182,
                    length: 2,
                    token: Token::Word(Word::Word("ή".to_string())),
                },
                PositionalToken {
                    source: uws,
                    offset: 184,
                    length: 1,
                    token: Token::Special(Special::Punctuation('/')),
                },
            ],
        };
        (
            uws.chars()
                .take(100)
                .fold(String::new(), |acc, c| acc + &format!("{}", c)),
            tokens,
        )
    }
}
