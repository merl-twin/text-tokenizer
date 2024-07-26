use std::collections::BTreeSet;
use crate::{
    Tokens,
    TextToken, Token2, Bound,
    TokenizerOptions,
    TokenizerParams,
    SentenceBreaker,
    Text,
    IntoTokenizer,
};

use text_parsing::{
    Breaker, Local, Snip,
    Localize,
};

#[derive(Debug,Clone)]
pub(crate) struct InnerBound {
    pub bytes: Snip,
    pub chars: Snip,
    pub breaker: Breaker,
    pub original: Option<Local<()>>,
}

impl<'t> IntoTokenizer for &'t Text {
    type IntoTokens = TextTokens<'t>;

    fn into_tokenizer<S: SentenceBreaker>(self, params: TokenizerParams<S>) -> Self::IntoTokens {
        TextTokens::new(self, params)
    }
}

pub struct TextTokens<'t> {
    text: &'t Text,
    bounds: BoundEnum<'t>,
    current_offset: usize,
    current_char_offset: usize,
    current_tokens: Option<Tokens<'t>>,

    options: BTreeSet<TokenizerOptions>,
    next_offset: usize,
    next_char_offset: usize,
    next_bound: Option<TextToken>,
}
enum BoundEnum<'t> {
    Iter(std::slice::Iter<'t,InnerBound>),
    IntoIter(std::vec::IntoIter<InnerBound>),
}
impl<'t> Iterator for BoundEnum<'t> {
    type Item = std::borrow::Cow<'t,InnerBound>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            BoundEnum::Iter(iter) => iter.next().map(std::borrow::Cow::Borrowed),
            BoundEnum::IntoIter(iter) => iter.next().map(std::borrow::Cow::Owned),
        }
    }
}
impl<'t> TextTokens<'t> {
    fn new<S: SentenceBreaker>(text: &Text, params: TokenizerParams<S>) -> TextTokens {
        fn btoc(txt: &str) -> Vec<usize> {
            let mut v = Vec::with_capacity(txt.len());
            v.resize(txt.len(),0);
            let mut max = 0;
            for (ci,(bi,c)) in txt.char_indices().enumerate() {
                for i in bi .. bi + c.len_utf8() {
                    v[i] = ci;
                }
                max = ci;
            }
            v.push(max+1);
            v
        }
        
        let mut bounds = BoundEnum::Iter(text.breakers.iter());
        if params.options.contains(&TokenizerOptions::WithSentences) {
            let mut new_b = Vec::new();
            let mut cnt = 0;
            let mut offset = 0;
            let mut char_offset = 0;
            for ib in text.breakers.iter() {
                let InnerBound{ bytes, chars, breaker, original: _ } = ib;
                if bytes.offset < offset { continue; }
                match breaker {
                    Breaker::None | Breaker::Space | Breaker::Line | Breaker::Word => {},
                    Breaker::Sentence | Breaker::Paragraph | Breaker::Section => {
                        let txt = &text.buffer[offset .. bytes.offset];
                        //println!("{}",txt);
                        let btoc = btoc(txt);                            
                        for snip in params.sentence_breaker.break_text(txt) {
                            //println!("{:?} -> {}",snip, offset + snip.offset);
                            if text.buffer[offset + snip.offset .. offset + snip.offset + snip.length].trim().len() > 0 {
                                cnt += 1;
                                new_b.push(InnerBound {
                                    bytes: Snip{ offset: offset + snip.offset + snip.length, length: 0 },
                                    chars: Snip{ offset: char_offset + btoc[snip.offset + snip.length], length: 0 },
                                    breaker: Breaker::Sentence,
                                    original: None,
                                });
                            }
                        }
                    },
                }
                new_b.push(ib.clone());
                offset = bytes.offset + bytes.length;
                char_offset = chars.offset + chars.length;
                //println!("");
            }
            let txt = &text.buffer[offset ..];
            for snip in params.sentence_breaker.break_text(txt) {
                //println!("{}",txt);
                //println!("{:?} -> {}",snip,offset + snip.offset);
                if text.buffer[offset + snip.offset .. offset + snip.offset + snip.length].trim().len() > 0 {
                    cnt += 1;
                    let btoc = btoc(txt);  
                    new_b.push(InnerBound {
                        bytes: Snip{ offset: offset + snip.offset + snip.length, length: 0 },
                        chars: Snip{ offset: char_offset + btoc[snip.offset + snip.length], length: 0 },
                        breaker: Breaker::Sentence,
                        original: None,
                    });
                }
            }
            if cnt > 0 {
                bounds = BoundEnum::IntoIter(new_b.into_iter());
            }
        }
        TextTokens {
            text,
            bounds,
            current_offset: 0,
            current_char_offset: 0,
            current_tokens: None,
            options: params.options,

            next_offset: 0,
            next_char_offset: 0,
            next_bound: None,
        }
    }
}
impl<'t> Iterator for TextTokens<'t> {
    type Item = TextToken;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.current_tokens {
                Some(tokens) => match tokens.next() {
                    Some(local_token) => {
                        let (local,token) = local_token.into_inner();
                        let local = local.with_shift(self.current_char_offset, self.current_offset);
                        let Snip { offset: first, length: len } = local.chars();
                        if len > 0 {
                            let last = first + len - 1;
                            let original = match len == 1 {
                                false => match Local::from_segment(self.text.originals[first],self.text.originals[last]) {
                                    Ok(loc) => loc,
                                    Err(_) => continue,
                                },
                                true => self.text.originals[first],                                
                            };                            
                            break Some(TextToken {
                                locality: local,
                                original: Some(original),
                                token: token.into(),
                            });
                        }
                    },
                    None => {
                        self.current_tokens = None;
                        self.current_offset = self.next_offset;
                        self.current_char_offset = self.next_char_offset;
                        if let Some(tok) = self.next_bound.take() {
                            break Some(tok);
                        }
                    },
                },
                None => {                    
                    let (txt,next_offset,opt_bound) = match self.bounds.next() {
                        Some(ib) => {
                            let InnerBound{ bytes, chars, breaker, original } = ib.as_ref();
                            if bytes.offset < self.current_offset { continue; }
                            let txt = &self.text.buffer[self.current_offset .. bytes.offset];
                            let next_offset = bytes.offset + bytes.length;
                            let next_char_offset = chars.offset + chars.length;
                            let opt_bound = match match breaker {
                                Breaker::None | Breaker::Space | Breaker::Line | Breaker::Word => None,
                                Breaker::Sentence => Some(Bound::Sentence),
                                Breaker::Paragraph => Some(Bound::Paragraph),
                                Breaker::Section => Some(Bound::Section),
                            } {
                                Some(bound) => Some(TextToken {
                                    locality: ().localize(*chars,*bytes),
                                    original: *original,
                                    token: Token2::Bound(bound),
                                }),
                                None => None,
                            };
                            (txt,(next_offset,next_char_offset),opt_bound)
                        },
                        None => match self.current_offset < self.text.buffer.len() {
                            true => {
                                let txt = &self.text.buffer[self.current_offset .. ];
                                let next_offset = self.text.buffer.len();
                                let next_char_offset = self.text.originals.len();
                                let opt_bound = None;
                                (txt,(next_offset,next_char_offset),opt_bound)
                            },
                            false => break None,
                        },
                    };
                    self.next_offset = next_offset.0;
                    self.next_char_offset = next_offset.1;
                    self.next_bound = opt_bound;
                    self.current_tokens = Some(Tokens::new(txt,&self.options));
                },
            }
        }
    }
}

