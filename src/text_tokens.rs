use crate::{
    Bound, IntoTokenizer, SentenceBreaker, Text, TextLocality, TextStr, TextToken, Token2,
    TokenizerOptions, TokenizerParams, Tokens,
};
use std::collections::BTreeSet;

use text_parsing::{Breaker, Local, Localize, Snip};

use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct InnerBound {
    pub bytes: Snip,
    pub chars: Snip,
    pub breaker: Breaker,
    pub original: Option<Local<()>>,
}

impl<'t> IntoTokenizer for &'t Text {
    type IntoTokens = TextTokens<'t>;

    fn into_tokenizer<S: SentenceBreaker>(self, params: TokenizerParams<S>) -> Self::IntoTokens {
        TextTokens::new(
            InnerText {
                buffer: &self.buffer,
                localities: self.localities.clone(),
            },
            self.breakers.clone(),
            params,
        )
    }
}
impl<'t> IntoTokenizer for TextStr<'t> {
    type IntoTokens = TextTokens<'t>;

    fn into_tokenizer<S: SentenceBreaker>(self, params: TokenizerParams<S>) -> Self::IntoTokens {
        TextTokens::new(
            InnerText {
                buffer: self.buffer,
                localities: self.localities.clone(),
            },
            self.breakers.clone(),
            params,
        )
    }
}

struct InnerText<'t> {
    buffer: &'t str,
    localities: Arc<Vec<TextLocality>>,
}

pub struct TextTokens<'t> {
    text: InnerText<'t>,

    bounds: BoundEnum,
    current_offset: usize,
    current_char_offset: usize,
    current_tokens: Option<Tokens<'t>>,

    options: BTreeSet<TokenizerOptions>,
    next_offset: usize,
    next_char_offset: usize,
    next_bound: Option<TextToken>,
}
enum BoundEnum {
    IntoIter(std::vec::IntoIter<InnerBound>),
    Iter {
        next: usize,
        vec: Arc<Vec<InnerBound>>,
    },
}
impl Iterator for BoundEnum {
    type Item = InnerBound;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            BoundEnum::Iter { next, vec } => vec.get(*next).map(|ib| {
                *next += 1;
                *ib
            }),
            /*match vec.get(next) {
                Some(ib) => {
                    *next += 1;
                    Some(std::borrow::Cow::Borrowed(ib))
                }
                None => None,
            },*/
            BoundEnum::IntoIter(iter) => iter.next(),
        }
    }
}
impl<'t> TextTokens<'t> {
    pub fn text(&self) -> &'t str {
        self.text.buffer
    }
}
impl<'t> TextTokens<'t> {
    fn new<'q, S: SentenceBreaker>(
        text: InnerText<'q>,
        breakers: Arc<Vec<InnerBound>>,
        params: TokenizerParams<S>,
    ) -> TextTokens<'q> {
        fn btoc(txt: &str) -> Vec<usize> {
            let mut v = Vec::with_capacity(txt.len());
            v.resize(txt.len(), 0);
            let mut max = 0;
            for (ci, (bi, c)) in txt.char_indices().enumerate() {
                for i in bi..bi + c.len_utf8() {
                    v[i] = ci;
                }
                max = ci;
            }
            v.push(max + 1);
            v
        }

        let mut bounds = None;
        if params.options.contains(&TokenizerOptions::WithSentences) {
            let mut new_b = Vec::new();
            let mut offset = 0;
            let mut char_offset = 0;
            let cnt = breakers.len();
            for ib in breakers.iter() {
                let InnerBound {
                    bytes,
                    chars,
                    breaker,
                    original: _,
                } = ib;
                if bytes.offset < offset {
                    continue;
                }
                match breaker {
                    Breaker::None | Breaker::Space | Breaker::Line | Breaker::Word => {}
                    Breaker::Sentence | Breaker::Paragraph | Breaker::Section => {
                        let txt = &text.buffer[offset..bytes.offset];
                        //println!("{}",txt);
                        let btoc = btoc(txt);
                        for snip in params.sentence_breaker.break_text(txt) {
                            //println!("{:?} -> {}",snip, offset + snip.offset);
                            if text.buffer[offset + snip.offset..offset + snip.offset + snip.length]
                                .trim()
                                .len()
                                > 0
                            {
                                new_b.push(InnerBound {
                                    bytes: Snip {
                                        offset: offset + snip.offset + snip.length,
                                        length: 0,
                                    },
                                    chars: Snip {
                                        offset: char_offset + btoc[snip.offset + snip.length],
                                        length: 0,
                                    },
                                    breaker: Breaker::Sentence,
                                    original: None,
                                });
                            }
                        }
                        new_b.pop(); // remove last sentence breaker
                    }
                }
                new_b.push(ib.clone());
                offset = bytes.offset + bytes.length;
                char_offset = chars.offset + chars.length;
                //println!("");
            }
            let txt = &text.buffer[offset..];
            for snip in params.sentence_breaker.break_text(txt) {
                //println!("{}",txt);
                //println!("{:?} -> {}",snip,offset + snip.offset);
                if text.buffer[offset + snip.offset..offset + snip.offset + snip.length]
                    .trim()
                    .len()
                    > 0
                {
                    let btoc = btoc(txt);
                    new_b.push(InnerBound {
                        bytes: Snip {
                            offset: offset + snip.offset + snip.length,
                            length: 0,
                        },
                        chars: Snip {
                            offset: char_offset + btoc[snip.offset + snip.length],
                            length: 0,
                        },
                        breaker: Breaker::Sentence,
                        original: None,
                    });
                }
            }
            new_b.pop(); // remove last sentence breaker
            if new_b.len() > cnt {
                bounds = Some(BoundEnum::IntoIter(new_b.into_iter()));
            }
        }
        let bounds = match bounds {
            Some(b) => b,
            None => BoundEnum::Iter {
                next: 0,
                vec: breakers,
            },
        };
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
                        let (local, token) = local_token.into_inner();
                        let local = local.with_shift(self.current_char_offset, self.current_offset);
                        let Snip {
                            offset: first,
                            length: len,
                        } = local.chars();
                        if len > 0 {
                            let last = first + len - 1;
                            let original = match len == 1 {
                                false => match Local::from_segment(
                                    self.text.localities[first].original,
                                    self.text.localities[last].original,
                                ) {
                                    Ok(loc) => loc,
                                    Err(_) => continue,
                                },
                                true => self.text.localities[first].original,
                            };
                            break Some(TextToken {
                                locality: local,
                                original: Some(original),
                                token: token.into(),
                            });
                        }
                    }
                    None => {
                        self.current_tokens = None;
                        self.current_offset = self.next_offset;
                        self.current_char_offset = self.next_char_offset;
                        if let Some(tok) = self.next_bound.take() {
                            break Some(tok);
                        }
                    }
                },
                None => {
                    let (txt, next_offset, opt_bound) = match self.bounds.next() {
                        Some(ib) => {
                            let InnerBound {
                                bytes,
                                chars,
                                breaker,
                                original,
                            } = &ib;
                            if bytes.offset < self.current_offset {
                                continue;
                            }
                            let txt = &self.text.buffer[self.current_offset..bytes.offset];
                            let next_offset = bytes.offset + bytes.length;
                            let next_char_offset = chars.offset + chars.length;
                            let opt_bound = match match breaker {
                                Breaker::None | Breaker::Space | Breaker::Line | Breaker::Word => {
                                    None
                                }
                                Breaker::Sentence => Some(Bound::Sentence),
                                Breaker::Paragraph => Some(Bound::Paragraph),
                                Breaker::Section => Some(Bound::Section),
                            } {
                                Some(bound) => Some(TextToken {
                                    locality: ().localize(*chars, *bytes),
                                    original: *original,
                                    token: Token2::Bound(bound),
                                }),
                                None => None,
                            };
                            (txt, (next_offset, next_char_offset), opt_bound)
                        }
                        None => match self.current_offset < self.text.buffer.len() {
                            true => {
                                let txt = &self.text.buffer[self.current_offset..];
                                let next_offset = self.text.buffer.len();
                                let next_char_offset = self.text.localities.len();
                                let opt_bound = None;
                                (txt, (next_offset, next_char_offset), opt_bound)
                            }
                            false => break None,
                        },
                    };
                    self.next_offset = next_offset.0;
                    self.next_char_offset = next_offset.1;
                    self.next_bound = opt_bound;
                    self.current_tokens = Some(Tokens::new(txt, &self.options));
                }
            }
        }
    }
}
