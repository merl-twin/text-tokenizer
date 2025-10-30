use crate::{SentenceBreaker, UnicodeSentenceBreaker};

use std::collections::BTreeSet;

pub trait IntoTokenizer: Sized {
    type IntoTokens;
    fn into_tokenizer<S: SentenceBreaker>(self, params: TokenizerParams<S>) -> Self::IntoTokens;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenizerOptions {
    NoComplexTokens,
    StructTokens,
    SplitDot,
    SplitUnderscore,
    SplitColon,
    SplitSemiColon,
    SplitNumberSign,
    MergePunctuation,
    MergeWhites,
    WithSentences,
    NumberDefaultEnNotation,
    NumberDefaultRuNotation,
}

pub struct TokenizerParams<S: SentenceBreaker> {
    pub(crate) options: BTreeSet<TokenizerOptions>,
    pub(crate) sentence_breaker: S,
}
impl Default for TokenizerParams<()> {
    fn default() -> TokenizerParams<()> {
        TokenizerParams {
            options: BTreeSet::new(),
            sentence_breaker: (),
        }
    }
}
impl TokenizerParams<()> {
    pub fn v1() -> TokenizerParams<()> {
        TokenizerParams::default()
            .add_option(TokenizerOptions::SplitDot)
            .add_option(TokenizerOptions::SplitUnderscore)
            .add_option(TokenizerOptions::SplitColon)
            .add_option(TokenizerOptions::SplitSemiColon)
            .add_option(TokenizerOptions::MergeWhites)
            .add_option(TokenizerOptions::MergePunctuation)
            .add_option(TokenizerOptions::StructTokens)
    }
    pub fn basic() -> TokenizerParams<()> {
        TokenizerParams::default()
            .add_option(TokenizerOptions::NoComplexTokens)
            .add_option(TokenizerOptions::MergeWhites)
            .add_option(TokenizerOptions::MergePunctuation)
    }
    pub fn complex() -> TokenizerParams<()> {
        TokenizerParams::default()
            .add_option(TokenizerOptions::StructTokens)
            .add_option(TokenizerOptions::MergeWhites)
            .add_option(TokenizerOptions::MergePunctuation)
    }
}
impl<S: SentenceBreaker> TokenizerParams<S> {
    pub fn add_option(mut self, option: TokenizerOptions) -> TokenizerParams<S> {
        self.options.insert(option);
        self
    }
    pub fn remove_option(mut self, option: TokenizerOptions) -> TokenizerParams<S> {
        self.options.remove(&option);
        self
    }
    pub fn with_default_sentences(mut self) -> TokenizerParams<UnicodeSentenceBreaker> {
        self.options.insert(TokenizerOptions::WithSentences);
        TokenizerParams {
            options: self.options,
            sentence_breaker: UnicodeSentenceBreaker,
        }
    }
    pub fn with_sentence_breaker<U: SentenceBreaker>(mut self, sb: U) -> TokenizerParams<U> {
        self.options.insert(TokenizerOptions::WithSentences);
        TokenizerParams {
            options: self.options,
            sentence_breaker: sb,
        }
    }
    pub fn options(&self) -> &BTreeSet<TokenizerOptions> {
        &self.options
    }
}
