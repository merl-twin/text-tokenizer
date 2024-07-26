use text_parsing::{
    Snip,
};
use unicode_segmentation::{
    UnicodeSegmentation,
    USentenceBoundIndices,
};


pub trait SentenceBreaker {
    type Iter<'t>: Iterator<Item = Snip>;

    fn break_text<'t>(&self, text: &'t str) -> Self::Iter<'t>;
}

impl SentenceBreaker for () {
    type Iter<'t> = std::iter::Empty<Snip>;
    fn break_text<'t>(&self, _text: &'t str) -> Self::Iter<'t> {
        std::iter::empty()
    }
}

pub struct UnicodeSentenceBreaker;

impl SentenceBreaker for UnicodeSentenceBreaker {
    type Iter<'t> = UnicodeIter<'t>;
    
    fn break_text<'t>(&self, text: &'t str) -> Self::Iter<'t> {
        UnicodeIter {
            iter: text.split_sentence_bound_indices(),
        }
    }
}

pub struct UnicodeIter<'t> {
    iter: USentenceBoundIndices<'t>,
}
impl<'t> Iterator for UnicodeIter<'t> {
    type Item = Snip;
    fn next(&mut self) -> Option<Snip> {
        self.iter.next().map(|(offset,s)| Snip{ offset, length: s.len() } )
    }
}
