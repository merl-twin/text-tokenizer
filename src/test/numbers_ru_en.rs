use crate::{
    Number, Text, TextToken, Token2, Word,
    options::{IntoTokenizer, TokenizerOptions, TokenizerParams},
};

use std::collections::BTreeMap;

/*2 октября: 20.000 рублей, с 3 октября - 22.000 рублей, в день семинара - 25.000 рублей.


Μаткапитал выратет c 1 фeвpaля 2026 гοдa
▪️️ 974.000₽ нa втοpοгο peбёнκa, ecли ceмья нe пοлучaлa выплaт нa пepвeнцa
▪️737.000₽ — нa пepвοгο
▪️️ Εдинοвpeмeннοe пοcοбиe пpи pοждeнии peбёнκa увeличитcя дο 28.773₽.
️📍Μaκcимaльнaя cуммa пοcοбия пο бepeмeннοcти и pοдaм выpacтeт дο 955.000₽
️📍Ποcοбиe пο уxοду зa peбёнκοм дο 1,5 лeт для paбοтaющиx гpaждaн — дο 83.000₽ в мecяц.

1.950.000₽ торг
Μарка:Vоlkswаgеn
Μодель: Jettа
Γод выпуска: 01/2020
Двигатель: 1.4 Бeнзин
Πробeг: 104т км

стоимость вездеходов базовой комплектации начинается от 2.080.000 рублей и выше.

Εдинοвpeмeннοe пοcοбиe пpи pοждeнии peбёнκa увeличитcя дο 28.773₽.
 */

struct WordChecker<'s> {
    words: BTreeMap<&'s str, (Token2, usize)>,
}
impl<'s> WordChecker<'s> {
    // be careful with the same words used several time
    fn new<'q>(words: &[(&'q str, Token2)]) -> WordChecker<'q> {
        WordChecker {
            words: words.iter().cloned().map(|(s, t)| (s, (t, 0))).collect(),
        }
    }

    fn check(&mut self, text: &Text, token: &TextToken) {
        let s = text.token_text(&token);
        if let Some((rtok, cnt)) = self.words.get_mut(s) {
            *cnt += 1;
            assert_eq!(rtok.clone(), token.token);
        }
    }
    fn check_count(&self) {
        for (s, (t, cnt)) in &self.words {
            if *cnt == 0 {
                panic!("Missed word '{}': {:?}", s, t);
            }
        }
    }
}

#[test]
fn test_ru_01() {
    let txt =
        "2 октября: 20.000 рублей, с 3 октября - 22.000 рублей, в день семинара - 25.000 рублей.";

    let mut checker = WordChecker::new(&[
        (
            "20.000",
            Token2::Word(Word::Number(Number::Integer(20_000))),
        ),
        (
            "22.000",
            Token2::Word(Word::Number(Number::Integer(22_000))),
        ),
        (
            "25.000",
            Token2::Word(Word::Number(Number::Integer(25_000))),
        ),
    ]);

    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
fn test_ru_02() {
    let txt = "1.950.000₽ торг
Μарка:Vоlkswаgеn
Μодель: Jettа
Γод выпуска: 01/2020
Двигатель: 1.4 Бeнзин
Πробeг: 104т км";

    let mut checker = WordChecker::new(&[
        (
            "1.950.000",
            Token2::Word(Word::Number(Number::Integer(1_950_000))),
        ),
        ("1.4", Token2::Word(Word::Number(Number::Float(1.4)))),
    ]);

    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
#[rustfmt::skip]
fn custom_numbers() {
    let txt = "115,7 123,398,398 2,123.45 0,05%";

    let mut checker = WordChecker::new(&[
        ("115,7", Token2::Word(Word::Number(Number::Float(115.7)))),
        ("123,398,398", Token2::Word(Word::Number(Number::Integer(123398398)))),
        ("2,123.45", Token2::Word(Word::Number(Number::Float(2123.45)))),
        ("0,05", Token2::Word(Word::Number(Number::Float(0.05)))),
    ]);

    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
#[rustfmt::skip]
fn custom_numbers_ftoi() {
    let txt = "1.1 10.0000";

    let mut checker = WordChecker::new(&[
        ("1.1", Token2::Word(Word::Number(Number::Float(1.1)))),
        ("10.0000", Token2::Word(Word::Number(Number::Integer(10)))),
    ]);

    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
#[rustfmt::skip]
fn custom_numbers_en_1() {
    let txt = "1.1 10,000";
    
    let mut checker = WordChecker::new(&[
        ("1.1", Token2::Word(Word::Number(Number::Float(1.1)))),
        ("10,000", Token2::Word(Word::Number(Number::Integer(10000)))),
    ]);
    
    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
#[rustfmt::skip]
fn custom_numbers_en_2() {
    let txt = "1,000.1 10,000";

    let mut checker = WordChecker::new(&[
        ("1,000.1", Token2::Word(Word::Number(Number::Float(1000.1)))),
        ("10,000", Token2::Word(Word::Number(Number::Integer(10000)))),
    ]);
    
    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
#[rustfmt::skip]
fn custom_numbers_ru_1() {
    let txt = "1.1 10,001";

    let mut checker = WordChecker::new(&[
        ("1.1", Token2::Word(Word::Number(Number::Float(1.1)))),
        ("10,001", Token2::Word(Word::Number(Number::Integer(10001)))),
    ]);
    
    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1().add_option(TokenizerOptions::NumberUnknownComaAsDot)) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
#[rustfmt::skip]
fn custom_numbers_ru_2() {
    let txt = "1,1 10,001";

    let mut checker = WordChecker::new(&[
        ("1,1", Token2::Word(Word::Number(Number::Float(1.1)))),
        ("10,001", Token2::Word(Word::Number(Number::Integer(10001)))),
    ]);
    
    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}

#[test]
#[rustfmt::skip]
fn custom_numbers_ru_3() {
    let txt = "10000,1 10,001";

    let mut checker = WordChecker::new(&[
        ("10000,1", Token2::Word(Word::Number(Number::Float(10000.1)))),
        ("10,001", Token2::Word(Word::Number(Number::Integer(10001)))),
    ]);
    
    let text = Text::try_from(txt).unwrap();
    for tok in text.into_tokenizer(TokenizerParams::v1()) {
        println!("[ {} ] {:?}", text.token_text(&tok), tok);
        checker.check(&text, &tok);
    }
    checker.check_count();
}
