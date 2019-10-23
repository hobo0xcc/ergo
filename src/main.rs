extern crate inkwell;

use std::env;
use std::process;
use std::fs;
use std::io::BufReader;
use std::io::Read;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{FloatValue, FunctionValue};
use std::error::Error;
use std::fmt;

struct Compiler<'a> {
    context: &'a Context,
    module: &'a Module,
    builder: &'a Builder,
    execution_engine: &'a ExecutionEngine,
}

type Sum = unsafe extern "C" fn () -> f64;

impl<'a> Compiler<'a> {
    pub fn new(context: &'a Context,
               module: &'a Module,
               builder: &'a Builder,
               execution_engine: &'a ExecutionEngine) -> Self {
        let mut compiler = Compiler {
            context,
            module,
            builder,
            execution_engine,
        };
        compiler.create_func("Sum");
        compiler.builder.build_alloca(compiler.context.f64_type(), "sumval");
        compiler
    }

    pub fn create_func(&mut self, name: &'a str) -> FunctionValue {
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[], false);
        let func = self.module.add_function(name, fn_type, None);
        let basic_block = self.context.append_basic_block(&func, "entry");
        self.builder.position_at_end(&basic_block);

        func
    }
}

pub enum Token {
    Number(f64),    // [0-9]+
    Ident(String),  // [a-zA-Z_]+
    LParen,         // ')'
    RParen,         // '('
    Plus,           // '+'
    Minus,          // '-'
    AsterRisk,      // '*'
    Slash,          // '/'
    ErrorToken(Option<char>),
    EOF,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tmp: String;
        let tok_str: &str = match self {
            Token::Number(n) => {
                tmp = format!("{}", n);
                &tmp
            },
            Token::Ident(s) => {
                tmp = format!("{}", s);
                &tmp
            },
            Token::LParen => "(",
            Token::RParen => ")",
            Token::Plus => "+",
            Token::Minus => "-",
            Token::AsterRisk => "*",
            Token::Slash => "/",
            Token::ErrorToken(_) => "Illegal Token",
            Token::EOF => "EOF",
        };
        write!(f, "{}", tok_str)
    }
}

pub struct Lexer {
    line: u32,
    src: Vec<u8>,
    err: u32,
    p: usize,
}

impl Lexer {
    pub fn new(src: Vec<u8>) -> Self {
        Lexer {
            line: 1,
            src,
            err: 0,
            p: 0
        }
    }

    pub fn peek(&self, i: usize) -> Option<char> {
        let optch = self.src.get(self.p + i);
        match optch {
            Some(ch) => Some(*ch as char),
            None => None
        }
    }

    pub fn cur(&self) -> Option<char> {
        let optch = self.src.get(self.p);
        match optch {
            Some(ch) => Some(*ch as char),
            None => None
        }
    }

    pub fn eat(&mut self) -> Option<char> {
        self.p += 1;
        let optch = self.src.get(self.p - 1);
        match optch {
            Some(ch) => Some(*ch as char),
            None => None
        }
    }

    pub fn current_p(&self) -> usize {
        self.p
    }

    pub fn is<F>(&self, optch: Option<char>, f: F) -> bool
        where F: Fn(&char) -> bool {
        match optch {
            Some(ch) => f(&ch),
            None => false
        }
    }

    pub fn is_eof(&self, optch: Option<char>) -> bool {
        match optch {
            Some(_) => false,
            None => true
        }
    }

    pub fn is_ident_ch(&self, optch: Option<char>) -> bool {
        self.is(optch, |c| (*c).is_ascii_alphabetic() || *c == '_')
    }

    pub fn is_space(&self, optch: Option<char>) -> bool {
        self.is(optch, |c| *c == ' ' || *c == '\n' || *c == '\t')
    }

    pub fn is_number_ch(&self, optch: Option<char>) -> bool {
        self.is(optch, |c| *c >= '0' && *c <= '9')
    }

    pub fn get_token(&mut self) -> Token {
        loop {
            if self.is_eof(self.cur()) {
                return Token::EOF;
            }

            // Skip space.
            if self.is_space(self.cur()) {
                if self.is(self.cur(), |c| *c == '\n') {
                    self.line += 1;
                }

                self.p += 1;
                continue;
            }

            // Parse identifier.
            if self.is_ident_ch(self.cur()) {
                let start = self.current_p();
                self.p += 1;
                loop {
                    if !self.is_ident_ch(self.cur()) {
                        break;
                    }

                    self.p += 1;
                }

                let ident: String =
                    self.src[start..self.p]
                    .iter()
                    .map(|c| *c as char)
                    .collect();

                match &ident {
                    _ => return Token::Ident(ident)
                }
            }

            // Parse number.
            if self.is_number_ch(self.cur()) {
                let start = self.current_p();
                loop {
                    if !self.is_number_ch(self.cur()) {
                        break;
                    }

                    self.p += 1;
                }

                let number: f64 = self.src[start..self.p]
                    .iter()
                    .map(|c| *c as char)
                    .collect::<String>()
                    .parse::<f64>()
                    .ok()
                    .unwrap();

                return Token::Number(number);
            }

            match self.eat() {
                Some('(') =>    return Token::LParen,
                Some(')') =>    return Token::RParen,
                Some('+') =>    return Token::Plus,
                Some('-') =>    return Token::Minus,
                Some('*') =>    return Token::AsterRisk,
                Some('/') =>    return Token::Slash,
                Some(ch)  =>    return Token::ErrorToken(Some(ch)),
                None      =>    return Token::ErrorToken(None),
            }
        }
    }
}

impl Iterator for Lexer {
    type Item = (Token, u32);

    fn next(&mut self) -> Option<Self::Item> {
        let tok = self.get_token();
        let line = self.line;

        match &tok {
            Token::EOF => None,
            Token::ErrorToken(optch) => {
                if let Some(ch) = optch {
                    eprintln!("Error (at line {}): Unknown character: {}", line, ch);
                } else {
                    eprintln!("Error (at line {}): Unexpected EOF.", line);
                }

                self.err += 1;
                Some((tok, line))
            },
            _ => Some((tok, line)),
        }
    }
}

fn primary_expr(compiler: &Compiler, p: &Vec<(Token, u32)>, pc: &mut usize) -> FloatValue {
    *pc += 1;
    match p[*pc - 1].0 {
        Token::Number(n) => compiler.context.f64_type().const_float(n),
        Token::LParen => {
            let exp = add_expr(compiler, p, pc);
            *pc += 1;
            match p[*pc - 1].0 {
                Token::RParen => return exp,
                _ => {
                    eprintln!("Error (at line {}): Parentheses are not closed.", p[*pc].1);
                    process::exit(1);
                }
            }
        },
        _ => {
            eprintln!("Error (at line {}): Unknown Token: {}", p[*pc].1, p[*pc].0);
            process::exit(1);
        }
    }
}

fn mul_expr(compiler: &Compiler, p: &Vec<(Token, u32)>, pc: &mut usize) -> FloatValue {
    let mut left = primary_expr(compiler, p, pc);
    let mut op: Token;
    while p.len() > *pc {
        match p[*pc].0 {
            Token::AsterRisk    => op = Token::AsterRisk,
            Token::Slash        => op = Token::Slash,
            _ => break,
        }
        *pc += 1;

        let right = primary_expr(compiler, p, pc);
        match op {
            Token::AsterRisk    => left = compiler.builder.build_float_mul(left, right, "sumval"),
            Token::Slash        => left = compiler.builder.build_float_div(left, right, "sumval"),
            _ => break,
        }
    }

    left
}

fn add_expr(compiler: &Compiler, p: &Vec<(Token, u32)>, pc: &mut usize) -> FloatValue {
    let mut left = mul_expr(compiler, p, pc);
    let mut op: Token;
    while p.len() > *pc {
        match p[*pc].0 {
            Token::Plus     => op = Token::Plus,
            Token::Minus    => op = Token::Minus,
            _ => break,
        }
        *pc += 1;

        let right = mul_expr(compiler, p, pc);
        match op {
            Token::Plus     => left = compiler.builder.build_float_add(left, right, "sumval"),
            Token::Minus    => left = compiler.builder.build_float_sub(left, right, "sumval"),
            _ => break,
        }
    }

    left
}

fn jit_compile(compiler: &Compiler, funcname: &str) -> Option<JitFunction<Sum>> {
    unsafe { compiler.execution_engine.get_function(funcname).ok() }
}


#[test]
fn debug_lexer() {
    let src = String::from("10 + 20 + 30 * 40 / 5").into_bytes();
    let lex: Lexer = Lexer::new(src);
    let tokens: Vec<(Token, u32)> = lex.collect();
    for (i, (tok, line)) in tokens.iter().enumerate() {
        println!("[{}]\t(line {}): '{}'", i, line, tok);
    }
}

fn read_file(filename: &str) -> Vec<u8> {
    let file = fs::File::open(filename).ok().unwrap();
    let mut buf = BufReader::new(file);
    let mut data: Vec<u8> = Vec::new();
    buf.read_to_end(&mut data).ok();
    data
}

fn main() -> Result<(), Box<dyn Error>> {
    let argv = env::args().collect::<Vec<String>>();
    // If Argument is nothing, exit as failure.
    if argv.len() < 2 {
        eprintln!("Error: Cannot find an input file.");
        process::exit(1);
    }

    // Read file.
    let data = read_file(&argv[1]);

    let lex = Lexer::new(data);
    let tokens = lex.collect();

    // Counter of tokens.
    let mut pc: usize = 0;
    // LLVM Settings.
    let context             = Context::create();
    let module              = context.create_module("sum");
    let builder             = context.create_builder();
    let execution_engine    = module.create_jit_execution_engine(OptimizationLevel::None)?;

    let compiler = Compiler::new(&context, &module, &builder, &execution_engine);

    let sum = add_expr(&compiler, &tokens, &mut pc);
    compiler.builder.build_return(Some(&sum));
    let func = jit_compile(&compiler, "Sum").ok_or("Error Occurred.")?;
    unsafe {
        println!("{}", func.call());
    }
    Ok(())
}
