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
use inkwell::values::{FloatValue, FunctionValue, PointerValue};
use std::error::Error;
use std::fmt;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),    // [0-9]+
    Ident(String),  // [a-zA-Z_]+
    Var,            // 'var'
    LParen,         // ')'
    RParen,         // '('
    SemiColon,      // ';'
    Plus,           // '+'
    Minus,          // '-'
    AsterRisk,      // '*'
    Slash,          // '/'
    Assign,         // '='
    Return,         // 'return'
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
            Token::Var => "var",
            Token::Assign => "=",
            Token::SemiColon => ";",
            Token::Return => "return",
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

                match &ident as &str {
                    "return" => return Token::Return,
                    "var" => return Token::Var,
                    _ => return Token::Ident(ident),
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
                Some('=') =>    return Token::Assign,
                Some(';') =>    return Token::SemiColon,
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

pub enum AST {
    BinaryExpr {
        lhs: Box<AST>,
        op: Token,
        rhs: Box<AST>,
    },
    VariableDecl {
        name: String,
    },
    Variable(String),
    StmtList {
        list: Vec<AST>
    },
    Return {
        expr: Box<AST>
    },
    Number(f64),
}

impl fmt::Display for AST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_str = match self {
            AST::BinaryExpr { lhs, op, rhs } => {
                format!("binary({} {} {})", lhs, op, rhs)
            },
            AST::Number(n) => {
                format!("number({})", n)
            },
            AST::VariableDecl { name } => {
                format!("vardecl({})", name)
            },
            AST::Return { expr } => {
                format!("return({})", *expr)
            },
            AST::Variable(name) => {
                format!("var({})", name)
            },
            AST::StmtList { list } => {
                let mut fmt_list = String::new();
                for e in list.iter() {
                    fmt_list += format!("{}\n", e).as_str();
                }
                fmt_list
            }
        };
        write!(f, "{}", fmt_str)
    }
}

pub struct Parser {
    tokens: Vec<(Token, u32)>,
    p: usize,
}

impl Parser {
    pub fn new(src: Vec<u8>) -> Self {
        let lex = Lexer::new(src);
        let tokens: Vec<(Token, u32)> = lex.collect();

        Parser {
            tokens,
            p: 0,
        }
    }

    pub fn eat(&mut self) -> (Token, u32) {
        self.p += 1;
        if let Some(t) = self.tokens.get(self.p - 1) {
            t.clone()
        } else {
            (Token::EOF, 0)
        }
    }

    pub fn cur(&self) -> (Token, u32) {
        if let Some(t) = self.tokens.get(self.p) {
            t.clone()
        } else {
            (Token::EOF, 0)
        }
    }

    pub fn peek(&self, i: usize) -> (Token, u32) {
        if let Some(t) = self.tokens.get(self.p + i) {
            t.clone()
        } else {
            (Token::EOF, 0)
        }
    }

    pub fn expect(&mut self, tok: Token) -> Token {
        if self.cur().0 != tok {
            eprintln!("Error (at line {}): Expected {}, but got {}.", self.cur().1, tok, self.cur().0);
            process::exit(1);
        }

        self.eat().0
    }

    pub fn primary_expr(&mut self) -> AST {
        match self.eat() {
            (Token::LParen, _) => {
                let expr = self.add_expr();
                if self.cur().0 != Token::RParen {
                    eprintln!("Error (at line {}): Parentheses are not closed.", self.cur().1);
                    process::exit(1);
                }
                self.p += 1;
                expr
            },
            (Token::Number(n), _) => AST::Number(n),
            (Token::Ident(s), _) => AST::Variable(s),
            (tok, line) => {
                eprintln!("Error (at line {}): Unexpected Token: {}", line, tok);
                process::exit(1);
            }
        }
    }

    pub fn mul_expr(&mut self) -> AST {
        let mut left = self.primary_expr();
        loop {
            let op = self.cur();
            match op {
                (Token::AsterRisk, _) | (Token::Slash, _) => self.p += 1,
                _ => break,
            }

            let right = self.primary_expr();
            match op.0 {
                Token::AsterRisk | Token::Slash => {
                    left = AST::BinaryExpr {
                        lhs: Box::new(left),
                        op: op.0.clone(),
                        rhs: Box::new(right) }
                },
                _ => break,
            }
        }

        left
    }

    pub fn add_expr(&mut self) -> AST {
        let mut left = self.mul_expr();
        loop {
            let op = self.cur();
            match op {
                (Token::Plus, _) | (Token::Minus, _) => self.p += 1,
                _ => break,
            }

            let right = self.mul_expr();
            match op.0 {
                Token::Plus | Token::Minus => {
                    left =  AST::BinaryExpr {
                        lhs: Box::new(left),
                        op: op.0.clone(),
                        rhs: Box::new(right) }
                },
                _ => break,
            }
        }

        left
    }

    pub fn assign_expr(&mut self) -> AST {
        let mut left = self.add_expr();
        loop {
            let op = self.cur();
            match op {
                (Token::Assign, _) => self.p += 1,
                _ => break,
            }

            let right = self.add_expr();
            match op.0 {
                Token::Assign => {
                    left =  AST::BinaryExpr {
                        lhs: Box::new(left),
                        op: op.0.clone(),
                        rhs: Box::new(right) }
                },
                _ => break,
            }
        }

        left
    }

    pub fn var_stmt(&mut self) -> AST {
        self.expect(Token::Var);
        let name = match self.cur() {
            (Token::Ident(s), _) => s,
            _ => {
                eprintln!("Error (at line {}): Unexpected {}.", self.cur().1, self.cur().0);
                process::exit(1);
            }
        };
        self.eat();
        let var_decl = AST::VariableDecl {
            name: name.clone(),
        };
        if self.cur().0 == Token::Assign {
            let mut list: Vec<AST> = Vec::new();
            let var = AST::Variable(name);
            let op = self.eat().0;
            let assign_expr = AST::BinaryExpr {
                lhs: Box::new(var),
                op,
                rhs: Box::new(self.assign_expr()),
            };
            list.push(var_decl);
            list.push(assign_expr);
            AST::StmtList {
                list,
            }
        } else {
            var_decl
        }
    }

    pub fn return_stmt(&mut self) -> AST {
        self.expect(Token::Return);
        let expr = self.assign_expr();
        AST::Return {
            expr: Box::new(expr),
        }
    }

    pub fn statement(&mut self) -> AST {
        let stmt: AST;
        match self.cur() {
            (Token::Var, _) => {
                stmt = self.var_stmt();
                self.expect(Token::SemiColon);
            },
            (Token::Return, _) => {
                stmt = self.return_stmt();
                self.expect(Token::SemiColon);
            },
            _ => {
                stmt = self.assign_expr();
                self.expect(Token::SemiColon);
            },
        }

        stmt
    }

    pub fn parse(&mut self) -> AST {
        let mut ast_list: Vec<AST> = Vec::new();
        while self.cur().0 != Token::EOF {
            ast_list.push(self.statement());
        }
        AST::StmtList {
            list: ast_list
        }
    }
}

struct Compiler<'a> {
    context: &'a Context,
    module: &'a Module,
    builder: &'a Builder,
    execution_engine: &'a ExecutionEngine,
    variables: HashMap<String, PointerValue>,
    func: Option<FunctionValue>,
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
            variables: HashMap::new(),
            func: None,
        };
        compiler.func = Some(compiler.create_func("main"));
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

    pub fn create_alloca(&mut self, name: String) -> PointerValue {
        let builder = self.context.create_builder();
        let entry = self.func.unwrap().get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(inst) => builder.position_before(&inst),
            None => builder.position_at_end(&entry),
        }

        builder.build_alloca(self.context.f64_type(), &name)
    }

    pub fn compile_expr(&mut self, expr: AST) -> FloatValue {
        match expr {
            AST::Number(n) => {
                self.context.f64_type().const_float(n)
            },
            AST::BinaryExpr { lhs, op, rhs } => {
                if op == Token::Assign {
                    let name = match *lhs {
                        AST::Variable(name) => name.clone(),
                        _ => {
                            eprintln!("Error: Not a variable.");
                            process::exit(1);
                        }
                    };
                    let right = self.compile_expr(*rhs);
                    let var = self.variables.get(&name).ok_or("Error: Undefined variable name.").unwrap();
                    self.builder.build_store(*var, right);
                    right
                } else {
                    let left = self.compile_expr(*lhs);
                    let right = self.compile_expr(*rhs);
                    return match op {
                        Token::Plus => self.builder.build_float_add(left, right, "tmp"),
                        Token::Minus => self.builder.build_float_sub(left, right, "tmp"),
                        Token::AsterRisk => self.builder.build_float_mul(left, right, "tmp"),
                        Token::Slash => self.builder.build_float_div(left, right, "tmp"),
                        _ => {
                            eprintln!("Error: Unknown operator.");
                            process::exit(1);
                        }
                    }
                }
            },
            AST::Variable(name) => {
                let var = self.variables.get(&name).unwrap();
                self.builder.build_load(*var, &name).into_float_value()
            },
            _ => {
                eprintln!("Error: Not an expression.");
                process::exit(1);
            }
        }
    }

    pub fn compile_stmt(&mut self, stmt: AST) {
        match stmt {
            AST::StmtList { list } => {
                for s in list {
                    self.compile_stmt(s);
                }
            },
            AST::VariableDecl { name } => {
                let name_cloned = name.clone();
                let alloca = self.create_alloca(name_cloned);
                self.variables.insert(name, alloca);
            },
            AST::Return { expr } => {
                let result = self.compile_expr(*expr);
                self.builder.build_return(Some(&result));
            }
            _ => { self.compile_expr(stmt); },
        }
    }

    pub fn compile(&mut self, ast: AST) {
        self.compile_stmt(ast);
    }
}

fn jit_compile(compiler: &Compiler, funcname: &str) -> Option<JitFunction<Sum>> {
    unsafe { compiler.execution_engine.get_function(funcname).ok() }
}


#[test]
fn debug_lexer() {
    let src = String::from("return 10 + 20 + 30 * 40 / 5;").into_bytes();
    let lex: Lexer = Lexer::new(src);
    let tokens: Vec<(Token, u32)> = lex.collect();
    for (i, (tok, line)) in tokens.iter().enumerate() {
        println!("[{}]\t(line {}): '{}'", i, line, tok);
    }
}

#[test]
fn debug_parser() {
    let src = String::from("var i = 0; return 10 + 20 + 30 * 40 / 5;").into_bytes();
    let mut parser = Parser::new(src);
    let ast = parser.parse();
    println!("{}", ast);
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

    let mut parser = Parser::new(data);
    let ast = parser.parse();

    // LLVM Settings.
    let context             = Context::create();
    let module              = context.create_module("sum");
    let builder             = context.create_builder();
    let execution_engine    = module.create_jit_execution_engine(OptimizationLevel::None)?;

    let mut compiler = Compiler::new(&context, &module, &builder, &execution_engine);

    compiler.compile(ast);
    let func = jit_compile(&compiler, "main").ok_or("Error Occurred.")?;
    unsafe {
        println!("{}", func.call());
    }
    Ok(())
}
