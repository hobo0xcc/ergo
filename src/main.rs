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

// Read number and convert it to f64.
fn to_f64(compiler: &Compiler, p: &Vec<u8>, c: &mut usize) -> FloatValue {
    let mut sum: f64 = 0.0;
    // Skip whitespace.
    while p.len() > *c && p[*c].is_ascii_whitespace() {
        *c += 1;
    }
    while p.len() > *c && p[*c].is_ascii_digit() {
        sum = (sum * 10.0) + (p[*c] - 0x30) as f64;
        *c += 1;
    }

    compiler.context.f64_type().const_float(sum)
}

fn mul_expr(compiler: &Compiler, p: &Vec<u8>, pc: &mut usize) -> FloatValue {
    let mut left = to_f64(compiler, &p, pc);
    let mut op: char;
    while p.len() > *pc && p[*pc] != 0 {
        if p[*pc].is_ascii_whitespace() {
            *pc += 1;
            continue;
        }
        if p[*pc] == ('*' as u8) {
            op = p[*pc] as char;
            *pc += 1;
        } else if p[*pc] == ('/' as u8) {
            op = p[*pc] as char;
            *pc += 1;
        } else {
            break;
        }

        let right = to_f64(compiler, p, pc);
        match op {
            '*' => left = compiler.builder.build_float_mul(left, right, "sumval"),
            '/' => left = compiler.builder.build_float_div(left, right, "sumval"),
            _ => break,
        }
    }

    left
}

fn add_expr(compiler: &Compiler, p: &Vec<u8>, pc: &mut usize) -> FloatValue {
    let mut left = mul_expr(compiler, p, pc);
    let mut op: char;
    while p.len() > *pc && p[*pc] != 0 {
        if p[*pc].is_ascii_whitespace() {
            continue;
        }
        if p[*pc] == ('+' as u8) {
            op = p[*pc] as char;
            *pc += 1;
        } else if p[*pc] == ('-' as u8) {
            op = p[*pc] as char;
            *pc += 1;
        } else {
            break;
        }

        let right = mul_expr(compiler, p, pc);
        match op {
            '+' => left = compiler.builder.build_float_add(left, right, "sumval"),
            '-' => left = compiler.builder.build_float_sub(left, right, "sumval"),
            _ => break,
        }
    }

    left
}

fn jit_compile(compiler: &Compiler, funcname: &str) -> Option<JitFunction<Sum>> {
    unsafe { compiler.execution_engine.get_function(funcname).ok() }
}

fn main() -> Result<(), Box<dyn Error>> {
    let argv = env::args().collect::<Vec<String>>();
    // If Argument is nothing, exit as failure.
    if argv.len() < 2 {
        eprintln!("Error: Cannot find an input file.");
        process::exit(1);
    }

    // Read file.
    let file = fs::File::open(&argv[1]).ok().unwrap();
    let mut buf = BufReader::new(file);
    let mut data: Vec<u8> = Vec::new();
    buf.read_to_end(&mut data).ok();

    // Counter of data.
    let mut pc: usize = 0;
    // LLVM Settings.
    let context             = Context::create();
    let module              = context.create_module("sum");
    let builder             = context.create_builder();
    let execution_engine    = module.create_jit_execution_engine(OptimizationLevel::None)?;

    let compiler = Compiler::new(&context, &module, &builder, &execution_engine);

    let sum = add_expr(&compiler, &data, &mut pc);
    compiler.builder.build_return(Some(&sum));
    let func = jit_compile(&compiler, "Sum").ok_or("Error Occurred.")?;
    unsafe {
        println!("{}", func.call());
    }
    Ok(())
}
