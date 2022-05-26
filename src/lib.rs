#![feature(core_intrinsics)]
#![feature(once_cell)]
#![feature(asm)]
#![feature(naked_functions)]
#![feature(const_fn_fn_ptr_basics)]

mod compile;
pub(crate) mod pads;
mod dmir;
mod codegen;
mod ref_count;
mod variable_termination_pass;
mod dmir_annotate;

#[cfg(feature = "test_time")]
mod time;

#[cfg(feature = "bench_utils")]
mod bench_utils;

#[cfg(feature = "tools")]
mod tools;

#[cfg(feature = "test_utils")]
mod test_utils;
pub(crate) mod stack_map;
mod section_memory_manager_bindings;
pub(crate) mod proc_meta;
pub(crate) mod dfa;
pub(crate) mod cfa;
pub(crate) mod ref_count2;


#[macro_use]
extern crate auxtools;
extern crate log;
extern crate core;

use std::collections::HashMap;
use auxtools::{hook, CompileTimeHook, StringRef, raw_types, DMResult, Runtime};
use auxtools::Value;
use auxtools::Proc;
use auxtools::inventory;


use log::LevelFilter;
use std::panic::{UnwindSafe, catch_unwind};
use std::path::Path;
use auxtools::raw_types::procs::ProcId;
use inkwell::execution_engine::ExecutionEngine;


pub struct DisassembleEnv;

impl dmasm::disassembler::DisassembleEnv for DisassembleEnv {
    fn get_string_data(&mut self, index: u32) -> Option<Vec<u8>> {
        unsafe {
            Some(
                StringRef::from_id(raw_types::strings::StringId(index))
                    .data()
                    .to_vec(),
            )
        }
    }

    fn get_variable_name(&mut self, index: u32) -> Option<Vec<u8>> {
        unsafe {
            Some(
                StringRef::from_variable_id(raw_types::strings::VariableId(index))
                    .data()
                    .to_vec(),
            )
        }
    }

    fn get_proc_name(&mut self, index: u32) -> Option<String> {
        Proc::from_id(raw_types::procs::ProcId(index)).map(|x| x.path)
    }

    fn value_to_string_data(&mut self, tag: u32, data: u32) -> Option<Vec<u8>> {
        unsafe {
            let value = Value::new(std::mem::transmute(tag as u8), std::mem::transmute(data));
            match value.to_dmstring() {
                Ok(s) => Some(s.data().to_vec()),
                _ => None,
            }
        }
    }
}


pub enum BoxResult<T, E> {
    /// Contains the success value
    Ok(T),

    /// Contains the error value
    Err(E),
}

pub fn guard<F: FnOnce() -> DMResult + UnwindSafe>(f: F) -> DMResult {
    let res = catch_unwind(move ||
        match f() {
            Ok(value) => BoxResult::Ok(value),
            Err(err) => BoxResult::Err(err)
        }
    );
    match res {
        Ok(BoxResult::Ok(value)) => Ok(value),
        Ok(BoxResult::Err(err)) => {
            log::error!("Hook error: {}", err.message);
            Err(err)
        },
        Err(err) => {
            log::error!("Hook panic: {:?}", err);
            Result::Err(Runtime::new(format!("Panic over boundary {:?}", err)))
        }
    }
}

#[hook("/proc/dmjit_dump_call_count")]
pub fn dump_call_count() -> DMResult {
    log::info!("Dump call count");
    if let Some(mut vec) = call_counts() {
        vec.sort_by_key(|h| -(h.count as i32));
        log::info!("Total {} procs", vec.len());
        for count in vec {
            log::info!("{}\t{}", count.count, count.proc.path);
        }
    }
    Ok(Value::null())
}

macro_rules! log_file {
    ($name:literal) => { concat!(env!("DMJIT_LOG_PREFIX"), $name) };
}

fn rotate_logs(from: &Path, num: u32) {
    let target_name = format!(log_file!("dmjit.log.{}"), num);
    let target = Path::new(target_name.as_str());
    if target.exists() && num < 10 {
        rotate_logs(target, num + 1)
    }
    if let Err(error) = std::fs::copy(from, target) {
        log::error!("Failed to rotate logs ({:?} -> {:?}): {}", from, target, error)
    }
}

#[hook("/proc/dmjit_hook_log_init")]
pub fn log_init() -> DMResult {

    // Do not remove, will break lto
    ExecutionEngine::link_in_mc_jit();

    macro_rules! ver_string {
        () => {
            format!("{}-{} built on {}", env!("VERGEN_GIT_SEMVER"), env!("VERGEN_CARGO_PROFILE"), env!("VERGEN_BUILD_TIMESTAMP"))
        };
    }

    if cfg!(rotate_logs) {
        rotate_logs(Path::new(log_file!("dmjit.log")), 0);
    }
    simple_logging::log_to_file(log_file!("dmjit.log"), LevelFilter::Debug).unwrap();
    log_panics::init();
    log::info!("Log startup, {}", ver_string!());

    for hook in inventory::iter::<CompileTimeHook> {
        log::info!("Hooked {}", hook.proc_path)
    }

    auxtools::hooks::install_interceptor(intercept_proc_call);

    pads::init();

    Value::from_string(format!("dmJIT init success, {}", ver_string!()))
}

pub type ByondProcFunc = unsafe extern "C" fn(out: *mut raw_types::values::Value, src: raw_types::values::Value, usr: raw_types::values::Value, args: *mut raw_types::values::Value, arg_count: u32) -> ();
static mut CHAD_HOOKS: Vec<Option<ByondProcFunc>> = Vec::new();

static mut ENABLE_CHAD_HOOKS: bool = true;

static mut CALL_COUNT: Option<HashMap<ProcId, u32>> = Option::None;

pub struct CallCount {
    pub proc: Proc,
    pub count: u32
}

pub fn call_counts() -> Option<Vec<CallCount>> {
    if let Some(counts) = unsafe { &CALL_COUNT } {
        let counts = counts.iter().filter_map(|(proc_id, count)|
            if let Some(proc) = Proc::from_id(*proc_id) {
                Some(CallCount { proc, count: *count })
            } else {
                None
            }
        ).collect::<Vec<_>>();
        return Some(counts);
    }
    return None;
}


fn intercept_proc_call(
    ret: *mut raw_types::values::Value,
    usr_raw: raw_types::values::Value,
    _proc_type: u32,
    proc_id: raw_types::procs::ProcId,
    _unknown1: u32,
    src_raw: raw_types::values::Value,
    args_ptr: *mut raw_types::values::Value,
    num_args: usize,
    _unknown2: u32,
    _unknown3: u32,
) -> u8 {
    unsafe {
        if ENABLE_CHAD_HOOKS {
            if let Some(Some(hook)) = CHAD_HOOKS.get(proc_id.0 as usize) {
                hook(ret, src_raw, usr_raw, args_ptr, num_args as u32);
                return 1
            }
        }
        if let Some(counts) = &mut CALL_COUNT {
            *counts.entry(proc_id)
                .or_insert(0)
                += 1;
        }
    }
    0
}

pub fn chad_hook_by_id(proc_id: ProcId, hook: ByondProcFunc) {
    unsafe {
        let hooks = &mut CHAD_HOOKS;
        let idx = proc_id.0 as usize;
        if idx >= hooks.len() {
            hooks.resize((idx + 1) as usize, None);
        }
        hooks[idx] = Some(hook);
    }
}


#[hook("/proc/dmjit_toggle_hooks")]
pub fn toggle_hooks() -> DMResult {
    unsafe {
        ENABLE_CHAD_HOOKS = !ENABLE_CHAD_HOOKS;
        return Ok(Value::from(ENABLE_CHAD_HOOKS))
    }

}

#[hook("/proc/dmjit_toggle_call_counts")]
pub fn toggle_call_counts() -> DMResult {
    unsafe {
        match CALL_COUNT {
            None => { CALL_COUNT = Option::Some(HashMap::new()) }
            Some(_) => { CALL_COUNT = Option::None }
        }
        return Ok(Value::from(CALL_COUNT.is_some()))
    }
}