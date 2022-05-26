use std::alloc::{alloc_zeroed, Layout};
use std::cmp::max;
use std::ffi::CStr;
use std::lazy::Lazy;
use std::mem::transmute_copy;

use auxtools::{Proc, Value};
use auxtools::DMResult;
use auxtools::raw_types::procs::ProcId;
use auxtools::raw_types::values::ValueTag;
use dmasm::format_disassembly;
use inkwell::attributes::AttributeLoc;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::OptimizationLevel;
use inkwell::passes::PassManager;
use inkwell::values::{AnyValue, MetadataValue};
use llvm_sys::execution_engine::LLVMExecutionEngineGetErrMsg;

use crate::{ByondProcFunc, chad_hook_by_id, DisassembleEnv, dmir, guard, pads};
use crate::codegen::CodeGen;
use crate::dfa::analyze_and_dump_dfa;
use crate::dmir::DMIR;
use crate::proc_meta::{ProcMeta, ProcMetaModuleBuilder};
use crate::ref_count2::generate_ref_count_operations2;
use crate::ref_count::generate_ref_count_operations;
use crate::section_memory_manager_bindings::{Section, SectionMemoryManager};
use crate::stack_map::{read_stack_map, StackMap};
use crate::variable_termination_pass::variable_termination_pass;

#[hook("/proc/dmjit_compile_proc")]
pub fn compile_and_call(proc_name: auxtools::Value) -> DMResult {
    guard(|| {
        let context = unsafe { &mut LLVM_CONTEXT };
        let module_context = unsafe { &mut LLVM_MODULE_CONTEXT };
        let module_context = module_context.get_or_insert_with(|| ModuleContext::new(context));

        let mut override_id = 0;
        let base_proc = match proc_name.raw.tag {
            ValueTag::String => {
                Proc::find(proc_name.as_string().unwrap())
            }
            ValueTag::ProcId => {
                Proc::from_id(ProcId(unsafe { proc_name.raw.data.id }))
            }
            _ => Option::None
        };

        let name = if let Some(base_proc) = base_proc {
            base_proc.path
        } else {
            return DMResult::Ok(Value::from(false))
        };

        loop {
            if let Some(proc) = Proc::find_override(&name, override_id) {
                compile_proc(
                    context,
                    &module_context.module,
                    &mut module_context.meta_module,
                    proc,
                );
                override_id += 1;
            } else {
                break;
            }
        }

        DMResult::Ok(Value::from(true))
    })
}


#[hook("/proc/dmjit_install_compiled")]
pub fn install_hooks() -> DMResult {
    guard(|| {
        let mut installed: Vec<String> = vec!();
        let module_context = unsafe { &mut LLVM_MODULE_CONTEXT };
        if module_context.is_none() {
            return DMResult::Ok(Value::from(false))
        }
        let module_context = module_context.as_mut().unwrap();
        let dmir_meta_kind_id = module_context.module.get_context().get_kind_id("dmir");

        let module = &module_context.module;

        let mpm = PassManager::create(());

        mpm.add_always_inliner_pass();
        mpm.run_on(module);

        let fpm = PassManager::create(module);

        fpm.add_early_cse_mem_ssa_pass();
        fpm.add_loop_idiom_pass();
        fpm.add_licm_pass();
        fpm.add_instruction_combining_pass();
        fpm.add_cfg_simplification_pass();
        fpm.add_basic_alias_analysis_pass();
        fpm.add_scalar_repl_aggregates_pass();
        fpm.add_bit_tracking_dce_pass();
        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();
        fpm.add_cfg_simplification_pass();
        fpm.add_basic_alias_analysis_pass();
        fpm.add_promote_memory_to_register_pass();
        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();

        fpm.initialize();

        let mut curr_function = module.get_first_function();
        while let Some(func_value) = curr_function {
            let dmir_meta = func_value.get_metadata(dmir_meta_kind_id);
            if dmir_meta.is_some() {
                fpm.run_on(&func_value);
            }
            curr_function = func_value.get_next_function();
        }

        if let Err(err) = module.verify() {
            log::error!("err: {}", err.to_string());
        }

        unsafe {
            MEM_MANAGER = alloc_zeroed(Layout::new::<SectionMemoryManager>()).cast();
            *MEM_MANAGER = SectionMemoryManager::new();
        }


        let mm_ref = unsafe {
            SectionMemoryManager::create_mcjit_memory_manager(MEM_MANAGER)
        };

        let execution_engine =
            module.create_jit_execution_engine_with_options(|opts| {
                opts.OptLevel = OptimizationLevel::Default as u32;
                opts.MCJMM = mm_ref;
            }).unwrap();

        pads::bind_runtime_externals(module, &execution_engine);

        log::info!("Module {}", module.print_to_string().to_string());

        let mut curr_function = module.get_first_function();
        while let Some(func_value) = curr_function {
            if let Some(dmir_meta) = func_value.get_metadata(dmir_meta_kind_id) {
                let name = func_value.get_name().to_str().unwrap();

                log::info!("installing {}", name);
                installed.push(name.to_string());
                let func: ByondProcFunc = unsafe {
                    transmute_copy(&execution_engine.get_function_address(name).unwrap())
                };

                log::info!("target is {} at {:?}", name, func as (*mut ()));

                let proc_id_meta = dmir_meta.get_node_values()[0];
                let proc_id_value = proc_id_meta.into_int_value().get_zero_extended_constant().unwrap() as u32;
                let proc_id = auxtools::raw_types::procs::ProcId(proc_id_value);
                let proc = Proc::from_id(proc_id).unwrap();
                chad_hook_by_id(proc.id, func);
            }
            curr_function = func_value.get_next_function();
        }

        unsafe {
            log::debug!("StackMap section lookup started");
            let stack_map_section = (*MEM_MANAGER).sections.iter().find(|Section { name, .. }|
                name.as_c_str() == CStr::from_bytes_with_nul_unchecked(".llvm_stackmaps\0".as_bytes())
            );

            log::debug!("StackMap section: {:?}", stack_map_section);

            let stack_map = stack_map_section.map(|section| read_stack_map(section.address, section.size));
            log::trace!("StackMap: {:#?}", stack_map);

            let mut meta_update = module_context.meta_module.build_update_transaction(stack_map);

            log::trace!("Meta update: {:#?}", meta_update);

            meta_update.commit_transaction_to(&mut PROC_META);

            log::debug!("All sections is: {:#?}", *(MEM_MANAGER));
        }


        Value::from_string(installed.join(", "))
    })
}

pub(crate) static mut PROC_META: Vec<ProcMeta> = Vec::new();

struct ModuleContext<'ctx> {
    module: Module<'ctx>,
    meta_module: ProcMetaModuleBuilder,
}


pub(crate) static mut MEM_MANAGER: *mut SectionMemoryManager = std::ptr::null_mut();

impl <'a, 'b> ModuleContext<'b> {
    fn new(context: &'a Context) -> ModuleContext<'b> {

        let buf = MemoryBuffer::create_from_memory_range(include_bytes!("../target/runtime.bc"), "runtime.ll");
        let module =
            unsafe {
                &*(context as *const Context)
            }.create_module_from_ir(buf).unwrap();

        module.set_name("dmir");

        log::info!("Initialize ModuleContext");

        ModuleContext {
            module,
            meta_module: ProcMetaModuleBuilder::new(),
        }
    }
}


static mut LLVM_CONTEXT: Lazy<Context> = Lazy::new(|| Context::create());
static mut LLVM_MODULE_CONTEXT: Option<ModuleContext<'static>> = Option::None;


fn compile_proc<'ctx>(
    context: &'static Context,
    module: &'ctx Module<'static>,
    meta_module: &mut ProcMetaModuleBuilder,
    proc: auxtools::Proc,
) {
    let dmir_meta_kind_id = context.get_kind_id("dmir");
    log::info!("Trying compile {}", proc.path);

    // Take bytecode of proc to compile
    let bc = unsafe { proc.bytecode() };

    let mut env = DisassembleEnv {};
    // disassemble DM asm
    let (nodes, res) = dmasm::disassembler::disassemble(bc, &mut env);

    log::debug!("{}", format_disassembly(&nodes, None));

    if let Some(res) = res {
        panic!("{:?}", res);
    }

    let mut irs = dmir::decode_byond_bytecode(nodes, proc.clone()).unwrap();

    log::debug!("DMIR created");
    variable_termination_pass(&mut irs);
    log::debug!("variable_termination_pass done");

    generate_ref_count_operations2(&mut irs, proc.parameter_names().len());
    log::debug!("====== DFA done ======");
    generate_ref_count_operations(&mut irs, proc.parameter_names().len());
    log::debug!("ref_count_pass done");

    fn compute_max_sub_call_arg_count(ir: &DMIR, max_sub_call_arg_count: &mut u32) {
        match ir {
            DMIR::CallProcById(_, _, arg_count) | DMIR::CallProcByName(_, _, arg_count) => {
                *max_sub_call_arg_count = max(*max_sub_call_arg_count, *arg_count);
            }
            DMIR::IncRefCount { target: _, op } |
            DMIR::DecRefCount { target: _, op } |
            DMIR::InfLoopCheckDeopt(op) |
            DMIR::ListCheckSizeDeopt(_, _, op) |
            DMIR::CheckTypeDeopt(_, _, op) |
            DMIR::NewAssocList(_, op) => {
                compute_max_sub_call_arg_count(op, max_sub_call_arg_count);
            }
            _ => {}
        }
    }

    let mut max_sub_call_arg_count = 0;
    for ir in &irs {
        compute_max_sub_call_arg_count(ir, &mut max_sub_call_arg_count);
    }

    let meta_builder = meta_module.create_meta_builder(proc.id);

    // Prepare LLVM internals for code-generation
    let mut code_gen = CodeGen::create(
        context,
        &module,
        context.create_builder(),
        meta_builder,
        proc.parameter_names().len() as u32,
        proc.local_names().len() as u32,
    );

    let func = code_gen.create_jit_func(proc.path.as_str());

    let node = context.metadata_node(&[context.i32_type().const_int(proc.id.0 as u64, false).into()]);
    func.set_metadata(dmir_meta_kind_id, node);

    code_gen.emit_prologue(func, max_sub_call_arg_count);
    // Emit LLVM IR nodes from DMIR
    for ir in irs {
        log::debug!("emit: {:?}", &ir);
        code_gen.emit(&ir, func);
    }

    log::info!("{}", func.print_to_string().to_string());
    let verify = func.verify(true);

    if let Err(err) = code_gen.module.verify() {
        log::error!("err: {}", err.to_string());
    }

    log::info!("Verify {}: {}", proc.path, verify);
}
