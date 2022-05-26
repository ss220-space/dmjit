use std::borrow::Borrow;
use std::collections::HashMap;

use auxtools::raw_types::values::ValueTag;
use dmasm::list_operands::TypeFilter;
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::{Linkage, Module};
use inkwell::types::{StructType};
use inkwell::values::{BasicMetadataValueEnum, AnyValue, ArrayValue, BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue, PhiValue, PointerValue, StructValue};

use crate::dmir::{DMIR, RefOpDisposition, ValueLocation, ValueTagPredicate};
use crate::pads;
use crate::proc_meta::ProcMetaBuilder;

pub struct CodeGen<'ctx, 'a> {
    context: &'ctx Context,
    pub(crate) module: &'a Module<'ctx>,
    builder: Builder<'ctx>,
    stack_loc: Vec<StructValue<'ctx>>,
    locals: HashMap<u32, StructValue<'ctx>>,
    cache: Option<StructValue<'ctx>>,
    val_type: StructType<'ctx>,
    loop_iter_counter: IntValue<'ctx>,
    sub_call_arg_array_ptr: Option<PointerValue<'ctx>>,
    sub_ret_ptr: Option<PointerValue<'ctx>>,
    test_res: IntValue<'ctx>,
    internal_test_flag: Option<IntValue<'ctx>>,
    block_map: BlockMap<'ctx>,
    block_ended: bool,
    parameter_count: u32,
    local_count: u32,
    args: Vec<StructValue<'ctx>>,
    iterator_stack: Vec<PointerValue<'ctx>>,
    active_iterator: Option<PointerValue<'ctx>>,
    allocated_iterator: Option<PointerValue<'ctx>>,
    proc_meta_builder: &'a mut ProcMetaBuilder
}

type BlockMap<'ctx> = HashMap<String, LabelBlockInfo<'ctx>>;

struct LabelBlockInfo<'ctx> {
    block: BasicBlock<'ctx>,
    args: Vec<PhiValue<'ctx>>,
    locals: HashMap<u32, PhiValue<'ctx>>,
    stack: Vec<PhiValue<'ctx>>,
    iterator_stack: Vec<PointerValue<'ctx>>,
    active_iterator: Option<PointerValue<'ctx>>,
    allocated_iterator: Option<PointerValue<'ctx>>,
    cache: Option<PhiValue<'ctx>>,
    test_res: Option<PhiValue<'ctx>>,
    loop_iter_counter: Option<PhiValue<'ctx>>
}

#[derive(Clone)]
struct MetaValue<'ctx> {
    tag: IntValue<'ctx>,
    data: BasicValueEnum<'ctx>,
}

impl MetaValue<'_> {
    fn new<'a>(tag: IntValue<'a>, data: BasicValueEnum<'a>) -> MetaValue<'a> {
        return MetaValue {
            tag,
            data,
        };
    }

    fn with_tag<'a>(tag: auxtools::raw_types::values::ValueTag, data: BasicValueEnum<'a>, code_gen: &CodeGen<'a, '_>) -> MetaValue<'a> {
        let tag = code_gen.context.i8_type().const_int(tag as u64, false);
        return Self::new(tag, data);
    }
}


struct StackManager<'ctx, 'a, 'b> {
    code_gen: &'a mut CodeGen<'ctx, 'b>
}


impl<'ctx> StackManager<'ctx, '_, '_> {
    fn push(&mut self, value: StructValue<'ctx>) {
        self.code_gen.stack_loc.push(value);
    }

    fn pop(&mut self) -> StructValue<'ctx> {
        return self.code_gen.stack_loc.pop().unwrap();
    }
}

impl<'ctx, 'b> CodeGen<'ctx, 'b> {
    fn stack(&mut self) -> StackManager<'ctx, '_, 'b> {
        return StackManager {
            code_gen: self
        }
    }
}

macro_rules! decl_type {
    ($code_gen:ident DMValue) => (
        $code_gen.val_type
    );
    ($code_gen:ident i8_ptr) => (
        $code_gen.context.i8_type().ptr_type(AddressSpace::Generic)
    );
    ($code_gen:ident $arg:ident) => (
        $code_gen.context.$arg()
    );
}

macro_rules! decl_sig {
    ($code_gen:ident ($($arg:ident),+,...) -> $result:tt) => (
        decl_type!($code_gen $result).fn_type(&[$(decl_type!($code_gen $arg).into()),+], true)
    );
    ($code_gen:ident ($($arg:ident),+) -> $result:tt) => (
        decl_type!($code_gen $result).fn_type(&[$(decl_type!($code_gen $arg).into()),+], false)
    );
}

macro_rules! decl_intrinsic {
    ($code_gen:ident $name:literal $($sig:tt)+) => ({
        let function =
            if let Some(func) = $code_gen.module.get_function($name) {
                func
            } else {
                let t = decl_sig!($code_gen $($sig)+);
                $code_gen.module.add_function($name, t, None)
            };
        function
    });
}

struct BlockBuilder<'ctx, 'a> {
    context: &'ctx Context,
    builder: &'a Builder<'ctx>,
    val_type: &'a StructType<'ctx>
}

struct CodeGenValuesRef<'ctx, 'a> {
    args: &'a Vec<StructValue<'ctx>>,
    stack: &'a Vec<StructValue<'ctx>>,
    iterator_stack: &'a Vec<PointerValue<'ctx>>,
    active_iterator: &'a Option<PointerValue<'ctx>>,
    allocated_iterator: &'a Option<PointerValue<'ctx>>,
    locals: &'a HashMap<u32, StructValue<'ctx>>,
    cache: &'a Option<StructValue<'ctx>>,
    test_res: &'a IntValue<'ctx>,
    loop_iter_counter: &'a IntValue<'ctx>,
}

macro_rules! create_code_gen_values_ref {
    ($code_gen:ident) => {
        CodeGenValuesRef {
            args: &$code_gen.args,
            stack: &$code_gen.stack_loc,
            iterator_stack: &$code_gen.iterator_stack,
            active_iterator: &$code_gen.active_iterator,
            allocated_iterator: &$code_gen.allocated_iterator,
            locals: &$code_gen.locals,
            cache: &$code_gen.cache,
            test_res: &$code_gen.test_res,
            loop_iter_counter: &$code_gen.loop_iter_counter,
        }
    };
}


impl<'ctx, 'a> BlockBuilder<'ctx, 'a> {

    fn merge_vec(&self, target: &mut Vec<PhiValue<'ctx>>, source: &Vec<StructValue<'ctx>>, phi_name: &str, source_block: BasicBlock<'ctx>, is_new_block: bool) {
        let element_type = source.first().map(|element| element.get_type());
        if is_new_block {
            target.resize_with(source.len(), || self.builder.build_phi(element_type.unwrap(), phi_name));
        } else {
            assert_eq!(source.len(), target.len(), "values count mismatch when creating {}", phi_name);
        }
        for (idx, arg) in source.into_iter().enumerate() {
            target[idx].add_incoming(&[(arg, source_block)])
        }
    }

    fn merge_option<V: BasicValue<'ctx>>(&self, target: &mut Option<PhiValue<'ctx>>, source: &Option<V>, phi_name: &str, source_block: BasicBlock<'ctx>, is_new_block: bool) {
        if is_new_block {
            if let Some(source_value) = source {
                *target = Some(self.builder.build_phi(source_value.as_basic_value_enum().get_type(), phi_name));
            }
        } else {
            assert_eq!(source.is_some(), target.is_some(), "value presence differs when creating {}", phi_name)
        }
        if let Some(target) = target {
            let source = source.as_ref().unwrap();
            target.add_incoming(&[(source, source_block)])
        }
    }

    fn emit_jump_target_block<'b>(
        &'a mut self,
        block_map: &'b mut BlockMap<'ctx>,
        values: CodeGenValuesRef<'ctx, '_>,
        func: FunctionValue<'ctx>,
        lbl: &String
    ) -> &'b LabelBlockInfo<'ctx> {
        let current_block = self.builder.get_insert_block().unwrap();

        let mut new_block_created = false;
        let context = self.context;
        let entry = block_map.entry(lbl.clone())
            .or_insert_with(|| {
                new_block_created = true;
                LabelBlockInfo {
                    block: context.append_basic_block(func, lbl),
                    args: vec![],
                    locals: Default::default(),
                    stack: vec![],
                    iterator_stack: vec![],
                    active_iterator: None,
                    allocated_iterator: None,
                    cache: None,
                    test_res: None,
                    loop_iter_counter: None
                }
            });


        self.builder.position_at_end(entry.block);

        self.merge_vec(&mut entry.args, values.args, "arg_phi", current_block, new_block_created);

        for (idx, value) in values.locals {
            entry.locals.entry(*idx)
                .or_insert_with(|| {
                    assert!(new_block_created, "locals mismatch when creating phi");
                    self.builder.build_phi(self.val_type.clone(), "local_phi")
                })
                .add_incoming(&[(value, current_block)])
        }

        self.merge_vec(&mut entry.stack, values.stack, "stack_phi", current_block, new_block_created);

        // Note: No PHI nodes for iterators allowed
        if new_block_created {
            entry.iterator_stack = values.iterator_stack.clone();
            entry.active_iterator = values.active_iterator.clone();
            entry.allocated_iterator = values.allocated_iterator.clone();
        } else {
            assert_eq!(entry.iterator_stack.len(), values.iterator_stack.len());
            for (index, iter) in entry.iterator_stack.iter().enumerate() {
                assert_eq!(values.iterator_stack[index].as_any_value_enum(), iter.as_any_value_enum())
            }
            assert_eq!(&entry.active_iterator, values.active_iterator);
            assert_eq!(&entry.allocated_iterator, values.allocated_iterator);
        }

        self.merge_option(&mut entry.cache, values.cache, "cache_phi", current_block, new_block_created);
        self.merge_option(&mut entry.test_res, &Some(values.test_res.clone()), "test_res_phi", current_block, new_block_created);
        self.merge_option(&mut entry.loop_iter_counter, &Some(values.loop_iter_counter.clone()), "loop_iter_counter_phi", current_block, new_block_created);

        self.builder.position_at_end(current_block);

        entry
    }
}

impl<'ctx> CodeGen<'ctx, '_> {

    pub fn create<'a>(
        context: &'ctx Context,
        module: &'a Module<'ctx>,
        builder: Builder<'ctx>,
        proc_meta_builder: &'a mut ProcMetaBuilder,
        parameter_count: u32,
        local_count: u32,
    ) -> CodeGen<'ctx, 'a> {

        // TODO: Cleanup
        let val_type =
            if let Some(val_type) = module.get_struct_type("DMValue") {
                val_type
            } else {
                // type of BYOND operands, actually a struct { type: u8, value: u32 }
                let val_type = context.opaque_struct_type("DMValue");
                val_type.set_body(&[context.i8_type().into(), context.i32_type().into()], false);
                val_type
            };

        CodeGen {
            context,
            module,
            builder,
            stack_loc: Vec::new(),
            locals: HashMap::new(),
            cache: None,
            val_type,
            loop_iter_counter: context.i32_type().const_int(0xFFFFF, false),
            sub_call_arg_array_ptr: None,
            sub_ret_ptr: None,
            test_res: context.bool_type().const_int(false as u64, false),
            internal_test_flag: None,
            block_map: HashMap::new(),
            block_ended: false,
            parameter_count,
            local_count,
            args: Vec::new(),
            iterator_stack: Vec::new(),
            active_iterator: Option::None,
            allocated_iterator: Option::None,
            proc_meta_builder
        }
    }

    pub fn create_jit_func(&self, name: &str) -> FunctionValue<'ctx> {
        // each function in our case should be void *(ret, src)
        let ptr = self.val_type.ptr_type(AddressSpace::Generic);
        let ft = self.context.void_type().fn_type(
            &[
                ptr.into(),
                self.val_type.into(),
                self.val_type.into(),
                ptr.into(),
                self.context.i32_type().into()
            ],
            false,
        );

        // register function in LLVM
        let func = self.module.add_function(name, ft, None);

        // create start basic block for our function
        let block = self.context.append_basic_block(func, "base");
        self.builder.position_at_end(block);
        if cfg!(debug_on_call_print) {
            self.dbg(format!("llvm function call {}", name).as_str());
        }

        return func;
    }

    fn dbg(&self, str: &str) {
        let ptr = self.builder.build_global_string_ptr(str, "dbg_str").as_pointer_value();
        self.builder.build_call(self.module.get_function("dmir.runtime.debug.handle_debug").unwrap(), &[ptr.into()], "call_dbg");
    }

    fn dbg_val(&self, val: StructValue<'ctx>) {
        self.builder.build_call(self.module.get_function("dmir.runtime.debug.handle_debug_val").unwrap(), &[val.into()], "call_dbg");
    }


    fn emit_load_meta_value(&self, from: StructValue<'ctx>) -> MetaValue<'ctx> {
        let tag = self.builder.build_extract_value(from, 0, "get_tag").unwrap().into_int_value();
        let data = self.builder.build_extract_value(from, 1, "get_data").unwrap().into_int_value();
        return MetaValue::new(tag, data.into());
    }

    fn emit_store_meta_value(&self, from: MetaValue<'ctx>) -> StructValue<'ctx> {
        let out_val = self.val_type.const_zero();
        let out_val = self.builder.build_insert_value(out_val, from.tag, 0, "set_tag").unwrap().into_struct_value();
        let out_val = self.builder.build_insert_value(out_val, from.data, 1, "set_data").unwrap().into_struct_value();

        return out_val;
    }

    fn emit_bin_op<F>(&mut self, op: F)
        where F: FnOnce(MetaValue<'ctx>, MetaValue<'ctx>, &mut Self) -> MetaValue<'ctx>
    {
        let first_struct = self.stack().pop();
        let second_struct = self.stack().pop();

        let first_meta = self.emit_load_meta_value(first_struct);
        let second_meta = self.emit_load_meta_value(second_struct);

        let result_meta = op(first_meta, second_meta, self);

        let out_val = self.emit_store_meta_value(result_meta);

        self.stack().push(out_val);
    }

    fn const_tag(&self, value_tag: ValueTag) -> IntValue<'ctx> {
        return self.context.i8_type().const_int(value_tag as u64, false).into();
    }

    fn emit_to_number_or_zero(&self, func: FunctionValue<'ctx>, value: MetaValue<'ctx>) -> FloatValue<'ctx> {
        let before = self.builder.get_insert_block().unwrap();
        let iff_number = self.context.append_basic_block(func, "iff_number");
        let next = self.context.append_basic_block(func, "next");

        let zero_value = self.context.f32_type().const_zero();

        let number_tag = self.const_tag(ValueTag::Number);

        self.builder.build_conditional_branch(
            self.builder.build_int_compare(IntPredicate::EQ, value.tag, number_tag, "check_number"),
            iff_number,
            next,
        );
        let number_value = {
            self.builder.position_at_end(iff_number);
            let r = self.builder.build_bitcast(value.data, self.context.f32_type(), "cast_float").into_float_value();
            self.builder.build_unconditional_branch(next);
            r
        };

        self.builder.position_at_end(next);
        let out_phi = self.builder.build_phi(self.context.f32_type(), "to_number_or_zero_res");
        out_phi.add_incoming(&[(&zero_value, before), (&number_value, iff_number)]);

        return out_phi.as_basic_value().into_float_value()
    }

    fn emit_null_to_zero(&mut self, func: FunctionValue<'ctx>, value: MetaValue<'ctx>) -> MetaValue<'ctx> {
        let tag_check = self.emit_tag_predicate_comparison(value.tag, &ValueTagPredicate::Tag(ValueTag::Null));

        let iff_null_block = self.context.append_basic_block(func, "iff_null");
        let next_block = self.context.append_basic_block(func, "next_block");
        let prev_block = self.builder.get_insert_block().unwrap();

        let original_value_packed = self.emit_store_meta_value(value);

        self.builder.build_conditional_branch(
            tag_check,
            iff_null_block,
            next_block
        );

        self.builder.position_at_end(iff_null_block);
        let zero_i32 = self.builder.build_bitcast(self.context.f32_type().const_zero(), self.context.i32_type(), "zero_to_int").into_int_value();
        let literal_zero = MetaValue::with_tag(ValueTag::Number, zero_i32.into(), self);
        let literal_zero_packed = self.emit_store_meta_value(literal_zero);
        self.builder.build_unconditional_branch(next_block);

        self.builder.position_at_end(next_block);
        let phi = self.builder.build_phi(self.val_type, "merge_result");
        phi.add_incoming(&[(&original_value_packed, prev_block), (&literal_zero_packed, iff_null_block)]);

        self.emit_load_meta_value(phi.as_basic_value().into_struct_value())
    }

    fn emit_lifetime_start(&mut self, pointer: PointerValue<'ctx>, name: &str) {
        let func = decl_intrinsic!(self "llvm.lifetime.start.p0i8" (i64_type, i8_ptr) -> void_type);

        let value_ptr = self.builder.build_pointer_cast(pointer, self.context.i8_type().ptr_type(AddressSpace::Generic), "ptr_cast");
        self.builder.build_call(
            func,
            &[
                self.context.i64_type().const_int((-1i64) as u64, true).into(),
                value_ptr.into()
            ],
            name
        );
    }

    fn emit_lifetime_end(&mut self, pointer: PointerValue<'ctx>, name: &str) {
        let func = decl_intrinsic!(self "llvm.lifetime.end.p0i8" (i64_type, i8_ptr) -> void_type);

        let value_ptr = self.builder.build_pointer_cast(pointer, self.context.i8_type().ptr_type(AddressSpace::Generic), "ptr_cast");
        self.builder.build_call(
            func,
            &[
                self.context.i64_type().const_int((-1i64) as u64, true).into(),
                value_ptr.into()
            ],
            name
        );
    }

    fn emit_pop_stack_to_array(&mut self, count: u32) -> ArrayValue<'ctx> {
        let out_stack_type = self.val_type.array_type(count as u32);
        let mut stack_out_array = out_stack_type.const_zero();
        for idx in (0..count).rev() {
            let arg = self.stack().pop();
            stack_out_array = self.builder.build_insert_value(stack_out_array, arg, idx as u32, "store_value_from_stack").unwrap().into_array_value();
        }
        return stack_out_array
    }

    // Actual return type is i1/bool
    fn emit_check_is_true(&self, value: MetaValue<'ctx>) -> IntValue<'ctx> {
        let is_null = self.builder.build_int_compare(IntPredicate::EQ, value.tag, self.const_tag(ValueTag::Null), "is_null");
        let is_x22 = self.builder.build_int_compare(IntPredicate::EQ, value.tag, self.context.i8_type().const_int(0x22, false), "is_x22");

        let is_false_tag = self.builder.build_or(is_null, is_x22, "check_false_tag");

        let is_num = self.builder.build_int_compare(IntPredicate::EQ, value.tag, self.const_tag(ValueTag::Number), "is_num");
        let is_str = self.builder.build_int_compare(IntPredicate::EQ, value.tag, self.const_tag(ValueTag::String), "is_str");

        let not_num_or_not_zero = self.builder.build_or(
            self.builder.build_not(is_num, "!is_num"),
            self.builder.build_float_compare(
                FloatPredicate::UNE,
                self.builder.build_bitcast(value.data, self.context.f32_type(), "cast_f32").into_float_value(),
                self.context.f32_type().const_zero(),
                "check_ne_zero"
            ),
            "not_num_or_not_zero"
        );

        let not_str_or_not_zero = self.builder.build_or(
            self.builder.build_not(is_str, "!is_str"),
            self.builder.build_int_compare(IntPredicate::NE, value.data.into_int_value(), self.context.i32_type().const_zero(), "check_ne_zero"),
            "not_str_or_not_zero"
        );


        // = !(is_null || is_x22) && (!is_num || value != 0.0) && (!is_str || value != 0)
        return self.builder.build_and(
            self.builder.build_not(is_false_tag, "not_(null_or_x22)"),
            self.builder.build_and(
                not_num_or_not_zero,
                not_str_or_not_zero,
                "is_true_data"
            ),
            "check_is_true"
        )
    }

    fn emit_conditional_jump(&mut self, func: FunctionValue<'ctx>, lbl: &String, condition: IntValue<'ctx>) {
        let next = self.context.append_basic_block(func, "next");

        let values = create_code_gen_values_ref!(self);
        let mut block_builder = BlockBuilder {
            context: self.context,
            builder: &self.builder,
            val_type: &self.val_type
        };
        let target = block_builder.emit_jump_target_block(&mut self.block_map, values, func, lbl);
        let target_block = target.block.clone();

        self.builder.build_conditional_branch(
            condition,
            target_block,
            next,
        );

        self.builder.position_at_end(next);
    }

    fn emit_boolean_to_number(&self, bool: IntValue<'ctx>) -> MetaValue<'ctx> {
        let bool_f32 = self.builder.build_unsigned_int_to_float(bool, self.context.f32_type(), "to_f32");
        let res_i32 = self.builder.build_bitcast(bool_f32, self.context.i32_type(), "to_i32").into_int_value();
        return MetaValue::with_tag(ValueTag::Number, res_i32.into(), self);
    }

    fn emit_inc_ref_count(&self, value: StructValue<'ctx>) {
        let func = self.module.get_function("dmir.runtime.inc_ref_count").unwrap();
        self.builder.build_call(func, &[value.into()], "call_inc_ref_count");
    }

    fn emit_dec_ref_count(&self, value: StructValue<'ctx>) {
        let func = self.module.get_function("dmir.runtime.dec_ref_count").unwrap();
        self.builder.build_call(func, &[value.into()], "call_dec_ref_count");
    }

    fn emit_meta_value_to_int(&self, value: MetaValue<'ctx>) -> IntValue<'ctx> {
        let f32 = self.builder.build_bitcast(value.data, self.context.f32_type(), "cast_to_float").into_float_value();

        return self.builder.build_float_to_signed_int(f32, self.context.i32_type(), "float_to_int");
    }

    fn emit_read_value_location(&self, location: &ValueLocation) -> StructValue<'ctx> {
        match location {
            ValueLocation::Stack(rel) => {
                self.stack_loc[self.stack_loc.len() - 1 - (*rel as usize)]
            }
            ValueLocation::Cache => {
                self.cache.unwrap()
            }
            ValueLocation::Local(idx) => {
                self.locals.get(idx).unwrap().clone()
            }
            ValueLocation::Argument(idx) => {
                self.args[*idx as usize]
            }
        }
    }

    fn emit_epilogue(&mut self, func: FunctionValue<'ctx>) {
        let iterators = self.iterator_stack.drain(..).collect::<Vec<_>>();
        for iterator in iterators {
            self.emit_iterator_destructor(iterator);
        }

        let caller_arg_count = func.get_nth_param(4).unwrap().into_int_value();
        let expected_arg_count = self.context.i32_type().const_int(self.parameter_count as u64, false);
        let arg_ptr = func.get_nth_param(3).unwrap().into_pointer_value();

        let unref_excess_arguments_func = self.module.get_function("dmir.intrinsic.unref_excess_arguments").unwrap();
        self.builder.build_call(
            unref_excess_arguments_func,
            &[
                arg_ptr.into(),
                expected_arg_count.into(),
                caller_arg_count.into()
            ],
            "call_unref_excess_args"
        );
    }

    fn emit_iterator_destructor(&self, iterator_value: PointerValue<'ctx>) {
        let destroy_func = self.module.get_function("dmir.intrinsic.iter.destroy").unwrap();
        self.builder.build_call(
            destroy_func,
            &[
                iterator_value.into()
            ],
            "iterator_destroy"
        );
    }

    fn emit_iterator_constructor(&mut self, call: &str, type_filter: &TypeFilter) {

        if let Some(prev) = self.active_iterator.take() {
            self.emit_iterator_destructor(prev);
        }

        let load_func = self.module.get_function(call).unwrap();
        let source = self.stack().pop();
        let new_iterator = self.allocated_iterator.take().unwrap();

        let mut type_filter = type_filter.clone();
        if type_filter.contains(TypeFilter::ANYTHING) {
            type_filter = TypeFilter::empty()
        }

        self.builder.build_call(
            load_func,
            &[
                new_iterator.into(),
                source.into(),
                self.context.i32_type().const_int(type_filter.bits() as u64, false).into(),
            ],
            "load_iterator"
        );
        self.active_iterator = Some(new_iterator);
    }

    fn emit_tag_predicate_comparison(&self, actual_tag: IntValue<'ctx>, predicate: &ValueTagPredicate) -> IntValue<'ctx> {
        match predicate {
            ValueTagPredicate::Any => self.context.bool_type().const_int(1, false),
            ValueTagPredicate::None => self.context.bool_type().const_int(0, false),
            ValueTagPredicate::Tag(tag) => self.builder.build_int_compare(IntPredicate::EQ, actual_tag, self.const_tag(tag.clone()), "check_tag"),
            ValueTagPredicate::Union(tags) => {
                let mut last_op = self.context.bool_type().const_int(1, false);
                for pred in tags.iter() {
                    last_op = self.builder.build_or(
                        last_op,
                        self.emit_tag_predicate_comparison(actual_tag, pred),
                        "or"
                    );
                }
                last_op
            }
        }

    }

    pub fn emit_prologue(&mut self, func: FunctionValue<'ctx>, max_sub_call_arg_count: u32) {

        self.builder.build_store(func.get_nth_param(0).unwrap().into_pointer_value(), self.val_type.const_zero()); // initialize out

        let caller_arg_count = func.get_nth_param(4).unwrap().into_int_value();
        let args_ptr = func.get_nth_param(3).unwrap().into_pointer_value();

        for arg_num in 0..self.parameter_count {

            let before_block = self.builder.get_insert_block().unwrap();
            let load_arg_block = self.context.append_basic_block(func, format!("arg_load_{}", arg_num).as_ref());
            let post_load_arg_block = self.context.append_basic_block(func, format!("post_arg_load_{}", arg_num).as_ref());

            // if arg_num < caller_arg_count
            self.builder.build_conditional_branch(
                self.builder.build_int_compare(
                    IntPredicate::ULT,
                    self.context.i32_type().const_int(arg_num as u64, false),
                    caller_arg_count,
                    "arg_num < caller_arg_count"
                ),
                load_arg_block,
                post_load_arg_block
            );


            // then
            self.builder.position_at_end(load_arg_block);
            let arg_ptr = unsafe {
                self.builder.build_gep(
                    args_ptr,
                    &[
                        self.context.i64_type().const_int(arg_num as u64, false)
                    ],
                    "arg_ptr"
                )
            };
            let loaded_val = self.builder.build_load(arg_ptr, "load_arg").into_struct_value();
            self.builder.build_unconditional_branch(post_load_arg_block);


            // else
            let null_val = self.val_type.const_zero();

            // post
            self.builder.position_at_end(post_load_arg_block);
            let arg_phi = self.builder.build_phi(self.val_type, "arg_phi");
            arg_phi.add_incoming(
                &[
                    (&loaded_val, load_arg_block),
                    (&null_val, before_block)
                ]
            );
            let arg_value = arg_phi.as_basic_value().into_struct_value();

            self.args.push(arg_value);
        }


        self.sub_call_arg_array_ptr = Option::Some(self.builder.build_alloca(self.val_type.array_type(max_sub_call_arg_count as u32), "proc_args_ptr"));
        self.sub_ret_ptr = Option::Some(self.builder.build_alloca(self.val_type, "proc_ret_ptr"));
    }

    pub fn emit(&mut self, ir: &DMIR, func: FunctionValue<'ctx>) {

        match ir {
            DMIR::Nop => {}
            // Load src onto stack
            DMIR::GetSrc => {
                // self.dbg("GetSrc");
                self.stack().push(func.get_nth_param(1).unwrap().into_struct_value());
            }
            // Set cache to stack top
            DMIR::SetCache => {
                let arg = self.stack().pop();
                self.cache = Some(arg);
            }
            DMIR::PushCache => {
                let cache = self.cache.unwrap();
                self.stack().push(cache);
            }
            // Read field from cache e.g cache["oxygen"] where "oxygen" is interned string at name_id
            DMIR::GetCacheField(name_id) => {
                let get_var_func = self.module.get_function("dmir.runtime.get_variable").unwrap();

                let receiver_value = self.cache.unwrap();

                self.emit_lifetime_start(self.sub_ret_ptr.unwrap(), "out_lifetime_start");
                self.builder.build_call(get_var_func, &[self.sub_ret_ptr.unwrap().into(), receiver_value.into(), self.context.i32_type().const_int(name_id.clone() as u64, false).into()], "get_cache_field");

                let out_value = self.builder.build_load(self.sub_ret_ptr.unwrap(), "out_value").into_struct_value();
                self.emit_lifetime_end(self.sub_ret_ptr.unwrap(), "out_lifetime_end");
                self.stack().push(out_value);
                // self.dbg("GetVar");
                // self.dbg_val(self.builder.build_load(out, "load_dbg").into_struct_value());
            }
            DMIR::SetCacheField(name_id) => {
                let value = self.stack().pop();

                let set_var_func = self.module.get_function("dmir.runtime.set_variable").unwrap();
                let receiver_value = self.cache.unwrap();

                self.builder.build_call(
                    set_var_func,
                    &[
                        receiver_value.into(),
                        self.context.i32_type().const_int(name_id.clone() as u64, false).into(),
                        value.into()
                    ],
                    "set_cache_field"
                );
            }
            DMIR::ValueTagSwitch(location, cases) => {
                let value = self.emit_read_value_location(location);
                let meta_value = self.emit_load_meta_value(value);
                let mut jumps = Vec::new();
                let mut default = Option::None;
                for (predicate, block) in cases {
                    let mut predicates = Vec::new();
                    fn to_linear(predicate: &ValueTagPredicate, out: &mut Vec<ValueTagPredicate>) {
                        match predicate {
                            ValueTagPredicate::Union(values) => {
                                for i in values {
                                    to_linear(i, out)
                                }
                            }
                            _ => { out.push(predicate.clone()) }
                        }
                    }
                    to_linear(predicate, &mut predicates);

                    for predicate in predicates {
                        let values = create_code_gen_values_ref!(self);
                        let mut block_builder = BlockBuilder {
                            context: self.context,
                            builder: &self.builder,
                            val_type: &self.val_type,
                        };
                        let target = block_builder.emit_jump_target_block(&mut self.block_map, values, func, block).block;
                        if let ValueTagPredicate::Tag(t) = predicate {
                            jumps.push((self.const_tag(t), target))
                        } else if matches!(predicate, ValueTagPredicate::Any) {
                            default = Option::Some(target)
                        }
                    }
                }
                self.builder.build_switch(meta_value.tag, default.unwrap(), &jumps);
                self.block_ended = true;
            }
            // Read two values from stack, add them together, put result to stack
            DMIR::FloatAdd => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_f32 = code_gen.builder.build_bitcast(first.data, code_gen.context.f32_type(), "first_f32").into_float_value();
                    let second_f32 = code_gen.builder.build_bitcast(second.data, code_gen.context.f32_type(), "second_f32").into_float_value();

                    let result_value = code_gen.builder.build_float_add(first_f32, second_f32, "add");
                    let result_i32 = code_gen.builder.build_bitcast(result_value, code_gen.context.i32_type(), "result_i32").into_int_value();

                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                })
            }
            DMIR::FloatSub => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_f32 = code_gen.builder.build_bitcast(first.data, code_gen.context.f32_type(), "first_f32").into_float_value();
                    let second_f32 = code_gen.emit_to_number_or_zero(func, second);

                    let result_value = code_gen.builder.build_float_sub(second_f32, first_f32, "sub");
                    let result_i32 = code_gen.builder.build_bitcast(result_value, code_gen.context.i32_type(), "result_i32").into_int_value();

                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                })
            }
            DMIR::BitAnd => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_i32 = code_gen.emit_meta_value_to_int(first);
                    let second_i32 = code_gen.emit_meta_value_to_int(second);

                    let result_value = code_gen.builder.build_and(second_i32, first_i32, "and");

                    let result_f32 = code_gen.builder.build_signed_int_to_float(result_value, code_gen.context.f32_type(), "result_f32");

                    let result_i32 = code_gen.builder.build_bitcast(result_f32, code_gen.context.i32_type(), "result_i32").into_int_value();

                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                })
            }
            DMIR::BitOr => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_i32 = code_gen.emit_meta_value_to_int(first);
                    let second_i32 = code_gen.emit_meta_value_to_int(second);

                    let result_value = code_gen.builder.build_or(second_i32, first_i32, "or");

                    let result_f32 = code_gen.builder.build_signed_int_to_float(result_value, code_gen.context.f32_type(), "result_f32");

                    let result_i32 = code_gen.builder.build_bitcast(result_f32, code_gen.context.i32_type(), "result_i32").into_int_value();

                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                })
            }
            DMIR::FloatMul => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_f32 = code_gen.emit_to_number_or_zero(func, first);
                    let second_f32 = code_gen.builder.build_bitcast(second.data, code_gen.context.f32_type(), "first_f32").into_float_value();

                    let result_value = code_gen.builder.build_float_mul(first_f32, second_f32, "mul");
                    let result_i32 = code_gen.builder.build_bitcast(result_value, code_gen.context.i32_type(), "result_i32").into_int_value();
                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                });
            }
            DMIR::FloatDiv => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_f32 = code_gen.builder.build_bitcast(first.data, code_gen.context.f32_type(), "first_f32").into_float_value();
                    let second_f32 = code_gen.builder.build_bitcast(second.data, code_gen.context.f32_type(), "second_f32").into_float_value();

                    let result_value = code_gen.builder.build_float_div(second_f32, first_f32, "div");
                    let result_i32 = code_gen.builder.build_bitcast(result_value, code_gen.context.i32_type(), "result_i32").into_int_value();
                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                })
            }
            DMIR::FloatCmp(predicate) => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_not_null = code_gen.emit_null_to_zero(func, first);
                    let second_not_null = code_gen.emit_null_to_zero(func, second);
                    let first_f32 = code_gen.builder.build_bitcast(first_not_null.data, code_gen.context.f32_type(), "first_f32").into_float_value();
                    let second_f32 = code_gen.builder.build_bitcast(second_not_null.data, code_gen.context.f32_type(), "second_f32").into_float_value();

                    let result_value = code_gen.builder.build_float_compare(predicate.clone(), second_f32, first_f32, "test_pred");
                    let result_f32 = code_gen.builder.build_unsigned_int_to_float(result_value, code_gen.context.f32_type(), "bool_to_f32");
                    let result_i32 = code_gen.builder.build_bitcast(result_f32, code_gen.context.i32_type(), "f32_as_i32").into_int_value();

                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                })
            }
            DMIR::FloatAbs => {
                let arg_value = self.stack().pop();
                let arg = self.emit_load_meta_value(arg_value);
                let arg_num = self.emit_to_number_or_zero(func, arg);

                let fabs = decl_intrinsic!(self "llvm.fabs.f32" (f32_type) -> f32_type);

                let result = self.builder.build_call(fabs, &[arg_num.into()], "abs").try_as_basic_value().left().unwrap().into_float_value();

                let result_i32 = self.builder.build_bitcast(result, self.context.i32_type(), "cast_result").into_int_value();
                let meta_result = MetaValue::with_tag(ValueTag::Number, result_i32.into(), self);
                let result_value = self.emit_store_meta_value(meta_result);
                self.stack().push(result_value);
            }
            DMIR::FloatInc => {
                let arg_value = self.stack().pop();
                let arg = self.emit_load_meta_value(arg_value);
                let arg_num = self.emit_to_number_or_zero(func, arg);

                let one_const = self.context.f32_type().const_float(1.0);

                let result_value = self.builder.build_float_add(arg_num, one_const, "add");

                let result_i32 = self.builder.build_bitcast(result_value, self.context.i32_type(), "result_i32").into_int_value();

                let meta_result = MetaValue::with_tag(ValueTag::Number, result_i32.into(), self);
                let result_value = self.emit_store_meta_value(meta_result);
                self.stack().push(result_value);
            }
            DMIR::FloatDec => {
                let arg_value = self.stack().pop();
                let arg = self.emit_load_meta_value(arg_value);
                let arg_num = self.emit_to_number_or_zero(func, arg);

                let one_const = self.context.f32_type().const_float(1.0);

                let result_value = self.builder.build_float_sub(arg_num, one_const, "add");

                let result_i32 = self.builder.build_bitcast(result_value, self.context.i32_type(), "result_i32").into_int_value();

                let meta_result = MetaValue::with_tag(ValueTag::Number, result_i32.into(), self);
                let result_value = self.emit_store_meta_value(meta_result);
                self.stack().push(result_value);
            }
            DMIR::RoundN => {
                self.emit_bin_op(|first, second, code_gen| {
                    let first_f32 = code_gen.emit_to_number_or_zero(func, first.clone());
                    let second_f32 = code_gen.emit_to_number_or_zero(func, second.clone());
                    // first_f32: round to
                    // second_f32: value to round
                    let divided = code_gen.builder.build_float_div(second_f32, first_f32, "div");
                    let divided_f64 =
                        code_gen.builder.build_float_ext(divided, code_gen.context.f64_type(), "extend_to_double");

                    let value_to_floor = code_gen.builder.build_float_add(divided_f64, code_gen.context.f64_type().const_float(0.5), "add_const");

                    let floor = decl_intrinsic!(code_gen "llvm.floor.f64" (f64_type) -> f64_type);
                    let discrete_counts = code_gen.builder.build_call(floor, &[value_to_floor.into()], "call_floor").try_as_basic_value().left().unwrap().into_float_value();
                    let discrete_counts_f32 = code_gen.builder.build_float_trunc(discrete_counts, code_gen.context.f32_type(), "trunc_to_float");
                    let result = code_gen.builder.build_float_mul(discrete_counts_f32, first_f32, "mul");
                    let result_i32 = code_gen.builder.build_bitcast(result, code_gen.context.i32_type(), "cast_result").into_int_value();

                    MetaValue::with_tag(ValueTag::Number, result_i32.into(), code_gen)
                })
            }
            DMIR::ListCopy => {
                let list_struct = self.stack().pop();

                let list_copy = self.module.get_function("dmir.runtime.list_copy").unwrap();

                let result = self.builder.build_call(
                    list_copy,
                    &[
                        list_struct.into()
                    ], "list_copy").as_any_value_enum().into_struct_value();

                self.stack().push(result);
            }
            DMIR::ListAddSingle => {
                let value_struct = self.stack().pop();
                let list_struct = self.stack().pop();

                let list_append = self.module.get_function("dmir.runtime.list_append").unwrap();

                self.builder.build_call(
                    list_append,
                    &[
                        list_struct.into(),
                        value_struct.into()
                    ], "list_append");
            }
            DMIR::ListSubSingle => {
                let value_struct = self.stack().pop();
                let list_struct = self.stack().pop();

                let list_remove = self.module.get_function("dmir.runtime.list_remove").unwrap();

                self.builder.build_call(
                    list_remove,
                    &[
                        list_struct.into(),
                        value_struct.into()
                    ], "list_remove");
            }
            DMIR::ListIndexedGet => {
                let index_struct = self.stack().pop();
                let list_struct = self.stack().pop();

                let index_meta = self.emit_load_meta_value(index_struct);
                let index_i32 = self.emit_meta_value_to_int(index_meta);

                let list_indexed_get = self.module.get_function("dmir.intrinsic.list_indexed_get").unwrap();

                let result = self.builder.build_call(
                    list_indexed_get,
                    &[
                        list_struct.into(),
                        index_i32.into()
                    ], "list_indexed_get").as_any_value_enum().into_struct_value();

                self.stack().push(result);
            }
            DMIR::ListIndexedSet => {
                let index_struct = self.stack().pop();
                let list_struct = self.stack().pop();
                let value_struct = self.stack().pop();

                let index_meta = self.emit_load_meta_value(index_struct);
                let index_i32 = self.emit_meta_value_to_int(index_meta);

                let list_indexed_set = self.module.get_function("dmir.intrinsic.list_indexed_set").unwrap();

                self.builder.build_call(
                    list_indexed_set,
                    &[
                        list_struct.into(),
                        index_i32.into(),
                        value_struct.into()
                    ], "list_indexed_set");
            }
            DMIR::ListAssociativeGet => {
                let index_struct = self.stack().pop();
                let list_struct = self.stack().pop();

                let list_indexed_get = self.module.get_function("dmir.runtime.list_associative_get").unwrap();

                let result = self.builder.build_call(
                    list_indexed_get,
                    &[
                        list_struct.into(),
                        index_struct.into()
                    ], "list_associative_get").as_any_value_enum().into_struct_value();

                self.stack().push(result);
            }
            DMIR::ListAssociativeSet => {
                let index_struct = self.stack().pop();
                let list_struct = self.stack().pop();
                let value_struct = self.stack().pop();

                let list_indexed_set = self.module.get_function("dmir.runtime.list_associative_set").unwrap();

                self.builder.build_call(
                    list_indexed_set,
                    &[
                        list_struct.into(),
                        index_struct.into(),
                        value_struct.into()
                    ], "list_associative_set");
            }
            DMIR::NewAssocList(count, deopt) => {
                let mut num_check = self.context.bool_type().const_int(0, false);
                for i in 0..count.clone() {
                    let key = self.stack_loc[self.stack_loc.len() - 2 - (i as usize)];
                    let actual_tag = self.emit_load_meta_value(key).tag;
                    num_check = self.builder.build_or(num_check, self.emit_tag_predicate_comparison(actual_tag, &ValueTagPredicate::Tag(ValueTag::Number)), "check_for_numbers");
                }

                let next_block = self.context.append_basic_block(func, "next");
                let deopt_block = self.context.append_basic_block(func, "deopt");

                self.builder.build_conditional_branch(
                    num_check,
                    deopt_block,
                    next_block
                );

                self.builder.position_at_end(deopt_block);
                if cfg!(debug_deopt_print) {
                    self.dbg(format!("NewAssocListKeyCheck({:?}, {:?}) failed: ", count, deopt).as_str());
                }
                self.emit(deopt.borrow(), func);
                self.builder.build_unconditional_branch(next_block);
                self.builder.position_at_end(next_block);

                let mut args = vec![];
                for _ in 0..count.clone() * 2 {
                    args.push(self.stack().pop());
                }

                let create_new_list = self.module.get_function("dmir.runtime.create_new_list").unwrap();
                let result = self.builder.build_call(
                    create_new_list,
                    &[
                        self.context.i32_type().const_int(0, false).into()
                    ], "create_new_list").as_any_value_enum().into_int_value();

                let result_meta = MetaValue::with_tag(ValueTag::List, result.into(), self);
                let result = self.emit_store_meta_value(result_meta);

                let list_associative_set = self.module.get_function("dmir.runtime.list_associative_set").unwrap();
                for _ in 0..count.clone() {
                    self.builder.build_call(
                        list_associative_set,
                        &[
                            result.into(),
                            args.pop().unwrap().into(),
                            args.pop().unwrap().into()
                        ], "list_associative_set");
                }

                self.stack().push(result);
            }
            DMIR::NewVectorList(count) => {
                let mut args = vec![];
                for _ in 0..count.clone() {
                    args.push(self.stack().pop());
                }

                let create_new_list = self.module.get_function("dmir.runtime.create_new_list").unwrap();
                let result = self.builder.build_call(
                    create_new_list,
                    &[
                        self.context.i32_type().const_int(count.clone() as u64, false).into()
                    ], "create_new_list").as_any_value_enum().into_int_value();

                let result_meta = MetaValue::with_tag(ValueTag::List, result.into(), self);
                let result = self.emit_store_meta_value(result_meta);

                let get_list_vector_part = self.module.get_function("dmir.intrinsic.get_list_vector_part").unwrap();
                let vector_part = self.builder.build_call(
                    get_list_vector_part,
                    &[
                        result.into()
                    ], "get_list_vector_part").as_any_value_enum().into_pointer_value();

                let list_indexed_set_internal = self.module.get_function("dmir.intrinsic.list_indexed_set_internal").unwrap();
                for index in 0..count.clone() {
                    self.builder.build_call(
                        list_indexed_set_internal,
                        &[
                            vector_part.into(),
                            self.context.i32_type().const_int(index.clone() as u64, false).into(),
                            args.pop().unwrap().into()
                        ], "list_indexed_set_internal");
                }

                self.stack().push(result);
            }
            DMIR::GetStep => {
                let first = self.stack().pop();
                let second = self.stack().pop();

                let dir = self.emit_load_meta_value(first);
                let dir_num = self.emit_to_number_or_zero(func, dir);

                let dir_u8 = self.builder.build_float_to_unsigned_int(dir_num, self.context.i8_type(), "dir_to_u8");

                let get_step = self.module.get_function("dmir.runtime.get_step").unwrap();

                let res = self.builder.build_call(
                    get_step,
                    &[
                        second.into(),
                        dir_u8.into()
                    ], "get_step").as_any_value_enum().into_struct_value();
                self.stack().push(res);
            }
            DMIR::CallProcById(proc_id, proc_call_type, arg_count) => {
                let src = self.stack().pop();

                let args_array = self.emit_pop_stack_to_array(arg_count.clone());
                let args_ptr = self.builder.build_pointer_cast(self.sub_call_arg_array_ptr.unwrap(), args_array.get_type().ptr_type(AddressSpace::Generic), "to_args_ptr");

                self.emit_lifetime_start(args_ptr, "args_lifetime_start");
                self.emit_lifetime_start(self.sub_ret_ptr.unwrap(), "out_lifetime_start");
                self.builder.build_store(args_ptr, args_array);
                let args = self.builder.build_pointer_cast(args_ptr, self.val_type.ptr_type(AddressSpace::Generic), "to_raw_ptr");

                let call_proc_by_id = self.module.get_function("dmir.runtime.call_proc_by_id").unwrap();

                let usr = func.get_nth_param(2).unwrap().into_struct_value(); // TODO: Proc can change self usr

                self.builder.build_call(
                    call_proc_by_id,
                    &[
                        self.sub_ret_ptr.unwrap().into(), // out: *mut values::Value,
                        usr.into(), // usr: values::Value,
                        self.context.i32_type().const_int(proc_call_type.clone() as u64, false).into(), // proc_type: u32,
                        self.context.i32_type().const_int(proc_id.0 as u64, false).into(), // proc_id: procs::ProcId,
                        self.context.i32_type().const_int(0, false).into(),  // unk_0: u32,
                        src.into(), // src: values::Value,
                        args.into(), // args: *const values::Value,
                        self.context.i32_type().const_int(arg_count.clone() as u64, false).into(), // args_count_l: usize,
                        self.context.i32_type().const_int(0, false).into(), // unk_1: u32,
                        self.context.i32_type().const_int(0, false).into(), // unk_2: u32,
                    ],
                    "call_proc_by_id",
                );

                let out_value = self.builder.build_load(self.sub_ret_ptr.unwrap(), "call_result_value").into_struct_value();
                self.emit_lifetime_end(args_ptr, "args_lifetime_end");
                self.emit_lifetime_end(self.sub_ret_ptr.unwrap(), "out_lifetime_end");

                self.stack().push(out_value);
            }
            DMIR::CallProcByName(string_id, proc_call_type, arg_count) => {
                let src = self.stack().pop();

                let args_array = self.emit_pop_stack_to_array(arg_count.clone());
                let args_ptr = self.builder.build_pointer_cast(self.sub_call_arg_array_ptr.unwrap(), args_array.get_type().ptr_type(AddressSpace::Generic), "to_args_ptr");

                self.emit_lifetime_start(args_ptr, "args_lifetime_start");
                self.emit_lifetime_start(self.sub_ret_ptr.unwrap(), "out_lifetime_start");
                self.builder.build_store(args_ptr, args_array);
                let args = self.builder.build_pointer_cast(args_ptr, self.val_type.ptr_type(AddressSpace::Generic), "to_raw_ptr");

                let call_proc_by_name = self.module.get_function("dmir.runtime.call_proc_by_name").unwrap();

                let usr = func.get_nth_param(2).unwrap().into_struct_value(); // TODO: Proc can change self usr

                self.builder.build_call(
                    call_proc_by_name,
                    &[
                        self.sub_ret_ptr.unwrap().into(), //out: *mut values::Value,
                        usr.into(), //usr: values::Value,
                        self.context.i32_type().const_int(proc_call_type.clone() as u64, false).into(), //proc_type: u32,
                        self.context.i32_type().const_int(string_id.0 as u64, false).into(), //proc_name: strings::StringId,
                        src.into(), //src: values::Value,
                        args.into(), //args: *mut values::Value,
                        self.context.i32_type().const_int(arg_count.clone() as u64, false).into(), //args_count_l: usize,
                        self.context.i32_type().const_int(0, false).into(), //unk_0: u32,
                        self.context.i32_type().const_int(0, false).into(), //unk_1: u32,
                    ],
                    "call_proc_by_name",
                );

                let out_value = self.builder.build_load(self.sub_ret_ptr.unwrap(), "call_result_value").into_struct_value();
                self.emit_lifetime_end(args_ptr, "args_lifetime_end");
                self.emit_lifetime_end(self.sub_ret_ptr.unwrap(), "out_lifetime_end");

                self.stack().push(out_value);
            }
            DMIR::NewDatum(location) => {
                let stack_pos = self.stack_loc.len() - 1 - *location as usize;
                let t = self.stack_loc[stack_pos];
                let type_meta = self.emit_load_meta_value(t);

                let usr = func.get_nth_param(2).unwrap().into_struct_value();

                let create_datum = self.module.get_function("dmir.runtime.create_datum").unwrap();
                let result = self.builder.build_call(
                    create_datum,
                    &[
                        usr.into(),
                        type_meta.data.into(),
                        self.context.i32_type().const_int(0xFFFF as u64, false).into()
                    ], "create_datum").as_any_value_enum().into_int_value();
                let result_meta = MetaValue::with_tag(ValueTag::Datum, result.into(), self);
                let r = self.emit_store_meta_value(result_meta);
                self.stack_loc[stack_pos] = r;
                self.stack().push(r);
            }
            DMIR::PushInt(val) => {
                // val is int, so convert it to float as all values in byond are floats, then bit-cast to store it within DMValue
                let value = self.builder.build_bitcast(
                    self.context.f32_type().const_float((*val) as f64),
                    self.context.i32_type(),
                    "f32_to_i32"
                );
                let out_val = self.val_type.const_named_struct(
                    &[
                        self.const_tag(ValueTag::Number).into(),
                        value.into(),
                    ]
                );

                self.stack().push(out_val);

                // self.dbg("PushInt");
                // self.dbg_val(out_val);
            }
            DMIR::PushVal(op) => {

                let result = MetaValue::new(
                    self.context.i8_type().const_int(op.tag as u64, false),
                    self.context.i32_type().const_int(op.data as u64, false).into(),
                );

                let result_val = self.emit_store_meta_value(result);
                self.stack().push(result_val);
            }
            DMIR::Pop => {
                self.stack().pop();
            }
            // Return stack top from proc
            DMIR::Ret => {

                // self.dbg("Ret");
                let value = self.stack().pop();
                // self.dbg_val(value);

                let out = func.get_nth_param(0).unwrap().into_pointer_value();
                self.builder.build_store(out, value);
                self.block_ended = true;
                self.emit_epilogue(func);
                self.builder.build_return(None);
            }
            // Set indexed local to stack top
            DMIR::SetLocal(idx) => {
                let value = self.stack().pop();
                self.locals.insert(idx.clone(), value);
            }
            // Push indexed local value to stack
            DMIR::GetLocal(idx) => {
                let value = self.locals.get(&idx).unwrap().clone();
                self.stack().push(value);
            }
            DMIR::GetArg(idx) => {
                let arg = self.args[*idx as usize];
                self.stack().push(arg);
            }
            DMIR::SetArg(idx) => {
                let new_value = self.stack().pop();
                self.args[*idx as usize] = new_value;
            }
            DMIR::TestIsDMEntity => {
                let value = self.stack().pop();
                let is_dm_entity = self.module.get_function("dmir.runtime.is_dm_entity").unwrap();

                let result = self.builder.build_call(
                    is_dm_entity,
                    &[
                        value.into()
                    ], "is_dm_entity").as_any_value_enum().into_int_value();

                self.test_res = result;
            }
            DMIR::IsSubtypeOf => {
                let first = self.stack().pop();
                let second = self.stack().pop();
                let is_subtype_of = self.module.get_function("dmir.runtime.is_subtype_of").unwrap();

                let result = self.builder.build_call(
                    is_subtype_of,
                    &[
                        second.into(),
                        first.into()
                    ], "is_subtype_of").as_any_value_enum().into_int_value();

                let meta_value = self.emit_boolean_to_number(result);
                let result_value = self.emit_store_meta_value(meta_value);

                self.stack().push(result_value);
            }
            DMIR::Test => {
                let value = self.stack().pop();
                let res = self.emit_check_is_true(self.emit_load_meta_value(value));

                self.test_res = res;
            }
            DMIR::TestEqual => {
                let second = self.stack().pop();
                let first = self.stack().pop();

                let first_meta = self.emit_load_meta_value(first);
                let second_meta = self.emit_load_meta_value(second);

                let const_tag_num = self.const_tag(ValueTag::Number);

                let is_number = self.builder.build_int_compare(IntPredicate::EQ, first_meta.tag, const_tag_num, "is_number");

                let total_eq =
                    self.builder.build_and(
                        self.builder.build_int_compare(IntPredicate::EQ, first_meta.tag, second_meta.tag, "check_tag"),
                        self.builder.build_int_compare(
                            IntPredicate::EQ,
                            first_meta.data.into_int_value(),
                            second_meta.data.into_int_value(),
                            "check_data"
                        ),
                        "total_eq"
                    );

                let first_f32 = self.builder.build_bitcast(first_meta.data, self.context.f32_type(), "first_f32").into_float_value();
                let second_f32 = self.builder.build_bitcast(second_meta.data, self.context.f32_type(), "second_f32").into_float_value();

                let result = self.builder.build_select(
                    is_number,
                    self.builder.build_float_compare(FloatPredicate::UEQ, first_f32, second_f32, "compare_num"),
                    total_eq,
                    "select_compare"
                ).into_int_value();

                self.test_res = result;
            }
            DMIR::Not => {
                let value = self.stack().pop();
                let result = self.emit_check_is_true(self.emit_load_meta_value(value));
                let invert = self.builder.build_not(result, "not");
                let meta_value = self.emit_boolean_to_number(invert);
                let result_value = self.emit_store_meta_value(meta_value);

                self.stack().push(result_value);
            }
            DMIR::PushTestFlag => {
                let test_value_f32 = self.builder.build_unsigned_int_to_float(self.test_res, self.context.f32_type(), "test_flag_to_f32");
                let test_value_i32 = self.builder.build_bitcast(test_value_f32, self.context.i32_type(), "test_value_i32");
                let test_value = self.emit_store_meta_value(MetaValue::with_tag(ValueTag::Number, test_value_i32, self));
                self.stack().push(test_value)
            }
            DMIR::SetTestFlag(val) => {
                self.test_res = self.context.bool_type().const_int((*val) as u64, false);
            }
            DMIR::JZ(lbl) => {
                self.emit_conditional_jump(func, lbl, self.builder.build_not(self.test_res, "jz"))
            }
            DMIR::Dup => {
                let value = self.stack().pop();
                self.stack().push(value);
                self.stack().push(value);
            }
            DMIR::DupX1 => {
                let b = self.stack().pop();
                let a = self.stack().pop();

                self.stack().push(b);
                self.stack().push(a);
                self.stack().push(b);
            }
            DMIR::DupX2 => {
                let c = self.stack().pop();
                let b = self.stack().pop();
                let a = self.stack().pop();

                self.stack().push(c);
                self.stack().push(a);
                self.stack().push(b);
                self.stack().push(c);
            }
            DMIR::Swap => {
                let a = self.stack().pop();
                let b = self.stack().pop();
                self.stack().push(a);
                self.stack().push(b);
            }
            DMIR::SwapX1 => {
                let a = self.stack().pop();
                let b = self.stack().pop();
                let c = self.stack().pop();
                self.stack().push(b);
                self.stack().push(a);
                self.stack().push(c);
            }
            DMIR::TestInternal => {
                let arg_value = self.stack().pop();
                let arg = self.emit_load_meta_value(arg_value);

                self.internal_test_flag = Some(self.emit_check_is_true(arg));
            }
            DMIR::JZInternal(lbl) => {
                let if_arg_false = self.builder.build_not(self.internal_test_flag.unwrap(), "jz");
                self.emit_conditional_jump(func, lbl, if_arg_false)
            }
            DMIR::JNZInternal(lbl) => {
                self.emit_conditional_jump(func, lbl, self.internal_test_flag.unwrap())
            }
            DMIR::Jmp(lbl) => {
                let values = create_code_gen_values_ref!(self);
                let mut block_builder = BlockBuilder {
                    context: self.context,
                    builder: &self.builder,
                    val_type: &self.val_type,
                };
                let target = block_builder.emit_jump_target_block(&mut self.block_map, values, func, lbl);
                self.builder.build_unconditional_branch(target.block);
                self.block_ended = true;
            }
            DMIR::EnterBlock(lbl) => {
                assert!(self.block_ended, "no direct fallthrough allowed");
                let target = self.block_map.get(lbl).unwrap();

                if !self.block_ended {
                    self.builder.build_unconditional_branch(target.block);
                }
                self.block_ended = false;

                self.stack_loc.clear();
                self.iterator_stack.clear();
                self.locals.clear();
                self.args.clear();
                self.cache = None;

                self.builder.position_at_end(target.block);

                self.args.extend(
                    target.args.iter().map(|phi| phi.as_basic_value().into_struct_value())
                );
                self.locals.extend(
                    target.locals.iter().map(|(idx, phi)| (*idx, phi.as_basic_value().into_struct_value()))
                );
                self.stack_loc.extend(
                    target.stack.iter().map(|phi| phi.as_basic_value().into_struct_value())
                );
                self.iterator_stack.extend(
                    target.iterator_stack.iter()
                );
                self.active_iterator = target.active_iterator;
                self.allocated_iterator = target.allocated_iterator;

                if let Some(cache_phi) = &target.cache {
                    self.cache = Some(cache_phi.as_basic_value().into_struct_value());
                }

                self.test_res = target.test_res.as_ref().unwrap().as_basic_value().into_int_value();
                self.loop_iter_counter = target.loop_iter_counter.as_ref().unwrap().as_basic_value().into_int_value();
            }
            DMIR::End => {
                if !self.block_ended {
                    self.emit_epilogue(func);
                    let out = func.get_nth_param(0).unwrap().into_pointer_value();
                    self.builder.build_store(out, self.val_type.const_zero());
                    self.builder.build_return(None);
                }
                self.block_ended = true;
            }
            DMIR::Deopt(offset, inc_ref_count_locations) => {
                let deopt_id = self.proc_meta_builder.add_deopt_point(
                    *offset,
                    inc_ref_count_locations,
                    self.active_iterator.map_or(false, |_| true),
                    self.iterator_stack.len() as u32
                );

                let insert_stack_map = decl_intrinsic!(self "llvm.experimental.stackmap" (i64_type, i32_type, ...) -> void_type);

                let src_meta = self.emit_load_meta_value(func.get_nth_param(1).unwrap().into_struct_value());
                let cache_meta = self.emit_load_meta_value(self.cache.unwrap_or(self.val_type.const_zero()));

                let test_res_u8 = self.builder.build_int_z_extend(
                    self.test_res,
                    self.context.i8_type(),
                    "test_res_u8"
                );

                let mut args: Vec<BasicMetadataValueEnum> = Vec::new();
                args.append(
                    &mut vec![
                        self.context.i64_type().const_int(deopt_id.0, false).into(),
                        self.context.i32_type().const_int(0, false).into(),
                        // actual stack map:
                        func.get_nth_param(0).unwrap().into(),
                        src_meta.tag.into(),
                        src_meta.data.into(),
                        test_res_u8.into(),
                        cache_meta.tag.into(),
                        cache_meta.data.into(),
                        self.context.i32_type().const_int(self.parameter_count as u64, false).into(), // arg_count
                        func.get_nth_param(4).unwrap().into(), // caller_arg_count
                        func.get_nth_param(3).unwrap().into(), // args_pointer, for excess arguments
                    ]
                );
                args.push(
                    self.context.i32_type().const_int(self.args.len() as u64, false).into()
                );
                for arg_value in &self.args {

                    let arg_value_meta = self.emit_load_meta_value(arg_value.clone());
                    args.push(arg_value_meta.tag.into());
                    args.push(arg_value_meta.data.into());
                }

                args.push(
                    self.context.i32_type().const_int(self.stack_loc.len() as u64, false).into()
                );
                for stack_value in &self.stack_loc {
                    let stack_value_meta = self.emit_load_meta_value(stack_value.clone());
                    args.push(
                        stack_value_meta.tag.into()
                    );
                    args.push(
                        stack_value_meta.data.into()
                    );
                }
                args.push(
                    self.context.i32_type().const_int(self.local_count as u64, false).into()
                );
                for id in 0..self.local_count {
                    let local_value = if let Option::Some(value) = self.locals.get(&id) {
                        value.clone().into()
                    } else {
                        self.val_type.const_zero().into()
                    };

                    let local_value_meta = self.emit_load_meta_value(local_value);
                    args.push(
                        local_value_meta.tag.into()
                    );
                    args.push(
                        local_value_meta.data.into()
                    );
                }

                // iterators
                for iterator_ptr in self.active_iterator.iter().chain(self.iterator_stack.iter()) {
                    let iterator_value = self.builder.build_load(*iterator_ptr, "load_iterator").into_struct_value();
                    // iterator_type = 0, iterator_array = 1, iterator_allocated = 2, iterator_length = 3, iterator_index = 4, iterator_filter_flags = 5, iterator_filter_type = 6
                    let filter_type_value = self.builder.build_extract_value(iterator_value, 6, "iterator_filter_type").unwrap();
                    let filter_type_meta = self.emit_load_meta_value(filter_type_value.into_struct_value());
                    args.append(&mut vec![
                        self.builder.build_extract_value(iterator_value, 0, "iterator_type").unwrap().into(),
                        self.builder.build_extract_value(iterator_value, 1, "iterator_array").unwrap().into(),
                        self.builder.build_extract_value(iterator_value, 2, "iterator_allocated").unwrap().into(),
                        self.builder.build_extract_value(iterator_value, 3, "iterator_length").unwrap().into(),
                        self.builder.build_extract_value(iterator_value, 4, "iterator_index").unwrap().into(),
                        self.builder.build_extract_value(iterator_value, 5, "iterator_filter_flags").unwrap().into(),
                        filter_type_meta.tag.into(),
                        filter_type_meta.data.into()
                    ]);
                }


                self.builder.build_call(
                    insert_stack_map,
                    args.as_slice(),
                    "stack_map"
                );

                self.builder.build_call(
                    self.module.get_function("dmir.runtime.deopt").unwrap(),
                    &[
                        self.context.i64_type().const_int(deopt_id.0, false).into()
                    ],
                    "call_deopt",
                );

                self.builder.build_return(None);
                let post_deopt_block = self.context.append_basic_block(func, "post_deopt");
                self.builder.position_at_end(post_deopt_block);
            }
            DMIR::CheckTypeDeopt(stack_pos, predicate, deopt) => {
                let stack_value = self.stack_loc[self.stack_loc.len() - 1 - (stack_pos.clone() as usize)];
                let actual_tag = self.emit_load_meta_value(stack_value).tag;

                let next_block = self.context.append_basic_block(func, "next");
                let deopt_block = self.context.append_basic_block(func, "deopt");


                self.builder.build_conditional_branch(
                    self.emit_tag_predicate_comparison(actual_tag, predicate),
                    next_block,
                    deopt_block,
                );

                self.builder.position_at_end(deopt_block);
                if cfg!(debug_deopt_print) {
                    self.dbg(format!("CheckType({}, {:?}, {:?}) failed: ", stack_pos, predicate, deopt).as_str());
                    self.dbg_val(stack_value);
                }
                self.emit(deopt.borrow(), func);
                self.builder.build_unconditional_branch(next_block);

                self.builder.position_at_end(next_block);
            }
            DMIR::InfLoopCheckDeopt(deopt) => {
                let one_const = self.context.i32_type().const_int(1, false);

                let result_value = self.builder.build_int_sub(self.loop_iter_counter, one_const, "sub");

                let next_block = self.context.append_basic_block(func, "next");
                let deopt_block = self.context.append_basic_block(func, "deopt");

                self.loop_iter_counter = result_value;

                self.builder.build_conditional_branch(
                    self.builder.build_int_compare(IntPredicate::NE, result_value, self.context.i32_type().const_zero(), "ne"),
                    next_block,
                    deopt_block,
                );

                self.builder.position_at_end(deopt_block);
                if cfg!(debug_deopt_print) {
                    self.dbg(format!("InfLoopCheck({:?}) failed: ", deopt).as_str());
                }
                self.emit(deopt.borrow(), func);
                self.builder.build_unconditional_branch(next_block);

                self.builder.position_at_end(next_block);
            }
            DMIR::ListCheckSizeDeopt(list, index, deopt) =>{
                let list_struct = self.emit_read_value_location(list);
                let index_struct = self.emit_read_value_location(index);

                let index_meta = self.emit_load_meta_value(index_struct);
                let index_i32 = self.emit_meta_value_to_int(index_meta);

                let list_check_size = self.module.get_function("dmir.intrinsic.list_check_size").unwrap();

                let result = self.builder.build_call(
                    list_check_size,
                    &[
                        list_struct.into(),
                        index_i32.into()
                    ], "list_check_size").as_any_value_enum().into_int_value();

                let next_block = self.context.append_basic_block(func, "next");
                let deopt_block = self.context.append_basic_block(func, "deopt");

                self.builder.build_conditional_branch(
                    result,
                    next_block,
                    deopt_block,
                );

                self.builder.position_at_end(deopt_block);
                if cfg!(debug_deopt_print) {
                    self.dbg(format!("CheckListSize({:?}, {:?}, {:?}) failed: ", list, index, deopt).as_str());
                    self.dbg_val(list_struct);
                }
                self.emit(deopt.borrow(), func);
                self.builder.build_unconditional_branch(next_block);

                self.builder.position_at_end(next_block);
            }
            DMIR::IncRefCount { target, op } => {
                if self.block_ended {
                    return;
                }
                match target {
                    RefOpDisposition::DupPost(location) => {
                        let value = self.emit_read_value_location(location);
                        self.emit(op, func);
                        self.emit_inc_ref_count(value);
                    }
                    RefOpDisposition::Post(location) => {
                        self.emit(op, func);
                        let value = self.emit_read_value_location(location);
                        self.emit_inc_ref_count(value);
                    }
                    RefOpDisposition::Pre(location) => {
                        let value = self.emit_read_value_location(location);
                        self.emit_inc_ref_count(value);
                        self.emit(op, func);
                    }
                }
            }
            DMIR::DecRefCount { target, op } => {
                if self.block_ended {
                    return;
                }
                match target {
                    RefOpDisposition::DupPost(location) => {
                        let value = self.emit_read_value_location(location);
                        self.emit(op, func);
                        self.emit_dec_ref_count(value);
                    }
                    RefOpDisposition::Post(location) => {
                        self.emit(op, func);
                        let value = self.emit_read_value_location(location);
                        self.emit_dec_ref_count(value);
                    }
                    RefOpDisposition::Pre(location) => {
                        let value = self.emit_read_value_location(location);
                        self.emit_dec_ref_count(value);
                        self.emit(op, func);
                    }
                }
            }
            DMIR::UnsetCache => {
                self.cache = Option::None;
            }
            DMIR::UnsetLocal(idx) => {
                self.locals.remove(idx);
            }
            DMIR::IterAllocate => {
                let first_block = func.get_first_basic_block().unwrap();
                let current_block = self.builder.get_insert_block().unwrap();

                let iterator_struct_type = self.module.get_struct_type("DMIterator").unwrap();
                self.builder.position_before(&first_block.get_last_instruction().unwrap());
                let new_iterator = self.builder.build_alloca(iterator_struct_type, "iter");
                self.builder.position_at_end(current_block);
                assert!(self.allocated_iterator.is_none());
                self.allocated_iterator = Some(new_iterator);
            }
            DMIR::ArrayIterLoadFromList(type_filter) => {
                self.emit_iterator_constructor("dmir.intrinsic.iter.load_array_from_list", type_filter);
            }
            DMIR::ArrayIterLoadFromObject(type_filter) => {
                self.emit_iterator_constructor("dmir.intrinsic.iter.load_array_from_object", type_filter);
            }
            DMIR::IterNext => {
                let iterator = self.active_iterator.unwrap();
                let func = self.module.get_function("dmir.intrinsic.iter.next").unwrap();
                let result = self.builder.build_call(
                    func,
                    &[
                        iterator.into()
                    ],
                    "next"
                ).try_as_basic_value().left().unwrap().into_struct_value();

                let value = self.builder.build_extract_value(result, 0, "load_value").unwrap().into_struct_value();
                let test_res = self.builder.build_extract_value(result, 1, "load_test_res").unwrap().into_int_value();

                self.stack().push(value);

                // self.dbg("iter next");
                // self.dbg_val(value);
                self.test_res = test_res;
            }
            DMIR::IterPush => {
                self.iterator_stack.push(self.active_iterator.take().unwrap());
            }
            DMIR::IterPop => {
                if let Some(iterator) = self.active_iterator.take() {
                    self.emit_iterator_destructor(iterator);
                }
                self.active_iterator = self.iterator_stack.pop();
                assert!(self.active_iterator.is_some());
            }
            _ => {}
        }
    }
}
