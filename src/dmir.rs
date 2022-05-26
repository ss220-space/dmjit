use std::borrow::Borrow;
use std::ffi::CString;

use auxtools::Proc;
use auxtools::raw_types::procs::ProcId;
use auxtools::raw_types::strings::StringId;
use auxtools::raw_types::values::ValueTag;
use dmasm::{DebugData, Instruction, Node};
use dmasm::operands::{Value, ValueOpRaw, Variable};
use inkwell::FloatPredicate;

use crate::dmir::DMIR::{CheckTypeDeopt, EnterBlock};

#[derive(Debug)]
pub enum DMIR {
    GetLocal(u32),
    SetLocal(u32),
    GetSrc,
    GetArg(u32),
    SetArg(u32),
    SetCache,
    GetCacheField(u32),
    SetCacheField(u32),
    PushCache,
    ValueTagSwitch(ValueLocation, Vec<(ValueTagPredicate, String)>),
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatDiv,
    FloatCmp(inkwell::FloatPredicate),
    FloatAbs,
    FloatInc,
    FloatDec,
    BitAnd,
    BitOr,
    RoundN,
    ListCheckSizeDeopt(ValueLocation, ValueLocation, Box<DMIR>),
    ListCopy,
    ListAddSingle,
    ListSubSingle,
    ListIndexedGet,
    ListIndexedSet,
    ListAssociativeGet,
    ListAssociativeSet,
    NewVectorList(u32),
    NewAssocList(u32, Box<DMIR>),
    ArrayIterLoadFromList(dmasm::list_operands::TypeFilter),
    ArrayIterLoadFromObject(dmasm::list_operands::TypeFilter),
    IterAllocate, // allocates new iterator and stores ref on iter stack, replacing last
    IterPop,
    IterPush,
    IterNext,
    GetStep,
    PushInt(i32),
    PushVal(dmasm::operands::ValueOpRaw),
    PushTestFlag, // Push test flag value as Number
    SetTestFlag(bool),
    Pop,
    Ret,
    Not,
    Test,
    TestEqual,
    TestIsDMEntity,
    IsSubtypeOf,
    JZ(String),
    Dup, // Duplicate last value on stack
    DupX1, // Duplicate top value and insert one slot back ..., a, b -> ..., b, a, b
    DupX2, // Duplicate top value and insert two slot back ..., a, b, c -> ..., c, a, b, c
    Swap, // Put value one slot back on stack top: ..., a, b -> ..., b, a
    SwapX1, // Put value two slot back on stack top: ..., a, b, c -> ..., b, c, a
    TestInternal,        // Perform Test and write internal_test_flag
    JZInternal(String),  // Jump based on internal_test_flag
    JNZInternal(String), // Jump based on internal_test_flag
    EnterBlock(String),
    Jmp(String),
    InfLoopCheckDeopt(Box<DMIR>),
    Deopt(u32, Vec<ValueLocation>),
    CheckTypeDeopt(u32, ValueTagPredicate, Box<DMIR>), // Doesn't consume stack value for now
    CallProcById(ProcId, u8, u32),
    CallProcByName(StringId, u8, u32),
    NewDatum(u32),
    IncRefCount { target: RefOpDisposition, op: Box<DMIR> },
    DecRefCount { target: RefOpDisposition, op: Box<DMIR> },
    Nop,
    UnsetLocal(u32),
    UnsetCache,
    End
}

#[derive(Clone, Debug)]
pub enum ValueTagPredicate {
    Any,
    None,
    Tag(ValueTag),
    Union(Vec<ValueTagPredicate>)
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum RefOpDisposition {
    DupPost(ValueLocation), // Read value before op, dup, execute op, execute ref count operation
    Post(ValueLocation),    // Read value after op, execute ref count operation
    Pre(ValueLocation)      // Read value before op, execute ref count operation, execute op
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ValueLocation {
    Stack(u8),
    Cache,
    Local(u32),
    Argument(u32)
}

macro_rules! type_switch {
    (@switch_counter $s:expr, @stack $n:expr, $(($($check:tt)+) => $body:expr),+) => ({
        let cases = vec![
            $((value_tag_pred!($($check)+), $body)),+
        ];
        let mut block = Vec::new();
        decode_switch(ValueLocation::Stack($n), $s, cases, &mut block);
        block
    });
}

macro_rules! value_tag_pred {
    (@any) => ({ ValueTagPredicate::Any });
    (@nothing) => ({ ValueTagPredicate::Nothing });
    ($tag:expr) => ({ ValueTagPredicate::Tag($tag) });
    (@union $($tag:expr),+) => ({ ValueTagPredicate::Union(vec![$(value_tag_pred!($tag)),+]) });
}

macro_rules! check_type_deopt {
    (@$slot:literal !is $pred:expr => $deopt:expr) => {
        DMIR::CheckTypeDeopt($slot, $pred, Box::new($deopt))
    };
}

fn get_string_id(str: &Vec<u8>) -> StringId {
    let mut id = auxtools::raw_types::strings::StringId(0);
    unsafe {
        // obtain id of string from BYOND
        auxtools::raw_types::funcs::get_string_id(&mut id, CString::from_vec_unchecked(str.clone()).as_ptr());
    }

    return id;
}

// GetVar cache = src; cache["oxygen"]
// ->
// GetVar src
// SetCache
// GetCacheField cache["oxygen"]

// decode Variable nesing in GetVar
fn decode_get_var(vr: &Variable, out: &mut Vec<DMIR>) {
    match vr {
        Variable::Src => out.push(DMIR::GetSrc),

        // Decode complex SetCache chains
        //  GetVar cache = src; cache["oxygen"]
        //  ->
        //  GetVar src
        //  SetCache
        //  GetCacheField cache["oxygen"]
        Variable::SetCache(first, second) => {
            decode_get_var(first.borrow(), out);
            out.push(DMIR::SetCache);
            decode_get_var(second.borrow(), out);
        }
        // Just cache["STR"]
        Variable::Field(str) => {
            // gen DMIR
            out.push(DMIR::GetCacheField(get_string_id(&(str.0)).0));
        }
        Variable::Local(idx) => out.push(DMIR::GetLocal(idx.clone())),
        Variable::Arg(idx) => out.push(DMIR::GetArg(idx.clone())),
        Variable::Null => out.push(DMIR::PushVal(ValueOpRaw { tag: 0x0, data: 0 })), // TODO: make special instruction to allow no ref-counting over that Null value
        _ => panic!("decode_get_var: Not supported {:?}", vr)
    }
}

// decode Variable nesing in SetVar
fn decode_set_var(vr: &Variable, out: &mut Vec<DMIR>) {
    match vr {
        Variable::Local(idx) => out.push(DMIR::SetLocal(idx.clone())),
        Variable::SetCache(first, second) => {
            decode_get_var(first.borrow(), out);
            out.push(DMIR::SetCache);
            decode_set_var(second.borrow(), out);
        }
        Variable::Field(str) => {
            out.push(DMIR::SetCacheField(get_string_id(&(str.0)).0))
        }
        Variable::Arg(idx) => out.push(DMIR::SetArg(idx.clone())),
        _ => panic!("decode_set_var: Not supported {:?}", vr)
    }
}


fn decode_call(vr: &Variable, arg_count: u32, out: &mut Vec<DMIR>) {
    match vr {
        Variable::SetCache(a, b) => {
            decode_call(a.borrow(), arg_count, out);
            out.push(DMIR::SetCache);
            decode_call(b.borrow(), arg_count, out);
        }
        Variable::StaticVerb(_) => panic!("Unsupported: {:?}", vr.clone()),
        Variable::DynamicVerb(_) => panic!("Unsupported: {:?}", vr.clone()),
        Variable::StaticProc(proc_id) => {
            let name_id = unsafe { &*Proc::find(&proc_id.path).unwrap().entry }.name;
            out.push(DMIR::PushCache);
            out.push(DMIR::CallProcByName(name_id, 2, arg_count))
        }
        Variable::DynamicProc(name) => {
            out.push(DMIR::PushCache);
            out.push(DMIR::CallProcByName(get_string_id(&(name.0)), 2, arg_count))
        }
        _ => decode_get_var(vr, out)
    }
}

fn decode_cmp(op: FloatPredicate, data: &DebugData, proc: &Proc, out: &mut Vec<DMIR>) {
    out.push(check_type_deopt!(
        @0 !is value_tag_pred!(@union ValueTag::Number, ValueTag::Null)
        => DMIR::Deopt(data.offset, vec![])
    ));
    out.push(check_type_deopt!(
        @1 !is value_tag_pred!(@union ValueTag::Number, ValueTag::Null)
        => DMIR::Deopt(data.offset, vec![])
    ));
    out.push(DMIR::FloatCmp(op));
}

fn gen_push_null(out: &mut Vec<DMIR>) {
    out.push(DMIR::PushVal(ValueOpRaw { tag: ValueTag::Null as u8, data: 0 }));
}

fn build_float_bin_op_deopt(action: DMIR, data: &DebugData, proc: &Proc, out: &mut Vec<DMIR>) {
    out.push(CheckTypeDeopt(0, ValueTagPredicate::Tag(ValueTag::Number), Box::new(DMIR::Deopt(data.offset, vec![]))));
    out.push(CheckTypeDeopt(1, ValueTagPredicate::Tag(ValueTag::Number), Box::new(DMIR::Deopt(data.offset, vec![]))));
    out.push(action);
}

fn decode_switch(value: ValueLocation, switch_id: &mut u32, cases: Vec<(ValueTagPredicate, Vec<DMIR>)>, out: &mut Vec<DMIR>) {
    let switch_exit = format!("switch_{}_exit", switch_id);
    let (predicates, blocks): (Vec<_>, Vec<_>) = cases.into_iter().unzip();
    out.push(
        DMIR::ValueTagSwitch(
            value,
            predicates.into_iter().enumerate().map(
                |(index, predicate)| (predicate, format!("switch_{}_case_{}", switch_id, index))
            ).collect()
        )
    );
    let mut case_counter = 0;
    for mut instructions in blocks {
        out.push(EnterBlock(format!("switch_{}_case_{}", switch_id, case_counter)));
        out.append(&mut instructions);
        if !matches!(out.last(), Option::Some(DMIR::End)) {
            out.push(DMIR::Jmp(switch_exit.clone()));
        }
        case_counter += 1;
    }
    out.push(EnterBlock(switch_exit));
    *switch_id += 1;
}

pub fn decode_byond_bytecode(nodes: Vec<Node<DebugData>>, proc: Proc) -> Result<Vec<DMIR>, ()> {
    // output for intermediate operations sequence
    let mut irs = vec![];

    // will set to false if some unsupported operation found
    let mut supported = true;

    // needed for generating fallthrough jumps on EnterBlock
    let mut block_ended = false;

    let mut switch_counter = 0;

    macro_rules! build_type_switch {
        (@stack $n:expr, $(($($check:tt)+) => $body:expr),+) => ({
            type_switch!(@switch_counter &mut switch_counter, @stack $n, $(($($check)+) => $body),+)
        });
    }

    // generate DMIR sequence for each instruction in dm-asm
    for nd in nodes {
        match nd {
            // if node contains instruction
            dmasm::Node::Instruction(insn, data) => {
                macro_rules! deopt {
                    () => { Box::new(DMIR::Deopt(data.offset, vec![])) };
                    (@type_switch) => ( vec![DMIR::Deopt(data.offset, vec![]), DMIR::End] );
                }
                match insn {
                    // skip debug info for now
                    Instruction::DbgFile(_f) => {}
                    Instruction::DbgLine(_ln) => {},
                    Instruction::GetVar(vr) => {
                        decode_get_var(&vr, &mut irs)
                    }
                    Instruction::SetVar(vr) => {
                        decode_set_var(&vr, &mut irs)
                    }
                    Instruction::Add => {
                        irs.append(&mut build_type_switch!(
                                @stack 1,
                                (ValueTag::Number) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null) => vec![DMIR::FloatAdd],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (ValueTag::List) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null, ValueTag::Datum, ValueTag::Turf, ValueTag::Obj, ValueTag::Mob, ValueTag::Area, ValueTag::Client, ValueTag::String) =>
                                        vec![DMIR::Swap, DMIR::ListCopy, DMIR::DupX1, DMIR::Swap, DMIR::ListAddSingle],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (@any) => deopt!(@type_switch)
                            )
                        );
                    }
                    Instruction::Sub => {
                        irs.append(&mut build_type_switch!(
                                @stack 1,
                                (ValueTag::Number) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null) => vec![DMIR::FloatSub],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (ValueTag::List) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null, ValueTag::Datum, ValueTag::Turf, ValueTag::Obj, ValueTag::Mob, ValueTag::Area, ValueTag::Client, ValueTag::String) =>
                                        vec![DMIR::Swap, DMIR::ListCopy, DMIR::DupX1, DMIR::Swap, DMIR::ListSubSingle],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (@any) => deopt!(@type_switch)
                            )
                        );
                    }
                    Instruction::Band => {
                        irs.append(&mut build_type_switch!(
                                @stack 1,
                                (ValueTag::Null) => vec![DMIR::Pop, DMIR::Pop, DMIR::PushInt(0)],
                                (ValueTag::Number) => build_type_switch!(
                                    @stack 0,
                                    (ValueTag::Null) => vec![DMIR::Pop, DMIR::Pop, DMIR::PushInt(0)],
                                    (ValueTag::Number) => vec![DMIR::BitAnd],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (@any) => deopt!(@type_switch)
                            )
                        );
                    }
                    Instruction::Bor => {
                        irs.append(&mut build_type_switch!(
                                @stack 1,
                                (ValueTag::Null) => vec![DMIR::Swap, DMIR::Pop],
                                (ValueTag::Number) => build_type_switch!(
                                    @stack 0,
                                    (ValueTag::Null) => vec![DMIR::Pop],
                                    (ValueTag::Number) => vec![DMIR::BitOr],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (@any) => deopt!(@type_switch)
                            )
                        );
                    }
                    Instruction::Mul => {
                        irs.push(CheckTypeDeopt(
                            1,
                            value_tag_pred!(ValueTag::Number),
                            deopt!()
                        ));
                        irs.push(DMIR::FloatMul);
                    }
                    Instruction::Div => {
                        build_float_bin_op_deopt(DMIR::FloatDiv, &data, &proc, &mut irs);
                    }
                    Instruction::RoundN => {
                        irs.push(DMIR::RoundN);
                    }
                    Instruction::Tg => {
                        decode_cmp(FloatPredicate::UGT, &data, &proc, &mut irs);
                    }
                    Instruction::Tl => {
                        decode_cmp(FloatPredicate::ULT, &data, &proc, &mut irs);
                    }
                    Instruction::Tge => {
                        decode_cmp(FloatPredicate::UGE, &data, &proc, &mut irs);
                    }
                    Instruction::Tle => {
                        decode_cmp(FloatPredicate::ULE, &data, &proc, &mut irs);
                    }
                    Instruction::Teq => {
                        irs.push(DMIR::DupX1);
                        irs.push(DMIR::Swap);
                        irs.push(DMIR::TestEqual);
                    }
                    Instruction::Not => {
                        irs.push(DMIR::Not)
                    }
                    Instruction::Abs => {
                        irs.push(DMIR::FloatAbs)
                    }
                    Instruction::ListGet => {
                        irs.push(CheckTypeDeopt(
                            1,
                            value_tag_pred!(ValueTag::List),
                            deopt!()
                        ));
                        irs.append(
                            &mut build_type_switch!(
                                @stack 0,
                                (ValueTag::Number) => vec![DMIR::ListCheckSizeDeopt(ValueLocation::Stack(1), ValueLocation::Stack(0), deopt!()), DMIR::ListIndexedGet],
                                (@any) => vec![DMIR::ListAssociativeGet]
                            )
                        );
                    }
                    Instruction::ListSet => {
                        irs.push(CheckTypeDeopt(
                            1,
                            value_tag_pred!(ValueTag::List),
                            deopt!()
                        ));
                        irs.append(
                            &mut build_type_switch!(
                                @stack 0,
                                (ValueTag::Number) => vec![DMIR::ListCheckSizeDeopt(ValueLocation::Stack(1), ValueLocation::Stack(0), deopt!()), DMIR::ListIndexedSet],
                                (@any) => vec![DMIR::ListAssociativeSet]
                            )
                        );
                    }
                    Instruction::NewList(count) => {
                        irs.push(DMIR::NewVectorList(count));
                    }
                    Instruction::NewAssocList(count) => {
                        irs.push(DMIR::NewAssocList(count, deopt!()));
                    }
                    Instruction::GetStep => {
                        irs.push(DMIR::GetStep);
                    }
                    Instruction::CallGlob(arg_count, callee) => {
                        match callee.path.as_ref() {
                            "/dm_jitaux_deopt" => {
                                irs.push(DMIR::Deopt(data.offset, vec![]));
                                gen_push_null(&mut irs);
                            }
                            "/dmjit_is_optimized" => {
                                irs.push(DMIR::PushInt(1))
                            }
                            _ => {
                                gen_push_null(&mut irs);
                                let id = Proc::find(callee.path).unwrap().id;
                                irs.push(DMIR::CallProcById(id, 2, arg_count))
                            }
                        }
                    }
                    Instruction::Call(var, arg_count) => {
                        decode_call(&var, arg_count, &mut irs);
                    }
                    Instruction::CallStatement(var, arg_count) => {
                        decode_call(&var, arg_count, &mut irs);
                    }
                    Instruction::Ret => {
                        irs.push(DMIR::Ret);
                        block_ended = true;
                    }
                    Instruction::End => {
                        irs.push(DMIR::End);
                        block_ended = true;
                    }
                    Instruction::Test => {
                        irs.push(DMIR::Test)
                    }
                    Instruction::Jz(lbl) => {
                        irs.push(DMIR::JZ(lbl.0))
                    }
                    Instruction::Jmp(lbl) => {
                        irs.push(DMIR::Jmp(lbl.0));
                        block_ended = true;
                    }
                    Instruction::JzLoop(lbl) => {
                        irs.push(DMIR::InfLoopCheckDeopt(deopt!()));
                        irs.push(DMIR::JZ(lbl.0));
                    }
                    Instruction::JmpLoop(lbl) => {
                        irs.push(DMIR::InfLoopCheckDeopt(deopt!()));
                        irs.push(DMIR::Jmp(lbl.0));
                        block_ended = true;
                    }
                    Instruction::JmpAnd(lbl) => {
                        irs.push(DMIR::Dup);
                        irs.push(DMIR::TestInternal);
                        irs.push(DMIR::JZInternal(lbl.0));
                        irs.push(DMIR::Pop);
                    }
                    Instruction::JmpOr(lbl) => {
                        irs.push(DMIR::Dup);
                        irs.push(DMIR::TestInternal);
                        irs.push(DMIR::JNZInternal(lbl.0));
                        irs.push(DMIR::Pop);
                    }
                    Instruction::PushInt(i32) => {
                        irs.push(DMIR::PushInt(i32))
                    }
                    Instruction::PushVal(op) => {
                        match op.value {
                            Value::Number(value) => {
                                irs.push(DMIR::PushVal(ValueOpRaw {
                                    tag: ValueTag::Number as u8,
                                    data: unsafe { std::mem::transmute(value) }
                                }))
                            }
                            _ => {
                                irs.push(DMIR::PushVal(op.raw.unwrap()))
                            }
                        }

                    }
                    Instruction::GetFlag => {
                        irs.push(DMIR::PushTestFlag);
                    }
                    Instruction::Pop => {
                        irs.push(DMIR::Pop)
                    }
                    Instruction::AugAdd(var) => {
                        decode_get_var(&var, &mut irs);
                        irs.push(DMIR::Swap);
                        let mut number_branch = vec![DMIR::FloatAdd];
                        decode_set_var(&var, &mut number_branch);
                        irs.append(&mut build_type_switch!(
                                @stack 1,
                                (ValueTag::Number) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null) => number_branch,
                                    (@any) => deopt!(@type_switch)
                                ),
                                (ValueTag::List) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null, ValueTag::Datum, ValueTag::Turf, ValueTag::Obj, ValueTag::Mob, ValueTag::Area, ValueTag::Client, ValueTag::String) =>
                                        vec![DMIR::ListAddSingle],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (@any) => deopt!(@type_switch)
                            )
                        );
                    }
                    Instruction::AugSub(var) => {
                        decode_get_var(&var, &mut irs);
                        irs.push(DMIR::Swap);
                        let mut number_branch = vec![DMIR::FloatSub];
                        decode_set_var(&var, &mut number_branch);
                        irs.append(&mut build_type_switch!(
                                @stack 1,
                                (ValueTag::Number) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null) => number_branch,
                                    (@any) => deopt!(@type_switch)
                                ),
                                (ValueTag::List) => build_type_switch!(
                                    @stack 0,
                                    (@union ValueTag::Number, ValueTag::Null, ValueTag::Datum, ValueTag::Turf, ValueTag::Obj, ValueTag::Mob, ValueTag::Area, ValueTag::Client, ValueTag::String) =>
                                        vec![DMIR::ListSubSingle],
                                    (@any) => deopt!(@type_switch)
                                ),
                                (@any) => deopt!(@type_switch)
                            )
                        );
                    }
                    Instruction::Inc(var) => {
                        decode_get_var(&var, &mut irs);
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Datum) => deopt!(@type_switch),
                            (@any) => vec![DMIR::FloatInc]
                        ));
                        decode_set_var(&var, &mut irs);
                    }
                    Instruction::Dec(var) => {
                        decode_get_var(&var, &mut irs);
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Datum) => deopt!(@type_switch),
                            (@any) => vec![DMIR::FloatDec]
                        ));
                        decode_set_var(&var, &mut irs);
                    }
                    Instruction::IsNull => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Null) => vec![DMIR::Pop, DMIR::PushInt(1)],
                            (@any) => vec![DMIR::Pop, DMIR::PushInt(0)]
                        ));
                    }
                    Instruction::IsNum => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Number) => vec![DMIR::Pop, DMIR::PushInt(1)],
                            (@any) => vec![DMIR::Pop, DMIR::PushInt(0)]
                        ));
                    }
                    Instruction::IsText => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::String) => vec![DMIR::Pop, DMIR::PushInt(1)],
                            (@any) => vec![DMIR::Pop, DMIR::PushInt(0)]
                        ));
                    }
                    Instruction::IsTurf => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Turf) => vec![DMIR::TestIsDMEntity],
                            (@any) => vec![DMIR::Pop, DMIR::SetTestFlag(false)]
                        ));
                    }
                    Instruction::IsObj => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Obj) => vec![DMIR::TestIsDMEntity],
                            (@any) => vec![DMIR::Pop, DMIR::SetTestFlag(false)]
                        ));
                    }
                    Instruction::IsMob => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Mob) => vec![DMIR::TestIsDMEntity],
                            (@any) => vec![DMIR::Pop, DMIR::SetTestFlag(false)]
                        ));
                    }
                    Instruction::IsArea => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (ValueTag::Area) => vec![DMIR::TestIsDMEntity],
                            (@any) => vec![DMIR::Pop, DMIR::SetTestFlag(false)]
                        ));
                    }
                    Instruction::IsLoc => {
                        irs.push(DMIR::TestIsDMEntity);
                    }
                    Instruction::IsMovable => {
                        irs.append(&mut build_type_switch!(
                            @stack 0,
                            (@union ValueTag::Mob, ValueTag::Obj) => vec![DMIR::TestIsDMEntity],
                            (@any) => vec![DMIR::Pop, DMIR::SetTestFlag(false)]
                        ));
                    }
                    Instruction::IsType => {
                        irs.append(&mut build_type_switch!(
                            @stack 1,
                            (@union ValueTag::Turf, ValueTag::Obj, ValueTag::Mob, ValueTag::Area, ValueTag::Client, ValueTag::Image, ValueTag::List, ValueTag::Datum) =>
                                vec![DMIR::IsSubtypeOf],
                            (@any) => deopt!(@type_switch)
                        ));
                    }
                    Instruction::IsSubPath => {
                        irs.append(&mut build_type_switch!(
                            @stack 1,
                            (@union ValueTag::MobTypepath, ValueTag::ObjTypepath, ValueTag::TurfTypepath, ValueTag::AreaTypepath, ValueTag::DatumTypepath, ValueTag::SaveFileTypepath, ValueTag::ListTypepath, ValueTag::ClientTypepath, ValueTag::ImageTypepath) =>
                                vec![DMIR::IsSubtypeOf],
                            (@any) => vec![DMIR::Pop, DMIR::PushInt(0)]
                        ));
                    }
                    Instruction::Check2Numbers => {
                        irs.push(CheckTypeDeopt(
                            0,
                            value_tag_pred!(ValueTag::Number),
                            deopt!(),
                        ));
                        irs.push(CheckTypeDeopt(
                            1,
                            value_tag_pred!(ValueTag::Number),
                            deopt!(),
                        ));
                    }
                    Instruction::Check3Numbers => {
                        irs.push(CheckTypeDeopt(
                            0,
                            value_tag_pred!(ValueTag::Number),
                            deopt!(),
                        ));
                        irs.push(CheckTypeDeopt(
                            1,
                            value_tag_pred!(ValueTag::Number),
                            deopt!(),
                        ));
                        irs.push(CheckTypeDeopt(
                            2,
                            value_tag_pred!(ValueTag::Number),
                            deopt!(),
                        ));
                    }
                    Instruction::PopN(count) => {
                        for _i in 0..count {
                            irs.push(DMIR::Pop)
                        }
                    }
                    Instruction::ForRange(lab, var) => {
                        // a - counter, b - upper bound
                        // stack ... a b
                        irs.push(DMIR::Swap); // b a
                        irs.push(DMIR::DupX1); // a b a
                        irs.push(DMIR::Swap); // a a b
                        irs.push(DMIR::DupX1); // a b a b
                        irs.push(DMIR::FloatCmp(FloatPredicate::ULE)); // a b r
                        irs.push(DMIR::TestInternal); // a b
                        irs.push(DMIR::JZInternal(lab.0)); // a b
                        irs.push(DMIR::Swap); // b a
                        irs.push(DMIR::Dup); // b a a
                        decode_set_var(&var, &mut irs); // b a
                        irs.push(DMIR::FloatInc); //b (a+1)
                        irs.push(DMIR::Swap); // (a+1) b
                    }
                    Instruction::ForRangeStep(lab, var) => {
                        // a - counter, b - upper bound, c - step
                        // stack ... a b c
                        irs.push(DMIR::SwapX1); // b c a
                        irs.push(DMIR::DupX2); // a b c a
                        irs.push(DMIR::SwapX1); // a c a b
                        irs.push(DMIR::DupX2); // a b c a b
                        irs.push(DMIR::FloatCmp(FloatPredicate::ULE)); // a b c r
                        irs.push(DMIR::TestInternal); // a b c
                        irs.push(DMIR::JZInternal(lab.0)); // a b c
                        irs.push(DMIR::SwapX1); // b c a
                        irs.push(DMIR::Dup); // b c a a
                        decode_set_var(&var, &mut irs); // b c a
                        irs.push(DMIR::Swap); // b a c
                        irs.push(DMIR::DupX2); // c b a c
                        irs.push(DMIR::FloatAdd); // c b (a+c)
                        irs.push(DMIR::Swap); // c (a+c) b
                        irs.push(DMIR::SwapX1); // (a+c) b c
                    }
                    Instruction::New(arg_count) => {
                        irs.append(&mut build_type_switch!(
                            @stack arg_count as u8,
                            (ValueTag::DatumTypepath) => vec![DMIR::NewDatum(arg_count), DMIR::CallProcByName(StringId(3), 6, arg_count), DMIR::Pop],
                            (@any) => deopt!(@type_switch)
                        ));
                    }
                    Instruction::IterLoad(kind, bitmask) => {
                        irs.push(DMIR::IterAllocate);
                        match kind {
                            5 => {
                                if bitmask.contains(dmasm::list_operands::TypeFilter::DATUM_INSTANCES) {
                                    log::info!("Unsupported iter type {}", insn);
                                    supported = false;
                                }
                                irs.append(
                                    &mut build_type_switch!(
                                        @stack 0,
                                        (@union ValueTag::Turf, ValueTag::Obj, ValueTag::Mob, ValueTag::Area) => vec![DMIR::ArrayIterLoadFromObject(bitmask)],
                                        (ValueTag::List) => vec![DMIR::ListCopy, DMIR::ArrayIterLoadFromList(bitmask)],
                                        (@any) => deopt!(@type_switch)
                                    )
                                )
                            }
                            _ => {
                                log::info!("Unsupported iter type {}", insn);
                                supported = false;
                            }
                        }
                    }
                    Instruction::IterNext => irs.push(DMIR::IterNext),
                    Instruction::IterPop => irs.push(DMIR::IterPop),
                    Instruction::IterPush => irs.push(DMIR::IterPush),
                    _ => {
                        log::info!("Unsupported insn {}", insn);
                        supported = false;
                    }
                }
            }
            dmasm::Node::Label(lbl) => {
                //log::info!("{}:", lbl)
                if !block_ended {
                    irs.push(DMIR::Jmp(lbl.clone()));
                }
                block_ended = false;
                irs.push(DMIR::EnterBlock(lbl))
            }
            _ => {}
        }
    }

    if !supported {
        return Err(());
    }


    return Ok(irs);
}
