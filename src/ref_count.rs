use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};

use typed_arena::Arena;

use crate::dmir::{DMIR, RefOpDisposition};
use crate::dmir::ValueLocation;
use crate::dmir_annotate::Annotator;
use crate::ref_count::RValue::Phi;
use crate::ref_count::RValueDrain::{ConsumeDrain, DeoptDrain, MoveOutDrain};

/// Denotes different types of value sources
enum RValue<'t> {
    ProduceSource(usize, IncRefOp), // on-demand ref count increment, example: getting value from variable, which already have non-zero ref count, and by this can be eliminated
    UncountedSource(usize), // produces value, that doesn't need to be counted, example: results of numeric operations
    MovedInSource(usize), // moves-in value with already incremented ref count, example: return result of other proc call
    PullInSource(usize, IncRefOp), // moves-in value and incrementing ref count forcefully, example: pulling in value from iterator
    Phi(u32, RefCell<Vec<&'t RValue<'t>>>) // combines values in results of branching
}

macro_rules! ref_identity {
    ($t:ty) => {
        impl<'a> PartialEq for &'a $t {
            fn eq(&self, other: &Self) -> bool {
                std::ptr::eq(*self, *other)
            }
        }
        impl<'a> Eq for &'a $t {}
        impl<'a> Hash for &'a $t {
            fn hash<H: Hasher>(&self, state: &mut H) {
                std::ptr::hash(*self, state);
            }
        }
    };
}

pub(crate) use ref_identity;

ref_identity!(RValue<'_>);

/// Denotes value drains
#[derive(Debug, Eq, Hash, PartialEq, Clone)]
enum RValueDrain<'t> {
    ConsumeDrain(usize, &'t RValue<'t>, DecRefOp), // decrement ref count on-demand, example: pop value off stack should dec ref count if value had incremented ref count before
    MoveOutDrain(usize, &'t RValue<'t>), // moves-out value to somewhere else, meaning value must have incremented ref count, example: return from proc
    DeoptDrain(usize, &'t RValue<'t>, IncRefOp), // moves-out value to deopt, fixing ref counts inplace
}

struct BasicBlockNodes<'t> {
    stack_phi: Vec<&'t RValue<'t>>,
    cache_phi: Option<&'t RValue<'t>>,
    locals_phi: HashMap<u32, &'t RValue<'t>>,
    args_phi: Vec<&'t RValue<'t>>
}

impl<'t> RValue<'t> {
    fn new_phi(phi_id: &mut u32, first_incoming: &'t RValue<'t>) -> RValue<'t> {
        let id = *phi_id;
        *phi_id += 1;
        Phi(id, RefCell::new(vec![first_incoming]))
    }
}

macro_rules! mk_value {
    ( $self:expr, $x:expr ) => (
        {
            let value = $self.values_arena.alloc($x);
            $self.values.push(value);
            value
        }
    )
}



#[derive(Clone, Eq, PartialEq, Debug)]
enum Decision {
    Keep,
    Remove,
    Undecided
}

fn rvalue_dfs<'t>(value: &'t RValue<'t>, visited: &mut HashSet<&'t RValue<'t>>) {
    if !visited.insert(value) {
        return;
    }
    match value {
        RValue::ProduceSource(_, _) => {}
        RValue::MovedInSource(_) => {}
        RValue::UncountedSource(_) => {}
        RValue::PullInSource(_, _) => {}
        Phi(_, incoming) => {
            for value in incoming.borrow().iter() {
                rvalue_dfs(*value, visited);
            }
        }
    }
}

struct Analyzer<'t> {
    values_arena: &'t Arena<RValue<'t>>,
    stack: Vec<&'t RValue<'t>>,
    cache: Option<&'t RValue<'t>>,
    drains: Vec<RValueDrain<'t>>,
    values: Vec<&'t RValue<'t>>,
    locals: HashMap<u32, &'t RValue<'t>>,
    args: Vec<&'t RValue<'t>>,
    blocks: HashMap<String, BasicBlockNodes<'t>>,
    phi_id: u32,
    block_ended: bool
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum IncRefOp {
    Post(ValueLocation),
    Pre(ValueLocation)
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum DecRefOp {
    Pre(ValueLocation), DupPost(ValueLocation)
}


impl<'t> Analyzer<'t> {

    fn merge_block(
        stack: &Vec<&'t RValue<'t>>,
        cache: &Option<&'t RValue<'t>>,
        locals: &HashMap<u32, &'t RValue<'t>>,
        args: &Vec<&'t RValue<'t>>,
        values_arena: &'t Arena<RValue<'t>>,
        phi_id: &mut u32,
        blocks: &mut HashMap<String, BasicBlockNodes<'t>>,
        lbl: String,
    ) {
        match blocks.entry(lbl) {
            Entry::Occupied(mut entry) => {
                let v = entry.get_mut();
                for (i, value) in stack.iter().enumerate() {
                    match v.stack_phi.get_mut(i).unwrap() {
                        Phi(_, incoming) => {
                            incoming.borrow_mut().push(value);
                        }
                        _ => {}
                    }
                }

                for (idx, value) in locals {
                    if let Phi(_, incoming) = v.locals_phi.get_mut(idx).unwrap() {
                        incoming.borrow_mut().push(value);
                    }
                }

                for (i, value) in args.iter().enumerate() {
                    if let Phi(_, incoming) = v.args_phi.get_mut(i).unwrap() {
                        incoming.borrow_mut().push(value);
                    }
                }


                if let Some(cache) = cache {
                    if let Phi(_, incoming) = v.cache_phi.unwrap() {
                        incoming.borrow_mut().push(cache);
                    }
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(
                    BasicBlockNodes {
                        stack_phi: stack.iter().map(|f| {
                            let node = RValue::new_phi(phi_id, f);
                            &*values_arena.alloc(node)
                        }).collect(),
                        cache_phi: cache.as_ref().map(|f|
                            &*values_arena.alloc(RValue::new_phi(phi_id, f))
                        ),
                        locals_phi: locals.iter().map(|(idx, v)| {
                            let node = RValue::new_phi(phi_id, v);
                            (idx.clone(), &*values_arena.alloc(node))
                        }).collect(),
                        args_phi: args.iter().map(|f| {
                            let node = RValue::new_phi(phi_id, f);
                            &*values_arena.alloc(node)
                        }).collect(),
                    }
                );
            }
        }
    }

    #[allow(unused_assignments, unused_variables)]
    fn analyze_instruction(&mut self, pos: usize, ir: &DMIR) {

        let mut stack_write_pos: u8 = 0;
        let mut stack_read_pos: u8 = 0;

        macro_rules! make_read {
            (@stack) => ({
                let value = self.stack.pop().unwrap();
                let location = ValueLocation::Stack(stack_read_pos);
                stack_read_pos += 1;
                (value, location)
            });
            (@cache) => ({
                let value = self.cache.unwrap();
                let location = ValueLocation::Cache;
                (value, location)
            });
        }

        macro_rules! make_write_location {
            (@stack) => ({
                let location = ValueLocation::Stack(stack_write_pos);
                stack_write_pos += 1;
                location
            });
            (@cache) => {
                ValueLocation::Cache
            };
            (@local $idx:expr) => {
                ValueLocation::Local($idx)
            };
        }

        macro_rules! make_write {
            ($value:expr, @stack) => ({
                self.stack.push(mk_value!(self, $value));
            });
        }

        macro_rules! single_effect {
            (@consume @$kind:ident $($idx:expr)?) => ({
                let (value, location) = make_read!(@$kind $($idx)?);
                self.drains.push(RValueDrain::ConsumeDrain(pos, value, DecRefOp::DupPost(location)));
            });
            (@move_out @$kind:ident $($idx:expr)?) => ({
                let (value, location) = make_read!(@$kind $($idx)?);
                self.drains.push(RValueDrain::MoveOutDrain(pos, value));
            });
            (@move_in @$kind:ident $($idx:expr)?) => ({
                make_write!(RValue::MovedInSource(pos), @$kind $($idx)?);
            });
            (@pull_in @$kind:ident $($idx:expr)?) => ({
                make_write!(RValue::PullInSource(pos, IncRefOp::Post(make_write_location!(@$kind $($idx)?))), @$kind $($idx)?);
            });
            (@produce_uncounted @$kind:ident $($idx:expr)?) => ({
                make_write!(RValue::UncountedSource(pos), @$kind $($idx)?);
            });
            (@produce @$kind:ident $($idx:expr)?) => ({
                make_write!(RValue::ProduceSource(pos, IncRefOp::Post(make_write_location!(@$kind $($idx)?))), @$kind $($idx)?);
            });
        }

        macro_rules! op_effect {
            ($(@$action:ident @$kind:ident $($idx:expr)?),+) => {
                $( single_effect!(@$action @$kind $($idx)?) );+
            };
        }

        macro_rules! unset_locals_and_cache {
            () => {
                if let Some(cache) = self.cache.as_ref() {
                    self.drains.push(ConsumeDrain(pos, cache, DecRefOp::Pre(ValueLocation::Cache)))
                }
                for (idx, value) in self.locals.iter() {
                    self.drains.push(ConsumeDrain(pos, *value, DecRefOp::Pre(ValueLocation::Local(*idx))))
                }
                for (idx, value) in self.args.iter().enumerate() {
                    self.drains.push(ConsumeDrain(pos, value, DecRefOp::Pre(ValueLocation::Argument(idx as u32))))
                }
            }
        }

        match ir {
            DMIR::Nop => {}
            DMIR::GetLocal(idx) => {
                op_effect!(
                    @produce @stack
                )
            }
            DMIR::SetLocal(idx) => {
                let value = self.stack.pop().unwrap();
                let old = self.locals.insert(idx.clone(), value);
                if let Some(old) = old {
                    self.drains.push(RValueDrain::ConsumeDrain(pos, old, DecRefOp::Pre(ValueLocation::Local(idx.clone()))))
                }
            }
            DMIR::GetSrc => {
                op_effect!(
                    @produce @stack
                );
            }
            DMIR::GetArg(_) => {
                op_effect!(
                    @produce @stack
                );
            }
            DMIR::SetArg(idx) => {
                let value = self.stack.pop().unwrap();
                let arg_value = &mut self.args[*idx as usize];
                self.drains.push(RValueDrain::ConsumeDrain(pos, arg_value, DecRefOp::Pre(ValueLocation::Argument(*idx))));
                *arg_value = value;
            }
            DMIR::SetCache => {
                let prev = self.cache.replace(self.stack.pop().unwrap());
                if let Some(prev) = prev {
                    self.drains.push(RValueDrain::ConsumeDrain(pos, prev, DecRefOp::Pre(ValueLocation::Cache)))
                }
            }
            DMIR::GetCacheField(_) => {
                op_effect!(
                    @produce @stack
                );
            }
            DMIR::SetCacheField(_) => {
                op_effect!(
                    @consume @stack
                );
            }
            DMIR::PushCache => {
                op_effect!(
                    @produce @stack
                );
            }
            DMIR::ValueTagSwitch(_, cases) => {
                for (_, block) in cases {
                    Analyzer::merge_block(
                        &self.stack,
                        &self.cache,
                        &self.locals,
                        &self.args,
                        &self.values_arena,
                        &mut self.phi_id,
                        &mut self.blocks, block.clone())
                }
                self.block_ended = true;
            }
            DMIR::FloatAdd | DMIR::FloatSub | DMIR::FloatMul | DMIR::FloatDiv | DMIR::RoundN | DMIR::BitAnd | DMIR::BitOr => {
                op_effect!(
                    @consume @stack,
                    @consume @stack,
                    @produce_uncounted @stack
                );
            }
            DMIR::FloatCmp(_) => {
                op_effect!(
                    @consume @stack,
                    @consume @stack,
                    @produce_uncounted @stack
                );
            }
            DMIR::IsSubtypeOf => {
                op_effect!(
                    @consume @stack,
                    @consume @stack,
                    @produce_uncounted @stack
                );
            }
            DMIR::FloatAbs => {
                op_effect!(
                    @consume @stack,
                    @produce_uncounted @stack
                );
            }
            DMIR::FloatInc | DMIR::FloatDec => {
                op_effect!(
                    @consume @stack,
                    @produce_uncounted @stack
                );
            }
            DMIR::PushInt(_) => {
                op_effect!(@produce_uncounted @stack);
            }
            DMIR::PushVal(_) => {
                op_effect!(@produce @stack);
            }
            DMIR::PushTestFlag => {
                op_effect!(@produce_uncounted @stack);
            }
            DMIR::SetTestFlag(_) => {}
            DMIR::Pop => {
                op_effect!(@consume @stack);
            }
            DMIR::ListCopy => {
                op_effect!(
                    @consume @stack,
                    @move_in @stack
                );
            }
            DMIR::ListAddSingle | DMIR::ListSubSingle => {
                op_effect!(
                    @consume @stack,
                    @consume @stack
                );
            }
            DMIR::ListAssociativeGet | DMIR::ListIndexedGet => {
                op_effect!(
                    @consume @stack,
                    @consume @stack,
                    @produce @stack
                );
            }
            DMIR::ListAssociativeSet | DMIR::ListIndexedSet => {
                op_effect!(
                    @consume @stack,
                    @consume @stack,
                    @move_out @stack
                );
            }
            DMIR::NewVectorList(count) => {
                for _ in 0..count.clone() {
                    op_effect!(@move_out @stack);
                }
                op_effect!(@move_in @stack);
            }
            DMIR::NewAssocList(count, deopt) => {
                let old_block_ended = self.block_ended;
                self.analyze_instruction(pos, deopt.borrow());
                self.block_ended = old_block_ended;
                for _ in 0..count.clone() {
                    op_effect!(@move_out @stack);
                    op_effect!(@consume @stack);
                }
                op_effect!(@move_in @stack);
            }
            DMIR::GetStep => {
                op_effect!(
                    @consume @stack,
                    @consume @stack,
                    @produce @stack
                );
            }
            DMIR::Ret => {
                unset_locals_and_cache!();
                op_effect!(@move_out @stack);
                self.block_ended = true;
            }
            DMIR::Test => {
                op_effect!(@consume @stack);
            }
            DMIR::JNZInternal(lbl) | DMIR::JZInternal(lbl) | DMIR::JZ(lbl) => {
                Analyzer::merge_block(
                    &self.stack,
                    &self.cache,
                    &self.locals,
                    &self.args,
                    &self.values_arena,
                    &mut self.phi_id,
                    &mut self.blocks,
                    lbl.to_string()
                )
            }
            DMIR::Dup => {
                let value = self.stack.pop().unwrap();
                self.stack.push(value);
                op_effect!(@produce @stack);
            }
            DMIR::Swap => {
                let a = self.stack.pop().unwrap();
                let b = self.stack.pop().unwrap();
                self.stack.push(a);
                self.stack.push(b);
            }
            DMIR::SwapX1 => {
                let a = self.stack.pop().unwrap();
                let b = self.stack.pop().unwrap();
                let c = self.stack.pop().unwrap();
                self.stack.push(b);
                self.stack.push(a);
                self.stack.push(c);
            }
            DMIR::TestInternal => {
                op_effect!(@consume @stack);
            }
            DMIR::EnterBlock(lbl) => {
                assert!(self.block_ended);
                let lbl_str = lbl.to_string();
                self.block_ended = false;
                let block = self.blocks.get(lbl).unwrap_or_else(|| panic!("{}: Block not found for {}", pos, lbl));
                self.stack = block.stack_phi.clone();
                self.locals = block.locals_phi.clone();
                self.cache = block.cache_phi;
                self.args = block.args_phi.clone();
            }
            DMIR::Deopt(_, _) => {
                if let Some(value) = self.cache {
                    self.drains.push(DeoptDrain(pos, value, IncRefOp::Pre(ValueLocation::Cache)))
                }
                for (value, stack_value) in self.stack.iter().rev().enumerate() {
                    self.drains.push(DeoptDrain(pos, stack_value, IncRefOp::Pre(ValueLocation::Stack(value as u8))))
                }
                for (value, local) in self.locals.iter() {
                    self.drains.push(DeoptDrain(pos, local, IncRefOp::Pre(ValueLocation::Local(value.clone()))));
                }
                self.block_ended = true;
            }
            DMIR::CheckTypeDeopt(_, _, deopt) | DMIR::ListCheckSizeDeopt(_, _, deopt) | DMIR::InfLoopCheckDeopt(deopt)=> {
                let old_block_ended = self.block_ended;
                self.analyze_instruction(pos, deopt.borrow());
                self.block_ended = old_block_ended;
            }
            DMIR::CallProcById(_, _, arg_count) | DMIR::CallProcByName(_, _, arg_count) => {
                op_effect!(@consume @stack);
                for _ in 0..arg_count.clone() {
                    op_effect!(@move_out @stack);
                }
                op_effect!(@move_in @stack);
            }
            DMIR::NewDatum(location) => {
                let stack_pos = self.stack.len() - 1 - *location as usize;
                let location_read = ValueLocation::Stack(*location as u8);
                self.drains.push(RValueDrain::ConsumeDrain(pos, self.stack[stack_pos], DecRefOp::DupPost(location_read)));
                self.stack[stack_pos] = mk_value!(self, RValue::MovedInSource(pos));
                op_effect!(@produce @stack);
            }
            DMIR::End => {
                unset_locals_and_cache!();
                self.block_ended = true;
            }
            DMIR::Not => {
                op_effect!(
                    @consume @stack,
                    @produce @stack
                );
            }
            DMIR::TestEqual => {
                op_effect!(
                    @consume @stack,
                    @consume @stack
                );
            }
            DMIR::TestIsDMEntity => {
                op_effect!(
                    @consume @stack
                );
            }
            DMIR::DupX1 => {
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();

                self.stack.push(b);
                self.stack.push(a);
                op_effect!(@produce @stack)
            }
            DMIR::DupX2 => {
                let c = self.stack.pop().unwrap();
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();

                self.stack.push(c);
                self.stack.push(a);
                self.stack.push(b);
                op_effect!(@produce @stack)
            }
            DMIR::Jmp(lbl) => {
                Analyzer::merge_block(
                    &self.stack,
                    &self.cache,
                    &self.locals,
                    &self.args,
                    self.values_arena.borrow(),
                    &mut self.phi_id,
                    &mut self.blocks,
                    lbl.to_string()
                );
                self.block_ended = true;
            }
            DMIR::UnsetLocal(idx) => {
                if let Some(local) = self.locals.remove(idx) {
                    self.drains.push(RValueDrain::ConsumeDrain(pos, local, DecRefOp::Pre(ValueLocation::Local(*idx))))
                } else {
                    panic!("No local found for idx: {} at pos: {}", idx, pos);
                }
            }
            DMIR::UnsetCache => {
                if let Some(cache) = self.cache {
                    self.drains.push(RValueDrain::ConsumeDrain(pos, cache, DecRefOp::Pre(ValueLocation::Cache)))
                }
                self.cache = Option::None;
            }
            DMIR::ArrayIterLoadFromList(_) => {
                op_effect!(
                    @consume @stack
                )
            }
            DMIR::ArrayIterLoadFromObject(_) => {
                op_effect!(
                    @consume @stack
                )
            }
            DMIR::IterNext => {
                op_effect!(
                    @pull_in @stack
                )
            }
            DMIR::IterPop | DMIR::IterPush => {}
            DMIR::IterAllocate => {}
            DMIR::IncRefCount { .. } => panic!(),
            DMIR::DecRefCount { .. } => panic!()
        }


    }

    fn analyze_instructions(&mut self, ir: &Vec<DMIR>) {
        for (pos, op) in ir.iter().enumerate() {
            self.analyze_instruction(pos, op)
        }
    }
}

impl<'t> Debug for RValue<'t> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        describe_source(self, f, &mut HashSet::new())
    }
}

fn describe_source(src: &RValue, fmt: &mut Formatter<'_>, phi_stack: &mut HashSet<u32>) -> std::fmt::Result {
    match src {
        RValue::ProduceSource(pos, _) => write!(fmt, "ProduceSource({})", pos),
        RValue::UncountedSource(pos) => write!(fmt, "UncountedSource({})", pos),
        RValue::MovedInSource(pos) => write!(fmt, "MovedInSource({})", pos),
        RValue::PullInSource(pos, _) => write!(fmt, "PullInSource({})", pos),
        Phi(id, sources) => {
            if phi_stack.insert(id.clone()) {
                write!(fmt, "Phi({}, [", id)?;
                for (idx, source) in sources.borrow().iter().enumerate() {
                    if idx != 0 {
                        write!(fmt, ", ")?;
                    }
                    describe_source(source, fmt, phi_stack)?
                }
                write!(fmt, "]")
            } else {
                write!(fmt, "Phi({}, ...)", id)
            }
        }
    }
}

fn create_inc_ref_count_ir(inner_instruction: DMIR, op: &IncRefOp) -> DMIR {
    return match op {
        IncRefOp::Post(loc) => {
            DMIR::IncRefCount {
                target: RefOpDisposition::Post(loc.clone()),
                op: Box::new(inner_instruction)
            }
        }
        IncRefOp::Pre(loc) => {
            DMIR::IncRefCount {
                target: RefOpDisposition::Pre(loc.clone()),
                op: Box::new(inner_instruction)
            }
        }
    }
}

pub fn generate_ref_count_operations(ir: &mut Vec<DMIR>, parameter_count: usize) {

    let arena = Arena::new();
    let mut analyzer = Analyzer {
        values_arena: &arena,
        stack: vec![],
        cache: None,
        drains: vec![],
        values: vec![],
        locals: Default::default(),
        args: vec![],
        blocks: Default::default(),
        phi_id: 0,
        block_ended: false
    };

    {
        let values = &mut analyzer.values;
        analyzer.args.resize_with(parameter_count, || {
            let value = arena.alloc(RValue::MovedInSource(0));
            values.push(value);
            value
        });
    }

    analyzer.analyze_instructions(ir);

    log::debug!("ref count analyzed");

    let mut sources_by_drain: HashMap<&RValueDrain, _> = HashMap::new();
    let mut decision_by_source: HashMap<&RValue, Decision> = HashMap::new();
    let mut decision_by_drain: HashMap<&RValueDrain, Decision> = HashMap::new();


    for drain in analyzer.drains.iter() {
        let mut sources = HashSet::new();
        match drain {
            ConsumeDrain(_, value, _) | MoveOutDrain(_, value) | DeoptDrain(_, value, _) => {
                rvalue_dfs(value, &mut sources)
            }
        }

        sources_by_drain.insert(drain, sources);
    }

    let mut pending = Vec::new();

    // Add initial constraints
    'outer: for (drain, sources) in &sources_by_drain {
        match drain {
            MoveOutDrain(_, _) => {
                for source in sources {
                    decision_by_source.insert(source, Decision::Keep);
                }
                decision_by_drain.insert(drain, Decision::Keep);
            }
            _ => {
                pending.push(*drain);
                for source in sources {
                    match source {
                        RValue::MovedInSource(_) | RValue::PullInSource(_, _) => {
                            decision_by_drain.insert(*drain, Decision::Keep);
                            decision_by_source.insert(*source, Decision::Keep);
                            continue'outer;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // iterate constraints until fixpoint
    let mut has_changes = false;
    loop {
        pending = pending.iter().filter_map(|drain| {
            let mut decision = decision_by_drain.get(drain).unwrap_or(&Decision::Undecided).clone();
            for source in sources_by_drain.get(drain).unwrap() {
                match decision_by_source.get(source).unwrap_or(&Decision::Undecided) {
                    Decision::Keep => {
                        decision = Decision::Keep;
                    }
                    _ => {}
                }
            }

            if decision != Decision::Undecided {
                has_changes = true;
                for source in sources_by_drain.get(drain).unwrap() {
                    decision_by_source.insert(source, decision.clone());
                }
                decision_by_drain.insert(drain, decision.clone());
                log::trace!("Fix {:?} into {:?}", drain, decision);
                None
            } else {
                Some(*drain)
            }
        }).collect();
        if !has_changes || pending.is_empty() {
            break
        }
        has_changes = false;
    }

    // pending now is only unreachable, let's remove them
    for drain in pending {
        let decision = Decision::Remove;
        for source in sources_by_drain.get(drain).unwrap() {
            decision_by_source.insert(source, decision.clone());
        }
        decision_by_drain.insert(drain, decision);
    }


    // build annotations
    let mut annotations = Annotator::new();

    for drain in &analyzer.drains {
        let (pos, source) = match drain {
            ConsumeDrain(pos, val, _) => (pos, format!("{:?}", val)),
            MoveOutDrain(pos, val) => (pos, format!("{:?}", val)),
            DeoptDrain(pos, val, _) => (pos, format!("{:?}", val))
        };

        let decision = decision_by_drain.get(drain);
        let annotation = format!("dec: {:?}", decision.unwrap_or(&Decision::Undecided));
        annotations.add(pos.clone(), annotation);
    }

    for value in &analyzer.values {
        let pos = match *value {
            RValue::ProduceSource(pos, _) => pos,
            RValue::MovedInSource(pos) => pos,
            RValue::UncountedSource(pos) => pos,
            RValue::PullInSource(pos, _) => pos,
            Phi(_, _) => continue,
        };

        let decision = decision_by_source.get(value);
        let annotation = format!("inc: {:?}", decision.unwrap_or(&Decision::Undecided));
        annotations.add(pos.clone(), annotation);
    }

    for drain in analyzer.drains.iter() {
        if decision_by_drain[drain] != Decision::Keep {
            continue;
        }
        match drain {
            ConsumeDrain(pos, _, op) => {
                let element = ir.get_mut(pos.clone()).unwrap();
                match element {
                    _ => {
                        let mut tmp = DMIR::End;
                        std::mem::swap(element, &mut tmp);
                        *element = match op {
                            DecRefOp::Pre(loc) => {
                                DMIR::DecRefCount {
                                    target: RefOpDisposition::Pre(loc.clone()),
                                    op: Box::new(tmp)
                                }
                            }
                            DecRefOp::DupPost(loc) => {
                                DMIR::DecRefCount {
                                    target: RefOpDisposition::DupPost(loc.clone()),
                                    op: Box::new(tmp)
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    for source in analyzer.values.iter() {
        if decision_by_source[source] != Decision::Keep {
            continue;
        }
        match &**source {
            RValue::ProduceSource(pos, op) | RValue::PullInSource(pos, op) => {
                let element = ir.get_mut(pos.clone()).unwrap();
                match element {
                    _ => {
                        let instruction = std::mem::replace(element, DMIR::Nop);
                        *element = create_inc_ref_count_ir(instruction, op);
                    }
                }
            }
            _ => {}
        }
    }


    for drain in analyzer.drains.iter() {
        if decision_by_drain[drain] != Decision::Remove {
            continue
        }
        match drain {
            DeoptDrain(pos, _, op) => {
                let location = match op {
                    IncRefOp::Pre(location) => location.clone(),
                    _ => panic!("Post inc ref not supported for deopt-s")
                };

                let instruction = ir.get_mut(pos.clone()).unwrap();
                match instruction {
                    DMIR::CheckTypeDeopt(_, _, deopt) | DMIR::ListCheckSizeDeopt(_, _, deopt) | DMIR::InfLoopCheckDeopt(deopt) | DMIR::NewAssocList(_, deopt)  => {
                        match deopt.borrow_mut() {
                            DMIR::Deopt(_, inc_ref_count_locations) => inc_ref_count_locations.push(location),
                            _ => panic!("not matching instruction type")
                        }
                    }
                    DMIR::Deopt(_, inc_ref_count_locations) => {
                        inc_ref_count_locations.push(location)
                    }
                    _ => panic!("not matching instruction type")
                }
            }
            _ => {}
        }
    }

    log::debug!("Generated:");
    annotations.dump_annotated(ir);
}
