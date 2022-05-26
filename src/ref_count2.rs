use std::cmp::max;
use std::collections::HashMap;
use std::env::var;
use itertools::Itertools;
use FlowVariableConsume::{Out, Unset};
use RefOpDisposition::{Post, Pre};
use crate::cfa::ControlFlowAnalyzer;
use crate::dfa::{analyze_and_dump_dfa, DataFlowAnalyzer, DFValueLocation, dump_dfa, FlowVariable, FlowVariableConsume, OperationEffect};
use crate::dmir::{DMIR, RefOpDisposition, ValueLocation};
use crate::dmir::RefOpDisposition::DupPost;
use crate::dmir::ValueLocation::{Argument, Cache, Local, Stack};
use crate::ref_count2::RefCountOp::{Dec, Inc};

pub fn generate_ref_count_operations2(
    ir: &mut Vec<DMIR>,
    parameter_count: usize,
) {

    let cfa = ControlFlowAnalyzer::new();
    let graph = cfa.analyze(ir);

    let mut analyzer = DataFlowAnalyzer::new(&graph);

    let data_flow_info = analyzer.analyze(
        ir,
        parameter_count as u32,
    );

    dump_dfa(ir, &data_flow_info);

    let mut operations_by_ref = HashMap::new();

    for (idx, instruction) in ir.iter().enumerate() {
        log::trace!("inx({}): {:?}", idx, instruction);
        let effect = analyze_instruction(instruction);
        let pre_stack_size = data_flow_info[idx].stack_size;
        let post_stack_size = data_flow_info[idx + 1].stack_size;

        let variables =
            data_flow_info[idx + 1].variables.iter().map(|var| (var.location.clone(), *var)).collect::<HashMap<_, _>>();

        let consumes =
            data_flow_info[idx + 1].consumes.iter().map(|consume| (consume.var().location.clone(), *consume)).collect::<HashMap<_, _>>();

        for ref_count_op in effect.operations.iter() {
            log::trace!("op: {:?}", ref_count_op);
            match ref_count_op {
                Inc(disposition) => {

                    let location = disposition.to_df_location(pre_stack_size, post_stack_size);

                    log::trace!("location is: {:?}", location);
                    operations_by_ref
                        .entry(FlowNodeRef::Variable(variables[&location]))
                        .or_insert(vec![]).push(ref_count_op.clone())
                }
                Dec(disposition) => {
                    let location = disposition.to_df_location(pre_stack_size, post_stack_size);

                    if let Some(consume) = consumes.get(&location) {
                        log::trace!("location is: {:?}", location);
                        operations_by_ref
                            .entry(FlowNodeRef::Consume(consume))
                            .or_insert(vec![]).push(ref_count_op.clone())
                    }
                }
            }
        }
    }


}

impl<'t> FlowVariableConsume<'t> {
    fn var(&self) -> &'t FlowVariable<'t> {
        match self {
            Unset(var) => *var,
            Out(var) => *var
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
enum FlowNodeRef<'t> {
    Variable(&'t FlowVariable<'t>),
    Consume(&'t FlowVariableConsume<'t>)
}

impl RefOpDisposition {
    fn to_df_location(&self, pre_stack_size: usize, post_stack_size: usize) -> DFValueLocation {
        match self {
            DupPost(location) => location.to_df_location(pre_stack_size),
            Post(location) => location.to_df_location(post_stack_size),
            Pre(location) => location.to_df_location(pre_stack_size)
        }
    }
}

impl ValueLocation {
    fn to_df_location(&self, stack_size: usize) -> DFValueLocation {
        match self {
            Stack(position) => DFValueLocation::Stack(stack_size as u8 - 1 - *position),
            Cache => DFValueLocation::Cache,
            Local(idx) => DFValueLocation::Local(*idx),
            Argument(idx) => DFValueLocation::Argument(*idx),
        }
    }
}


#[derive(Clone, Debug)]
enum RefCountOp {
    Inc(RefOpDisposition),
    Dec(RefOpDisposition),
}

struct RefCountEffect {
    operations: Vec<RefCountOp>,
}

fn analyze_instruction(instruction: &DMIR) -> RefCountEffect {
    macro_rules! effect {
        ($($op:expr),*) => {
            RefCountEffect {
                operations: vec![$($op),*]
            }
        };
    }

    match instruction {
        DMIR::GetLocal(_) => effect!(
            Inc(Post(Stack(0)))
        ),
        DMIR::SetLocal(local_idx) => effect!(
            Dec(Pre(Local(*local_idx)))
        ),
        DMIR::GetSrc => effect!(
            Inc(Post(Stack(0)))
        ),
        DMIR::GetArg(_) => effect!(
            Inc(Post(Stack(0)))
        ),
        DMIR::SetArg(arg_idx) => effect!(
            Dec(Pre(Argument(*arg_idx)))
        ),
        DMIR::SetCache => effect!(
            Dec(Pre(Cache))
        ),
        DMIR::GetCacheField(_) => effect!(
            Inc(Post(Stack(0)))
        ),
        DMIR::SetCacheField(_) => effect!(
            Dec(DupPost(Stack(0)))
        ),
        DMIR::PushCache => effect!(
            Inc(Post(Stack(0)))
        ),
        DMIR::ValueTagSwitch(_, _) => effect!(),
        DMIR::FloatAdd => effect!(),
        DMIR::FloatSub => effect!(),
        DMIR::FloatMul => effect!(),
        DMIR::FloatDiv => effect!(),
        DMIR::FloatCmp(_) => effect!(),
        DMIR::FloatAbs => effect!(),
        DMIR::FloatInc => effect!(),
        DMIR::FloatDec => effect!(),
        DMIR::BitAnd => effect!(),
        DMIR::BitOr => effect!(),
        DMIR::RoundN => effect!(),
        DMIR::ListCheckSizeDeopt(_, _, _) => effect!(),
        DMIR::ListCopy => effect!(
            Dec(DupPost(Stack(0)))
        ),
        DMIR::ListAddSingle | DMIR::ListSubSingle => effect!(
            Dec(DupPost(Stack(1))),
            Dec(DupPost(Stack(0)))
        ),
        DMIR::ListIndexedGet | DMIR::ListAssociativeGet => effect!(
            Dec(DupPost(Stack(1))),
            Dec(DupPost(Stack(0))),
            Inc(Post(Stack(0)))
        ),
        DMIR::ListIndexedSet | DMIR::ListAssociativeSet => effect!(
            Dec(DupPost(Stack(0))),
            Dec(DupPost(Stack(1)))
        ),
        DMIR::NewVectorList(_) => effect!(),
        DMIR::NewAssocList(_, _) => effect!(),
        DMIR::ArrayIterLoadFromList(_) => effect!(),
        DMIR::ArrayIterLoadFromObject(_) => effect!(),
        DMIR::IterAllocate => effect!(),
        DMIR::IterPop => effect!(),
        DMIR::IterPush => effect!(),
        DMIR::IterNext => effect!(
            Inc(Post(Stack(0)))
        ),
        DMIR::GetStep => effect!(),
        DMIR::PushInt(_) => effect!(),
        DMIR::PushVal(_) => effect!(),
        DMIR::PushTestFlag => effect!(),
        DMIR::SetTestFlag(_) => effect!(),
        DMIR::Pop => effect!(),
        DMIR::Ret => effect!(),
        DMIR::Not => effect!(),
        DMIR::Test => effect!(),
        DMIR::TestEqual => effect!(),
        DMIR::TestIsDMEntity => effect!(),
        DMIR::IsSubtypeOf => effect!(),
        DMIR::JZ(_) => effect!(),
        DMIR::Dup => effect!(),
        DMIR::DupX1 => effect!(),
        DMIR::DupX2 => effect!(),
        DMIR::Swap => effect!(),
        DMIR::SwapX1 => effect!(),
        DMIR::TestInternal => effect!(),
        DMIR::JZInternal(_) => effect!(),
        DMIR::JNZInternal(_) => effect!(),
        DMIR::EnterBlock(_) => effect!(),
        DMIR::Jmp(_) => effect!(),
        DMIR::InfLoopCheckDeopt(_) => effect!(),
        DMIR::Deopt(_, _) => effect!(),
        DMIR::CheckTypeDeopt(_, _, _) => effect!(),
        DMIR::CallProcById(_, _, _) => effect!(),
        DMIR::CallProcByName(_, _, _) => effect!(),
        DMIR::NewDatum(_) => effect!(),
        DMIR::IncRefCount { .. } => effect!(),
        DMIR::DecRefCount { .. } => effect!(),
        DMIR::Nop => effect!(),
        DMIR::UnsetLocal(local_idx) => effect!(
            Dec(Pre(Local(*local_idx)))
        ),
        DMIR::UnsetCache => effect!(
            Dec(Pre(Cache))
        ),
        DMIR::End => effect!()
    }
}