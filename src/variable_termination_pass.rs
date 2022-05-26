/*!
Variable termination pass is responsible for injection of variable termination operations into IR

Consider following example:
 ```
 /proc/conditional_var_set(c)
     var/w
     if (c)
         var/q = c
         w = q
     else
         w = c
     return w
 ```
 ```
Entry:
  GetVar arg(0) // c
  Test // c
  Jz LAB_0014 // if (c)
  GetVar arg(0) // c
  SetVar local(1) // var/q = c
  GetVar local(1) // q
  SetVar local(0) // w = q
  Jmp LAB_001A
LAB_0014:
  GetVar arg(0) // c
  SetVar local(0) // w = c
LAB_001A:
  GetVar local(0) // w
  Ret // return w
  End
 ```
Here, variable local(1) 'c' will not exist in block `LAB_001A`, as not all preceding blocks (`Entry`, `LAB_0014`) has set value for it

By that we'll have to inject `UnsetLocal(1)` instruction in-between of `Jmp LAB_001A` and corresponding block `LAB_001A` to make sure ref counts applied in-place

Same for special variable `cache`
*/
use std::borrow::{BorrowMut};
use std::cell::RefCell;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use typed_arena::Arena;
use crate::dmir::{DMIR, ValueLocation};
use crate::dmir_annotate::Annotator;

pub fn variable_termination_pass(ir: &mut Vec<DMIR>) {
    let arena = Arena::new();
    let mut analyzer = AnalyzerState::new(&arena);
    analyzer.analyze(ir);

    let mut annotations = Annotator::new();

    for (pos, instruction) in ir.iter().enumerate() {
        if let DMIR::EnterBlock(label ) = instruction {
            let block = &analyzer.blocks.get(label.as_str()).unwrap();
            annotations.add(pos, format!("incoming = {:?}", block.incoming_instructions));
            for (location, phi) in block.value_phi.iter() {
                annotations.add(pos, format!("{:?} = {:?}", location, phi));
            }
        }
    }
    log::debug!("value presence on blocks");


    let mut phi_solve_state: HashMap<usize, PhiDecision> = HashMap::new();
    let mut phi_queue = analyzer.all_phi.clone();
    let mut has_changes = false;

    // iterate till fix-point
    loop {
        phi_queue = phi_queue.iter().filter_map(|phi| {
            let mut decision = PhiDecision::Present;
            for incoming in phi.incoming.borrow().iter() {
                let incoming_decision = match incoming {
                    ValuePresence::Present(_, ValueSource::Source(_)) => {
                        PhiDecision::Present
                    }
                    ValuePresence::Present(_, ValueSource::Phi(other_phi)) => {
                        let other_phi_decision = phi_solve_state.get(&other_phi.id).unwrap_or(&PhiDecision::Undecided);
                        other_phi_decision.clone()
                    }
                    ValuePresence::Absent(_) => {
                        PhiDecision::Absent
                    }
                };
                decision = max(decision, incoming_decision);
            }
            phi_solve_state.insert(phi.id, decision.clone());
            if &decision == &PhiDecision::Undecided {
                Option::Some(*phi)
            } else {
                has_changes = true;
                Option::None
            }
        }).collect();
        if !has_changes {
            break
        }
        has_changes = false;
    }

    // Consider all remaining phi nodes as present
    // As Phi node can only exist if there is some source, if absence of value no proven for each of phi in cycle,
    // whole cycle should be considered present
    for phi in phi_queue {
        phi_solve_state.insert(phi.id, PhiDecision::Present);
    }

    // log::debug!("phi solve: {:?}", phi_solve_state);

    let mut values_to_unset: HashMap<usize, Vec<ValueLocation>> = HashMap::new();

    for block in analyzer.blocks.values() {
        for (location, phi) in block.value_phi.iter() {
            let decision = &phi_solve_state[&phi.id];
            for incoming in phi.incoming.borrow().iter() {
                match incoming {
                    ValuePresence::Present(pos, source) => {
                        let source_decision = match source {
                            ValueSource::Source(_) => &PhiDecision::Present,
                            ValueSource::Phi(ValuePhi { id: source_phi_id,  .. }) => &phi_solve_state[source_phi_id]
                        };
                        // log::debug!("Phi {:?} = {:?} on edge {}, d: {:?} sd: {:?}", location, phi, pos, decision, source_decision);
                        if *decision == PhiDecision::Absent && *source_decision != PhiDecision::Absent {
                            values_to_unset.entry(*pos)
                                .or_insert_with(|| vec![])
                                .push(location.clone())
                        }
                    }
                    ValuePresence::Absent(_) => {}
                }
            }
        }
    }

    for (pos, values) in values_to_unset.iter() {
        for location in values {
            annotations.add(*pos, format!("unset {:?}", location));
        }
    }

    annotations.dump_annotated(ir);

    let mut ir_to_insert = Vec::new();

    for (pos, values) in values_to_unset.iter() {
        if values.is_empty() {
            continue;
        }
        let mut new_ir = Vec::new();

        let instruction = &mut ir[pos.clone()];
        match instruction {
            DMIR::Jmp(_) => {
                generate_variable_terminations(values, &mut new_ir);
                // Insert terminations directly before jmp
                ir_to_insert.push((*pos, new_ir));
            }
            DMIR::JZ(label) | DMIR::JZInternal(label) | DMIR::JNZInternal(label) => {
                // Insert terminations into block after instruction, along with jump-over
                let name = generate_variable_termination_block(*pos, label, values, &mut new_ir);
                *label = name;
                ir_to_insert.push((pos + 1, new_ir));
            }
            _ => panic!("Unexpected instruction: {:?}", instruction)
        }
    }

    ir_to_insert.sort_by_key(|(pos, _)| *pos);

    let mut offset: usize = 0;
    for (pos, insert) in ir_to_insert {
        let final_pos = pos + offset;
        offset += insert.len();
        ir.splice(final_pos..final_pos, insert);
    }

    annotations.clear();
    log::debug!("variable_termination_pass generated:");
    annotations.dump_annotated(ir);
}


/// returns: name of generated block
fn generate_variable_termination_block(from: usize, label: &str, values_to_unset: &Vec<ValueLocation>, ir_to_append: &mut Vec<DMIR>) -> String {
    let name = format!("lt_from_{}_to_{}", from, label);
    let after_block_name = format!("lt_next_from_{}", from);
    ir_to_append.push(DMIR::Jmp(after_block_name.clone()));
    ir_to_append.push(DMIR::EnterBlock(name.clone()));
    generate_variable_terminations(values_to_unset, ir_to_append);
    ir_to_append.push(DMIR::Jmp(label.to_string()));
    ir_to_append.push(DMIR::EnterBlock(after_block_name));
    return name
}

fn generate_variable_terminations(values_to_unset: &Vec<ValueLocation>, ir_to_append: &mut Vec<DMIR>) {
    for location in values_to_unset {
        match location {
            ValueLocation::Stack(_) | ValueLocation::Argument(_) => panic!(),
            ValueLocation::Cache => ir_to_append.push(DMIR::UnsetCache),
            ValueLocation::Local(idx) => ir_to_append.push(DMIR::UnsetLocal(*idx)),
        }
    }
}

#[derive(Clone, Ord, Eq, PartialOrd, PartialEq, Debug)]
enum PhiDecision {
    Present,
    Undecided,
    Absent
}

#[derive(Clone, PartialEq, Debug)]
enum ValuePresence<'t> {
    /// argument is origin branching instruction location
    Present(usize, ValueSource<'t>),
    Absent(usize)
}

#[derive(Clone, PartialEq)]
enum ValueSource<'t> {
    Source(usize),
    Phi(&'t ValuePhi<'t>)
}

#[derive(PartialEq)]
struct ValuePhi<'t> {
    id: usize,
    incoming: RefCell<Vec<ValuePresence<'t>>>
}


impl Debug for ValueSource<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.dfs_format(f, &mut HashSet::new())
    }
}

impl ValueSource<'_> {
    fn dfs_format(&self, fmt: &mut Formatter<'_>, visited_phi_id: &mut HashSet<usize>) -> std::fmt::Result {
        match self {
            ValueSource::Source(pos) => write!(fmt, "Source({})", pos),
            ValueSource::Phi(phi) => {
                phi.dfs_format(fmt, visited_phi_id)
            }
        }
    }
}

impl Debug for ValuePhi<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.dfs_format(f, &mut HashSet::new())
    }
}

impl ValuePhi<'_> {
    fn dfs_format(&self, fmt: &mut Formatter<'_>, visited_phi_id: &mut HashSet<usize>) -> std::fmt::Result {
        write!(fmt, "Phi(id: {}, incoming: [", self.id)?;
        if !visited_phi_id.insert(self.id) {
            write!(fmt, "...")?;
        } else {
            for (idx, node) in self.incoming.borrow().iter().enumerate() {
                if idx != 0 {
                    write!(fmt, ", ")?;
                }
                match node {
                    ValuePresence::Present(pos, source) => {
                        write!(fmt, "Present({}, ", pos)?;
                        source.dfs_format(fmt, visited_phi_id)?;
                        write!(fmt, ")")?;
                    }
                    ValuePresence::Absent(_) => write!(fmt, "{:?}", node)?
                }
            }
        }
        write!(fmt, "])")
    }
}

#[derive(Debug)]
struct BlockData<'t> {
    incoming_instructions: Vec<usize>,
    value_phi: HashMap<ValueLocation, &'t ValuePhi<'t>>
}

struct AnalyzerState<'t> {
    phi_id: Box<usize>,
    arena: &'t Arena<ValuePhi<'t>>,
    blocks: HashMap<String, BlockData<'t>>,
    value_sources: HashMap<ValueLocation, ValueSource<'t>>,
    all_phi: Vec<&'t ValuePhi<'t>>
}

impl <'t> AnalyzerState<'t> {
    fn new(arena: &'t Arena<ValuePhi<'t>>) -> Self {
        AnalyzerState {
            phi_id: Box::new(0),
            arena,
            blocks: HashMap::new(),
            value_sources: HashMap::new(),
            all_phi: Vec::new()
        }
    }

    fn merge_presences(&mut self, pos: usize, block_label: String) {
        let block_data = self.blocks.entry(block_label)
            .or_insert_with(|| BlockData { incoming_instructions: vec![], value_phi: HashMap::new() });

        let block_keys = block_data.value_phi.keys().cloned().collect::<HashSet<_>>();
        let current_keys = self.value_sources.keys().cloned().collect::<HashSet<_>>();

        let incoming_instructions = &block_data.incoming_instructions;
        let arena = self.arena;
        let all_phi = &mut self.all_phi;
        for location in block_keys.union(&current_keys) {

            let phi_id = self.phi_id.borrow_mut();
            let phi = block_data.value_phi
                .entry(location.clone())
                .or_insert_with(|| {
                    let id = *phi_id;
                    *phi_id += 1;
                    let value = ValuePhi {
                        id,
                        incoming: RefCell::new(incoming_instructions.iter().map(|instruction_pos| ValuePresence::Absent(*instruction_pos)).collect())
                    };

                    let allocated = arena.alloc(
                        value
                    );
                    all_phi.push(allocated);
                    allocated
                });
            if let Some(source) = self.value_sources.get(location) {
                phi.incoming.borrow_mut().push(ValuePresence::Present(pos, source.clone()));
            } else {
                phi.incoming.borrow_mut().push(ValuePresence::Absent(pos));
            }
        }

        block_data.incoming_instructions.push(pos);
    }

    fn analyze<'q>(&'q mut self, ir: &'q Vec<DMIR>) {
        let mut block_ended = false;
        for (pos, instruction) in ir.iter().enumerate() {
            match instruction {
                DMIR::SetLocal(idx) => {
                    self.value_sources.insert(
                        ValueLocation::Local(*idx),
                        ValueSource::Source(pos)
                    );
                }
                DMIR::SetCache => {
                    self.value_sources.insert(
                        ValueLocation::Cache,
                        ValueSource::Source(pos)
                    );
                }
                DMIR::Ret => {
                    block_ended = true;
                }
                DMIR::ValueTagSwitch(_, cases) =>{
                    for (_, block) in cases {
                        self.merge_presences(pos, block.clone());
                    }
                    block_ended = true;
                }
                DMIR::JZ(label) | DMIR::JZInternal(label) | DMIR::JNZInternal(label) => {
                    self.merge_presences(pos, label.clone())
                }
                DMIR::EnterBlock(label) => {
                    assert!(block_ended, "fallthrough termination incorrect");

                    block_ended = false;
                    self.value_sources.clear();
                    let label = label.clone();
                    let block = self.blocks.get(label.as_str()).unwrap();
                    for (location, phi) in block.value_phi.iter() {
                        self.value_sources.insert(location.clone(), ValueSource::Phi(*phi));
                    }
                }
                DMIR::Jmp(label) => {
                    self.merge_presences(pos, label.clone());
                    block_ended = true;
                }
                DMIR::End => {
                    block_ended = true;
                }
                DMIR::UnsetLocal(idx) => {
                    assert!(self.value_sources.remove(&ValueLocation::Local(*idx)).is_some());
                }
                DMIR::UnsetCache => {
                    assert!(self.value_sources.remove(&ValueLocation::Cache).is_some());
                }
                _ => {}
            }
        }
    }
}
