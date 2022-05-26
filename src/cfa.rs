use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use auxtools::raw_types::values::ValueTag::Area;
use dmasm::Instruction::Ref;
use typed_arena::Arena;
use crate::dmir::DMIR;

pub struct ControlFlowGraph<'t> {
    analyzer: &'t ControlFlowAnalyzer<'t>,
    pub nodes: HashMap<String, &'t CFGNode<'t>>
}

pub struct CFGNode<'t> {
    pub label: String,
    pub inbound: RefCell<Vec<&'t CFGNode<'t>>>,
    pub outbound: RefCell<Vec<&'t CFGNode<'t>>>
}

impl<'t> CFGNode<'t> {
    fn new(label: String) -> Self {
        Self {
            label,
            inbound: RefCell::new(vec![]),
            outbound: RefCell::new(vec![])
        }
    }

    fn add_outbound_edge(&self, target: &'t CFGNode<'t>) {
        self.outbound.borrow_mut().push(target);
        target.inbound.borrow_mut().push(target);
    }

}

pub struct ControlFlowAnalyzer<'t> {
    arena: Arena<CFGNode<'t>>
}


impl<'t> ControlFlowGraph<'t> {
    fn get_or_create_node<'q>(&'q mut self, label: &'_ str) -> &'t CFGNode<'t> {
        let arena = &self.analyzer.arena;
        *self.nodes.entry(label.to_owned())
            .or_insert_with(|| arena.alloc(CFGNode::new(label.to_owned())))
    }
}

impl<'t> ControlFlowAnalyzer<'t> {
    pub fn new() -> Self {
        Self {
            arena: Default::default()
        }
    }

    pub fn analyze(&'t self, instructions: &Vec<DMIR>) -> ControlFlowGraph<'t> {
        let mut graph = ControlFlowGraph {
            analyzer: self,
            nodes: Default::default()
        };


        let entry = graph.get_or_create_node("entry");
        let exit = graph.get_or_create_node("exit");


        let mut current_node = entry;

        for instruction in instructions {
            match instruction {
                DMIR::ValueTagSwitch(_, cases) => {
                    for (_, label) in cases {
                        current_node.add_outbound_edge(graph.get_or_create_node(label));
                    }
                }
                DMIR::Ret => {
                    current_node.add_outbound_edge(exit);
                }
                DMIR::EnterBlock(label) => {
                    current_node = graph.get_or_create_node(label);
                }
                DMIR::JZ(label) |
                DMIR::JZInternal(label) |
                DMIR::JNZInternal(label) |
                DMIR::Jmp(label) => {
                    current_node.add_outbound_edge(
                        graph.get_or_create_node(label)
                    )
                }
                DMIR::Deopt(_, _) => {}
                DMIR::End => {
                    current_node.add_outbound_edge(exit);
                }
                _ => {}
            }
        }

        graph
    }
}
