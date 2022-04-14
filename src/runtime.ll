; ModuleID = 'runtime.ll'
; This file contains dmjit runtime support library bindings

%DMValue = type { i8, i32 }
%List = type { %DMValue*, i8*, i32, i32, i32, i8* }

@dmir.runtime.GLOB_LIST_ARRAY = external global %List**, align 4

declare external i8 @dmir.runtime.inc_ref_count(%DMValue)
declare external i8 @dmir.runtime.dec_ref_count(%DMValue)
declare external i8 @dmir.runtime.get_variable(%DMValue*, %DMValue, i32)
declare external i8 @dmir.runtime.set_variable(%DMValue, i32, %DMValue)
declare external i8 @dmir.runtime.call_proc_by_id(%DMValue*, %DMValue, i32, i32, i32, %DMValue, %DMValue*, i32, i32, i32)
declare external i8 @dmir.runtime.call_proc_by_name(%DMValue*, %DMValue, i32, i32, %DMValue, %DMValue*, i32, i32, i32)

declare external void @dmir.runtime.deopt(i64)

declare external void @dmir.runtime.debug.handle_debug(i8*)
declare external void @dmir.runtime.debug.handle_debug_val(%DMValue)

declare external void     @dmir.runtime.unset_assoc_list(i8**, %DMValue)
declare external %DMValue @dmir.runtime.list_associative_get(%DMValue, %DMValue)
declare external void     @dmir.runtime.list_associative_set(%DMValue, %DMValue, %DMValue)
declare external %DMValue @dmir.runtime.list_copy(%DMValue)
declare external void     @dmir.runtime.list_append(%DMValue, %DMValue)
declare external void     @dmir.runtime.list_remove(%DMValue, %DMValue)

declare external i1       @dmir.runtime.is_dm_entity(%DMValue)
declare external i1       @dmir.runtime.is_subtype_of(%DMValue, %DMValue)

declare external %DMValue @dmir.runtime.get_step(%DMValue, i8)

; It should be guaranteed by caller, that on entry to this function, arg_count is at least caller_arg_count - 1
define void @dmir.intrinsic.unref_excess_arguments_internal(%DMValue* %args, i32 %arg_count, i32 %caller_arg_count) cold {
entry:
    br label %unref
unref:
    %arg_count_phi = phi i32 [ %arg_count, %entry ], [ %arg_count_inc, %unref ]
    %ptr = getelementptr %DMValue, %DMValue* %args, i32 %arg_count
    %value = load %DMValue, %DMValue* %ptr
    %unused = call i8 @dmir.runtime.dec_ref_count(%DMValue %value)
    %arg_count_inc = add i32 %arg_count_phi, 1
    ; %arg_count < %caller_arg_count
    %cond = icmp ult i32 %arg_count_inc, %caller_arg_count
    br i1 %cond, label %unref, label %post_unref
post_unref:
    ret void
}

define available_externally void @dmir.intrinsic.unref_excess_arguments(%DMValue* %args, i32 %arg_count, i32 %caller_arg_count) alwaysinline {
; if (%arg_count < %caller_arg_count) {
check:
    %cond = icmp ult i32 %arg_count, %caller_arg_count
    br i1 %cond, label %unref, label %post
; call_unref()
unref:
    call void @dmir.intrinsic.unref_excess_arguments_internal(%DMValue* %args, i32 %arg_count, i32 %caller_arg_count)
    br label %post
; }
post:
    ret void
}


define %List* @dmir.intrinsic.get_list(%DMValue %list_id) alwaysinline {
entry:
    %id = extractvalue %DMValue %list_id, 1
    %glob_list = load %List**, %List*** @dmir.runtime.GLOB_LIST_ARRAY, align 4
    %array_element = getelementptr %List*, %List** %glob_list, i32 %id
    %ret = load %List*, %List** %array_element, align 4

    ret %List* %ret
}

define %DMValue @dmir.runtime.list_indexed_get(%DMValue %list_id, i32 %index) alwaysinline {
entry:
    %list_ptr = call %List* @dmir.intrinsic.get_list(%DMValue %list_id)

    %vector_part_ptr = getelementptr inbounds %List, %List* %list_ptr, i32 0, i32 0
    %vector_part = load %DMValue*, %DMValue** %vector_part_ptr, align 4
    %index_dec = sub i32 %index, 1
    %array_element = getelementptr %DMValue, %DMValue* %vector_part, i32 %index_dec
    %ret = load %DMValue, %DMValue* %array_element, align 4

    ret %DMValue %ret
}

define void @dmir.runtime.list_indexed_set(%DMValue %list_id, i32 %index, %DMValue %value) alwaysinline {
entry:
    %list_ptr = call %List* @dmir.intrinsic.get_list(%DMValue %list_id)

    %vector_part_ptr = getelementptr inbounds %List, %List* %list_ptr, i32 0, i32 0
    %assoc_part = getelementptr inbounds %List, %List* %list_ptr, i32 0, i32 1
    %vector_part = load %DMValue*, %DMValue** %vector_part_ptr, align 4
    %index_dec = sub i32 %index, 1
    %array_element = getelementptr %DMValue, %DMValue* %vector_part, i32 %index_dec
    %prev = load %DMValue, %DMValue* %array_element, align 4
    %unused = call i8 @dmir.runtime.dec_ref_count(%DMValue %prev)
    call void @dmir.runtime.unset_assoc_list(i8** %assoc_part, %DMValue %prev)
    store %DMValue %value, %DMValue* %array_element, align 4

    ret void
}

define i1 @dmir.runtime.list_check_size(%DMValue %list_id, i32 %index) alwaysinline {
entry:
    %list_ptr = call %List* @dmir.intrinsic.get_list(%DMValue %list_id)

    %len_ptr = getelementptr inbounds %List, %List* %list_ptr, i32 0, i32 3
    %len = load i32, i32* %len_ptr, align 4

    %gt_zero = icmp sgt i32 %index, 0
    %lt_size = icmp sle i32 %index, %len
    %ret = and i1 %gt_zero, %lt_size

    ret i1 %ret
}