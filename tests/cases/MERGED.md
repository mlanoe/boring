# Merged scenario files

These fixtures replace the individual cases below (merged to cut the number of cargo builds;
see `tools/merge_cases.py`). Each scenario keeps its original name as a `# ---- name ----` section,
with its top-level items prefixed by that name, and prints a `== name ==` header.

| scenario file | original cases |
|---|---|
| string_scenarios | string_temporary_method_chains, string_len_cast_parse_field, string_methods_after_index, string_unicode_length, string_graphemes, string_grapheme_untyped, string_grapheme_cast_var |
| actor_scenarios | actor_field_through, actor_call_result_ctor_arg, actor_inferred_param_caller_reuse, actor_field_method_via_call_local, actor_collection_field_derive |
| cast_scenarios | cast_lhs_of_lt, cast_float_lit_to_int |
| closure_scenarios | closure_owned_captures, closure_param_forward |
| field_scenarios | field_method_same_name, field_mut_content_mutation, field_mut_param_content_mutation |
| fn_scenarios | fn_handle_inline_closure_ctor, fn_typed_local_alias |
| for_scenarios | for_loop_enum_array_camel_case_method, for_mut_elem_iter_mut |
| json_scenarios | json_serde_rename, json_untagged_enum, json_serde_shapes |
| mut_scenarios | mut_struct_param, mut_collection_by_value_reuse, mut_local_trait_array_push, mut_elem_trait_array_literal |
| optional_scenarios | optional_field_is_nil, optional_fn_param, optional_let_return |
| parse_scenarios | parse_float_if_let_dict_index, parse_on_expr_receiver |
| self_scenarios | self_lent_reborrow, self_sibling_method_call, self_field_collection_return_by_value |
| struct_scenarios | struct_borrow_method_param, struct_default_labeled, struct_field_array_interp, struct_param_nested_loop_call |
| trait_scenarios | trait_dynamic_dispatch, trait_param_forwarding, trait_param_return, trait_method_return, trait_borrow_param |
| trait_scenarios_2 | trait_borrow_infer, trait_borrow_method_infer |
| trait_scenarios_3 | trait_borrow_contract, trait_borrow_mut, trait_borrow_method_param |
| var_scenarios | var_param_bare_trait_assign, var_rebind_alias_independent |
| view_scenarios | view_method_modifiers, view_state_seed |
| misc_scenarios | by_value_arg_then_reuse, dict_field_param_var_key_else |
| misc_scenarios_2 | do_expr_value_for_loop, enum_derive_no_debug, explicit_borrow_field_type_cast, guard_string_trim_temporary, init_bare_field_assign |
| misc_scenarios_3 | nested_dict_field_assign_in_place, print_optional_value, req_method_mut_param_write |
| misc_scenarios_4 | set_param_methods, single_letter_struct_holds_dyn |
| misc_scenarios_5 | untyped_string_lit_local_to_type_method, zero_arg_ctor_defaults_with_optional |
| actor_scenarios_2 | actor_guard_scalar_mutation, actor_field_method_outer_var |
| fn_scenarios_2 | fn_field_optional_alias, fn_field_collections, fn_field_collection_writes, fn_field_optional_self_call |
| labeled_scenarios | labeled_ctor_init_call, labeled_ctor_init_call_qualified |
| mut_scenarios_2 | mut_local_collection_alias, mut_local_float_array_elem_type |
| oversized_scenarios | oversized_return_boxed_direct, oversized_return_boxed_via_let, oversized_return_small_stays_unboxed, oversized_struct_field_stays_inline, oversized_param_borrow_stays_unboxed |
| view_scenarios_2 | view_field_struct_no_partial_eq, view_act_str_args_trait_obj, view_let_bound_callbacks, view_state_owner_report_build, view_trailing_block_multiline_callback |
| misc_scenarios_6 | closure_string_literal_result, external_read_buffer |
| misc_scenarios_7 | method_name_collides_builtin, optional_dyn_field_struct_literal_clone, split_call_arg_then_split, throws_method_name_collision, unwrap_or_struct_local_by_value |
| dict_scenarios | dict_index_else_on_call_result, dict_string_key_index |
| fn_scenarios_3 | fn_named_rust_keyword, fn_field_handle_call |
| mut_scenarios_3 | mut_param_if_let_field_forward, mut_scalar, mut_param_unresolved_receiver |
| optional_scenarios_2 | optional_field_else_keeps_field, optional_let_throws_non_optional_return |
| trait_scenarios_4 | trait_downcast_shared_actor_state, trait_method_casing_unknown_receiver |
| misc_scenarios_8 | field_by_value_reuse, init_bare_field_assign_qualified |
| misc_scenarios_9 | stored_closure_captures_self_and_loop_item, struct_count_field, system_time |
| qualified_scenarios | qualified_ctor_positional_args, qualified_struct_literal_positional_args |
| misc_scenarios_10 | mem_borrow_builtins, monomorphize_fn |
| misc_scenarios_11 | try_else_nil_if_let, uint_int_cross_eq |
