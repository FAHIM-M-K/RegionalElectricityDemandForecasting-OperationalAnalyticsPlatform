"""Schema-safe helpers for idempotent Delta upserts."""


def merge_source_columns(
    delta_target,
    source_df,
    condition,
    *,
    update_condition=None,
    update_exclude=(),
    target_alias="target",
    source_alias="source",
    update_matched=True,
):
    """Merge explicitly mapped source fields and reject unplanned schema drift.

    Nullable target-only fields are left untouched on matches and null on new
    rows. Source-only or required target-only fields fail before Delta executes,
    with an actionable schema-migration message instead of a cryptic MERGE error.
    """
    target_fields = {field.name: field for field in delta_target.toDF().schema.fields}
    source_columns = list(source_df.columns)
    source_names = set(source_columns)
    target_names = set(target_fields)

    source_only = sorted(source_names - target_names)
    if source_only:
        raise RuntimeError(
            "Delta MERGE schema mismatch: source has fields missing from the target "
            f"{source_only}. Apply an explicit target schema migration before merging."
        )

    required_target_only = sorted(
        name for name in target_names - source_names
        if not target_fields[name].nullable
    )
    if required_target_only:
        raise RuntimeError(
            "Delta MERGE schema mismatch: target has required fields absent from the "
            f"source {required_target_only}. Populate them or migrate the target schema."
        )

    excluded = set(update_exclude)
    update_values = {
        name: f"{source_alias}.`{name}`" for name in source_columns if name not in excluded
    }
    insert_values = {name: f"{source_alias}.`{name}`" for name in source_columns}

    merge = delta_target.alias(target_alias).merge(source_df.alias(source_alias), condition)
    if update_matched and update_values:
        merge = merge.whenMatchedUpdate(condition=update_condition, set=update_values)
    return merge.whenNotMatchedInsert(values=insert_values).execute()
