# KNOWN_WRONG — defects found in the prior codebase

Each entry has a **red test** in this repo so the rebuild cannot reintroduce it.
**This file records DEFECTS ONLY.** It is not a channel for importing prior results.

| # | defect | test |
|---|---|---|
| 1 | `region_len` held the annotated *region* span (core + flanks) while `sequence` held the core — a ~10x difference (observed: 11,008 vs 1,008). Any length statistic on `region_len` is wrong. This schema has no region-span field. | `test_seq_len_is_sequence` |
| 2 | The split builder initialised an empty manifest and rewrote the whole file, so building one class silently destroyed every sibling entry. Found live: 12 subclass datasets carrying an all-zero manifest, i.e. no recorded leakage verification at all. | `test_manifest_additive` |
| 3 | The novelty gate computed containment with `max(..., default=0.0)`, and 0.0 is the *passing* value — an empty k-mer set silently passed. A gate must fail closed. | `test_novelty_gate_fails_closed` |
| 4 | Split integrity was unchecked: an observed corpus had a class with 0 val and 0 test records. | `test_split_nonempty` |
| 5 | antiSMASH `--minlength` defaults to 1000 and filters the **input record** length, not the cluster length. On extracted cores this silently discarded 72% of TERPENE and 48% of RIPP while dropping nothing from long classes — a class-asymmetric denominator that presents as "short classes are harder". | `test_every_sequence_gets_a_verdict` |
| 6 | One metric name (`best_bio_bits`) meant two different quantities depending on which Pfam subset was used; the class argument was accepted but ignored for that score. It inverted a headline conclusion. | n/a — no HMM score is an endpoint here (spec §3.6) |
| 8 | **Our own, not inherited.** Accessions keyed as `<genome>.region<n>` collided: antiSMASH numbers regions PER RECORD, so region_number restarts on every contig and 307,069 of 540,695 records (56.8%) shared a key. Every downstream dict — cluster assignment, record_split, component lookup, mmseqs FASTA headers — silently overwrote. | `test_corpus_accessions_are_unique`, `test_a_record_lands_in_one_partition_across_all_classes` |
| 9 | **Our own.** `join()`/`order()` locations were collapsed to a min..max bounding box, so an origin-spanning gene on a circular replicon became a multi-megabase feature overlapping every cluster on it. `cds_count` was wrong for 13.6% of records. | `test_join_locations_do_not_use_the_bounding_box` |
| 7 | Sequence extraction truncated at a stray byte, discarding a reported median of 6.2 kb of generated output. | `test_no_scoring_window` + explicit length reporting (§7.8) |
