# Experiment continuity

The living experiment record is GitHub issue
https://github.com/guajun/gpt-dog-eval/issues/1 and its repository companion
`docs/experiment-journal.md`.

When conducting related experiments or implementing control/evaluation changes,
append the new result to that issue and update the journal. Include the hypothesis,
changed variables, code revision/dirty state, scene/seed/horizon, inference and
chunk budgets, complete run IDs, results, cost, failures, and next decision.
Retain all attempts, including timeouts, falls, give_up, and budget exhaustion.
Do not overwrite historical results or mix the best scenes from separate batches.
Mark planned experiments as unrun until actual logs exist.

For new comparisons, exclude v1 and other pre-feedback-v2 Astra runs at the user's
request. Keep their historical records intact. The current reference set is Zero,
native ONNX-PPO, Astra feedback v2 (`astra-feedback-v2-20260910T083830Z`), and the
new independently contextualized experiment. Static robot specifications are a
separate experimental condition; record their contents and hash.

Keep ONNX's original 48-value actor input intact. Current applied-action feedback
is an execution receipt, while contact sensors or a low-level stabilizer change
the experiment condition and must be labeled as separate comparisons.

Public experiment records must omit credentials, secret.toml contents, private
inference endpoints, and SSH configuration. Keep raw logs and large artifacts in
the ignored output directories; publish their run IDs and relevant summaries.
