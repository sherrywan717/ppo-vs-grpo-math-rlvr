# Error analysis

**Matched 0.5B pilot - not the final benchmark**

Across 96 completions, canonical statuses were `format_error`=83, `invalid_expression`=10, `invalid_number_usage`=3. Canonical correctness was zero in all runs. Most failures were protocol/format failures; invalid-expression or invalid-number-usage outcomes occasionally received partial shaped reward without becoming correct solutions. The execution path preserved graded signal, but the single-update 0.5B setting did not establish task success.
