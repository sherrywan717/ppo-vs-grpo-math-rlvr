# Case studies

**Matched 0.5B pilot - not the final benchmark**

Examples are mechanically selected from saved seed-2026 evidence (lowest and highest shaped reward), not cherry-picked as algorithm wins.

## PPO lowest reward

- Key: `countdown:train:1::generation:3`
- Status/reward: `format_error` / 0.0
- Saved completion excerpt: `<reasoning> We need to use the numbers 16, 4, 11, 13 exactly once to make -4. Let's try different combinations: - (13 * 16 - 11) / 4 = -4 - 16 - (13 * 11 - 4) / 13 = -4 - (13 * 11 - 16) / 4 = -4 - (13 * 11 - 4) / 16 = -4 - 16 - 1`

## PPO highest reward

- Key: `countdown:train:0::generation:0`
- Status/reward: `format_error` / 0.15
- Saved completion excerpt: `<answer> 5 + 9 - 13 * 6 </answer>`

## GRPO lowest reward

- Key: `countdown:train:1::generation:0`
- Status/reward: `format_error` / 0.0
- Saved completion excerpt: `<reasoning> We need to use each number exactly once and the operations +, -, *, / to make -4. Let's try different combinations: 1. \( 16 + 11 - 13 = 24 \) 2. \( 16 - 13 + 11 = 14 \) 3. \( 16 \times 11 - 13 = 161 - 13 = 148 \) 4. \( 16 \times 11 - 13 + 1 = 176 - 1`

## GRPO highest reward

- Key: `countdown:train:1::generation:1`
- Status/reward: `invalid_number_usage` / 0.15
- Saved completion excerpt: `<reasoning> -4 = -(16 - 4 + 11) -4 = -12 + 11 -4 = -1 </reasoning> <answer>4</answer>`

These examples illustrate artifact traceability and staged-reward behavior; they do not establish comparative quality.
