# v0/v1 rendered prompt diffs

## countdown:train:0

```diff
--- countdown:train:0/v0
+++ countdown:train:0/v1
@@ -1,5 +1,14 @@
 <|im_start|>system
-Solve the math problem. Output exactly one <reasoning>...</reasoning> block followed by exactly one <answer>...</answer> block. Put only the final answer in <answer>.<|im_end|>
+Solve the math problem concisely. Follow the output protocol at the end of the user message exactly.<|im_end|>
 <|im_start|>user
-Use each of [5, 9, 13, 6] exactly once with +, -, *, / to make 15/26.<|im_end|>
+Use each of [5, 9, 13, 6] exactly once with +, -, *, / to make 15/26.
+
+OUTPUT PROTOCOL — follow exactly:
+<reasoning>...</reasoning>
+<answer>...</answer>
+Return exactly those two closed blocks, in that order, with no text outside them.
+Keep reasoning to at most two short sentences.
+Inside <answer>, write one arithmetic expression only.
+Do not include prose, an equals sign, or the target value.
+Use every input number exactly once. Use only +, -, *, /, and parentheses.<|im_end|>
 <|im_start|>assistant
```

## countdown:train:1

```diff
--- countdown:train:1/v0
+++ countdown:train:1/v1
@@ -1,5 +1,14 @@
 <|im_start|>system
-Solve the math problem. Output exactly one <reasoning>...</reasoning> block followed by exactly one <answer>...</answer> block. Put only the final answer in <answer>.<|im_end|>
+Solve the math problem concisely. Follow the output protocol at the end of the user message exactly.<|im_end|>
 <|im_start|>user
-Use each of [16, 4, 11, 13] exactly once with +, -, *, / to make -4.<|im_end|>
+Use each of [16, 4, 11, 13] exactly once with +, -, *, / to make -4.
+
+OUTPUT PROTOCOL — follow exactly:
+<reasoning>...</reasoning>
+<answer>...</answer>
+Return exactly those two closed blocks, in that order, with no text outside them.
+Keep reasoning to at most two short sentences.
+Inside <answer>, write one arithmetic expression only.
+Do not include prose, an equals sign, or the target value.
+Use every input number exactly once. Use only +, -, *, /, and parentheses.<|im_end|>
 <|im_start|>assistant
```
