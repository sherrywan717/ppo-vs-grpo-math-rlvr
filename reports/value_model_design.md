# PPO Value Model Design Note

No value model is created in phase 3. TRL 0.24.0 requires an explicit `value_model` alongside
the policy, reference model, and reward model. The implementation decision remains open:

| Option | Memory | Expected quality | Main risk |
|---|---|---|---|
| Independent value model | Highest: another full backbone plus optimizer state | Strongest capacity and isolation from policy updates | Expensive PPO footprint and checkpoints |
| Frozen backbone + trainable value head | Lowest trainable memory | Stable but may underfit shifting policy distributions | Fixed representations limit value accuracy |
| Value LoRA + value head | Moderate; adapters and optimizer state added | Better task adaptation than a frozen backbone | More tuning and shared-base lifecycle complexity |

For the H800 experiment, first benchmark the frozen-backbone head on the 0.5B smoke set, then
compare value loss stability against value LoRA. An independent model should be used only if the
quality gain justifies its GPU-hour and checkpoint cost. The choice must preserve separate policy
and value checkpoints and must not allow gradients into the verifier reward adapter.

