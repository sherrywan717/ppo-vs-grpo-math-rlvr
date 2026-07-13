# PPO Value Model Static Contract

PPO uses the policy base checkpoint with `AutoModelForSequenceClassification(num_labels=1)`, an independent value LoRA (`r=8`, alpha 16, dropout 0) targeting q/v projections, and a trainable scalar score head. Policy and value adapters are independent. The verifier reward model remains parameter-free and detached. This phase provides schema and fake-model tests only and does not load a checkpoint.
