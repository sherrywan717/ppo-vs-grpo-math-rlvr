# Server destruction decision

status: safe_to_destroy_server

Verified conditions:

- fresh GitHub clone at patch tag `924e65e90fb64dcb671207df05e2e30ac4262178`;
- Q001–Q200, 200 questions, 200 answers, duplicate IDs 0;
- 185 Markdown links checked, broken links 0;
- all seven custom assets listed by GitHub Release;
- all seven independently downloaded, SHA256 checked and safely extracted;
- selected checkpoint-96 adapter, GRPO-v2 training evidence and four-model hidden evidence verified;
- all three evaluated adapter weight SHA identities verified;
- preservation inventory has 0 unclassified critical entries;
- Windows download and SHA verification steps are documented.

Release: https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr/releases/tag/v0.2.1-grpo-v2-archive

Explicit exclusions are classified rather than silently omitted: Qwen/Hugging Face cache
is rebuilt from the pinned public revision; complete Conda environment is rebuilt from
versions/configs; credentials are never archived; full optimizer/scheduler/RNG resume
state is private-only and is not required for final evaluated-adapter inference.
