"""
Qwen 2 online N/Z operator matching experiment.

TODO: Adapt inference loop patterns from kv_compaction_experiment.
TODO: Implement live query collection hooks.
TODO: Wire up query bank, beta-fit, and verification pipeline.
TODO: Implement evaluation and results logging.
"""
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen 2 online N/Z operator matching experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the experiment config YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/run",
        help="Directory to write results to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: load config from args.config (e.g., via PyYAML + dacite)
    # TODO: set random seed from config.experiment.seed
    # TODO: load Qwen 2 model and tokenizer from config.model.name
    # TODO: move model to config.model.device with config.model.dtype
    # TODO: register attention hooks to capture live query vectors per layer/head
    # TODO: load evaluation prompts (port from kv_compaction_experiment)
    # TODO: for each prompt:
    #         run inference token-by-token
    #         after each token, collect query vectors into QueryBank
    #         at each checkpoint boundary:
    #           for each budget_fraction in config.experiment.budget_fractions:
    #             for each baseline in config.experiment.baselines:
    #               propose support (recency/attention_mass/uniform)
    #               optionally refit beta
    #               run verification gate
    #               record L_true on holdout
    # TODO: aggregate results across prompts and log to args.output_dir

    raise NotImplementedError("Experiment scaffold — see TODOs above")


if __name__ == "__main__":
    main()
