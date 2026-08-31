import argparse
import os
import warnings

from LLMmodules.model_config import parse_model_arg, set_model_override

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MACRO-LLM on the CPP task.")
    parser.add_argument("--env", choices=["catchup", "slowdown"], default="catchup")
    parser.add_argument(
        "--model", type=parse_model_arg, default=None, help="Override MODEL for this process"
    )
    parser.add_argument("--max-steps", type=int, default=120, help="Number of LLM evaluation steps")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    return args


def init_env(input_args):
    from algorithms.envs.CACC import (
        CACC_catchup,
        CACC_catchup_test,
        CACC_slowdown,
        CACC_slowdown_test,
    )

    if input_args.env == "catchup":
        return CACC_catchup, CACC_catchup_test
    return CACC_slowdown, CACC_slowdown_test


def main():
    input_args = parse_args()
    set_model_override(input_args.model)

    os.environ["CACC_ENV"] = input_args.env

    import torch
    from LLMmodules.CACC.LLM_MB_DPPO_cacc import OnPolicyRunner_LLM

    env_fn_train, env_fn_test = init_env(input_args)
    env_train = env_fn_train()
    env_test = env_fn_test()
    torch.set_num_threads(1)
    print(f"n_threads {torch.get_num_threads()}")
    print(f"n_gpus {torch.cuda.device_count()}")

    runner = OnPolicyRunner_LLM(env_train, env_test, input_args)
    runner.test()


if __name__ == "__main__":
    main()
