import argparse
import os
from pathlib import Path


MODEL_OVERRIDE_ENV = "MACRO_LLM_MODEL"
RUN_NAME_OVERRIDE_ENV = "MACRO_LLM_RUN_NAME"
TEST_MODE_OVERRIDE_ENV = "MACRO_LLM_TEST_MODE"
CONFIG_MODEL_ENV = "MODEL"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


def load_runtime_config(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if separator and name and value and name not in os.environ:
            os.environ[name] = value


load_openrouter_config = load_runtime_config


def infer_model_flag(model, fallback):
    model_name = model.strip().lower()
    if model_name.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "GPT"
    if model_name.startswith(("meta-llama/", "mistralai/")):
        return "Llama"
    return fallback


def parse_model_arg(value):
    model = value.strip()
    if not model:
        raise argparse.ArgumentTypeError("--model must not be empty or whitespace")
    return model


def set_model_override(model):
    if model is None:
        os.environ.pop(MODEL_OVERRIDE_ENV, None)
    else:
        os.environ[MODEL_OVERRIDE_ENV] = model


def apply_model_override(config):
    model = os.environ.get(MODEL_OVERRIDE_ENV) or os.environ.get(CONFIG_MODEL_ENV)
    if model is not None:
        config.set("LLM_CONFIG", "MODEL", model)
    return config.get("LLM_CONFIG", "MODEL")


def apply_run_name_override(config):
    """Resolve the output label, accepting legacy TEST_MODE configuration."""
    run_name = os.environ.get(RUN_NAME_OVERRIDE_ENV)
    if run_name is None:
        run_name = os.environ.get(TEST_MODE_OVERRIDE_ENV)
    if run_name is None:
        option = "RUN_NAME" if config.has_option("LLM_CONFIG", "RUN_NAME") else "TEST_MODE"
        run_name = config.get("LLM_CONFIG", option)
    config.set("LLM_CONFIG", "RUN_NAME", run_name)
    return run_name


def apply_test_mode_override(config):
    """Compatibility wrapper for callers using the old output-label name."""
    run_name = apply_run_name_override(config)
    config.set("LLM_CONFIG", "TEST_MODE", run_name)
    return run_name
