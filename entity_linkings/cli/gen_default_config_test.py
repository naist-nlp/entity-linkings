import tempfile

import yaml

from entity_linkings.cli.gen_default_config import (
    TRAINING_CONFIG_NAME,
    generate_config,
    training_arguments,
)
from entity_linkings.trainer import TrainingArguments


def test_model_files_hold_only_the_model_config() -> None:
    # The training arguments are the same whatever the model, so they no longer get a
    # copy appended to each model's file.
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_config(tmpdir)
        written = yaml.safe_load(open(f"{tmpdir}/bm25.yaml"))

    assert list(written) == ["bm25"]


def test_training_arguments_get_a_file_of_their_own() -> None:
    # They used to be read off TrainingArguments.__dict__, which holds only what the
    # subclass restates, so report_to never reached the file.
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_config(tmpdir)
        written = yaml.safe_load(open(f"{tmpdir}/{TRAINING_CONFIG_NAME}"))

    assert list(written) == ["training_arguments"]
    assert set(written["training_arguments"]) == set(training_arguments)


def test_the_training_file_can_be_read_back() -> None:
    # Reading the defaults off an instance instead would store enums, which safe_load
    # refuses to construct.
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_config(tmpdir)
        written = yaml.safe_load(open(f"{tmpdir}/{TRAINING_CONFIG_NAME}"))

        arguments = TrainingArguments(output_dir=tmpdir, **written["training_arguments"])

    assert arguments.learning_rate == 1.e-5
