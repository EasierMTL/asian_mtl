import argparse
import os
import torch
import yaml
from dataclasses import dataclass
from dacite import from_dict
from datetime import datetime, timedelta
from novel_tl_models.models.base import ChineseToEnglishTranslator, Predictor
from novel_tl_models.evaluation.evaluation_pipeline import EvaluationPipeline


@dataclass
class EvaluationPipelineArgs:
    """Config strcuture for defining the EvaluationPipeline arguments.
    """
    num_samples_to_test: int
    num_workers: int
    sentence_bleu: bool
    print_bleu_every: int


@dataclass
class EvaluationConfig:
    """Config structure for the `EvaluationRunner`.

    Example configs can be found in `scripts/evaluation/configs`
    """
    model: str
    chinese_to_english: bool
    quantized: bool
    run: EvaluationPipelineArgs


class EvaluationRunner():
    """Runs EvaluationPipeline with command line arguments.
    """

    def initialize_translator(self, quantize: bool = True):
        """Initializes the translator with the desired model type.
        """
        translator = ChineseToEnglishTranslator()
        if quantize:
            translator.quantize()

        return translator

    def initialize_pipeline(self, translator: Predictor):
        """Initializes the evaluation pipeline by loading the translator and
        data into it. The data used should be located in the `scripts/test_data`
        directory.
        """
        if translator is None:
            raise ValueError("must call initialize_translator before load_data")

        pipeline = EvaluationPipeline(predictor=translator)
        # absolute path to cli.py, not the script directory.
        file_path = os.path.abspath(__file__)
        repo_base_path = os.path.dirname(
            os.path.dirname(os.path.dirname(file_path)))
        data_path = os.path.join(repo_base_path,
                                 "scripts/test_data/opus-2020-07-17.test.txt")
        if not os.path.isfile:
            raise ValueError(f"need data file {data_path} to exist")
        pipeline.load_dset(data_path)
        return pipeline

    def run_pipeline(self, pipeline: EvaluationPipeline,
                     config: EvaluationPipelineArgs):
        """Abstracts running the pipeline with the given config parameters.
        """
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        pipeline.evaluate(max_samples=config.num_samples_to_test,
                          num_workers=config.num_workers,
                          sentence_bleu=config.sentence_bleu,
                          print_bleu_every=config.print_bleu_every)
        after = datetime.now()
        after_time = after.strftime("%H:%M:%S")
        print("After Evaluation =", after_time)
        duration = (after - now).total_seconds()
        print("Duration: ", str(timedelta(seconds=duration)))

    def read_args(self):
        """Reads the yaml config from CLI and parses them into an object.
        """
        parser = argparse.ArgumentParser(
            prog='model_eval',
            description='CLI for evaluating translation models',
        )

        parser.add_argument('-c',
                            '--config',
                            nargs=1,
                            required=True,
                            help="Path to the config.")
        args = parser.parse_args()
        return args.config[0]

    def parse_config(self, config_path: str):
        """Build arguments from the local config.
        """
        # Read YAML file
        with open(config_path, 'r', encoding="utf-8") as stream:
            data_loaded = yaml.safe_load(stream)

        # Parses config to object
        config = from_dict(data_class=EvaluationConfig, data=data_loaded)
        return config

    def run(self):
        torch.set_num_threads(1)
        # Setup
        config_path = self.read_args()
        config = self.parse_config(config_path)
        print(f"Using configuration {config_path}:\n", config)
        translator = self.initialize_translator(config.quantized)
        pipeline = self.initialize_pipeline(translator)

        # Actually runs the evalation
        self.run_pipeline(pipeline, config.run)