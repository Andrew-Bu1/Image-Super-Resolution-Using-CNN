import models
import argparse
import os


def check_valid_file(path):
    if (os.path.isfile(path)):
        return path
    else:
        raise argparse.ArgumentError(
            "This path {:s} is not a valid file, check again.".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main argument parser")
    parser.add_argument("run_mode", choices=(
        "train", "eval", "infer"), help="Main running mode of the program")
    parser.add_argument("--model", type=str, choices=models.AvailableModels.keys(),
                        help="The type of model to be ran")
    parser.add_argument("--model_dir", type=str,
                        required=True, help="Location of model")
    parser.add_argument("--config", type=str, nargs="+",
                        default=None, help="Location of the config file")
    # arguments for inference
    parser.add_argument("--features_file", type=str,
                        help="Inference mode: Provide the location of features file")
    parser.add_argument("--predictions_file", type=str,
                        help="Inference mode: Provide Location of output file which is predicted from features file")
