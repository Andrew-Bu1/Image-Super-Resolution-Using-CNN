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

    args = parser.parse_args()
    # create directory if not exist
    os.makedirs(args.model_dir, exist_ok=True)

    # load model. Specific run mode required converting
    model = models.AvailableModels[args.model](
        model_dir=args.model_dir, mode=args.run_mode)
    # run model
    run_mode = args.run_mode
    if (run_mode == "train"):
        model.run_train(model_dir=args.model_dir)
    elif (run_mode == "eval"):
        model.run_eval(model_dir=args.model_dir)
    elif (run_mode == "infer"):
        model.run_infer(args.features_file,
                        args.predictions_file)
    else:
        raise ValueError("Run mode {:s} not implemented.".format(run_mode))
