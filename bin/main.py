import models
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main argument parser")
    parser.add_argument("run_mode", choices=(
        "train", "eval", "infer"), help="Main running mode of the program")
    parser.add_argument("--model", type=str, choices=models.AvailableModels.keys(),
                        help="The type of model to be ran")
    parser.add_argument("--model_dir", type=str,
                        required=True, help="Location of model")
