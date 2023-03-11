import argparse

from training.process import process
from training.policy_train import train
from training.evaluate import policy_evaluate, plot_policy_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        type=str,
        help="Specify a task: preprocess/train/play",
    )

    args = parser.parse_args()
    if args.task == "preprocess":
        process(0, 160000)
    elif args.task == "train":
        train(resume=False)
    elif args.task == "eval":
        policy_evaluate("./checkpoints/checkpoint_0_100.pth")
    elif args.task == "plot":
        plot_policy_curves()


if __name__ == "__main__":
    main()
