import argparse

from training.process import process, process_zero
from training import sl_policy, rl_policy, sl_resnet
from training.evaluate import policy_evaluate, plot_policy_curves
from training.eval_resnet import resnet_eval
from networks.policy import Conv256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        type=str,
        help="Specify a task: preprocess/train/play",
    )

    args = parser.parse_args()
    if args.task == "process":
        process(161200, 162000)
    elif args.task == "processz":
        process_zero(160000, 200000)
    elif args.task == "slpolicy":
        sl_policy.train(Conv256, False)
    elif args.task == "eval":
        # policy_evaluate("./checkpoints/checkpoint_0_100.pth")
        resnet_eval("./checkpoints/checkpoint_0_125.pth")
    elif args.task == "plot":
        plot_policy_curves()
    elif args.task == "rl":
        rl_policy.train()
    elif args.task == "res":
        sl_resnet.train(resume=True)


if __name__ == "__main__":
    main()
