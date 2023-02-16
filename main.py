import argparse

from processing.process import process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        type=str,
        help="Specify a task: preprocess/train/play",
    )

    args = parser.parse_args()
    if args.task == "preprocess":
        process(0, 0)


if __name__ == "__main__":
    process(135800, 136000)
