from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()

    parser.add_argument(
        "-m", "--model",
        help="choose one of models from lenet, alexnet, and shallow",
        default="shallow",
        choices=["lenet", "alexnet", "shallow"])
    parser.add_argument(
        "-l", "--log",
        default="local",
        choices=["local", "wandb"]
    )
    parser.add_argument(
        "-a", "--accelerator",
        default="cpu",
        choices=["cpu","gpu"]
    )
    parser.add_argument(
        "--devices",
        default=None
    )    
    args = parser.parse_args()

    return args.model, args.log, args.accelerator, args.devices
