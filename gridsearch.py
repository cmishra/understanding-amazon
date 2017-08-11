import subprocess
import numpy as np

ROOT_COMMANDS = "jbsub -queue x86_6h -cores 1+1 -mem 20G -proj understanding_amazon python3 runs/train.py "

most_args = {
    "data": "data_processed",
    "cache_filepath": "epoch_cache",
    "model": "alexnet",
    "workers": "1",
    "epochs": "10",
    "batch_size": "64",
    "optimizer": "adam",
    "momentum": "0.0",
    "lr_schedule": "Constant",
    "lr": "0.01",
    "weight_decay": "0.0001",
    "criterion": "mse",
}


def args_to_command(args):
    cmd = ROOT_COMMANDS
    for k, v in args.items():
        cmd += "--%s %s " % (k, v)
    return cmd


if __name__ == "__main__":
    lr_candidates = np.random.uniform(-5, -1, 2)
    weight_decay_candidates = np.random.uniform(-6, -2, 2)
    for c in ["mse", "l1"]:
        for lr in lr_candidates:
            for w in weight_decay_candidates:
                args = most_args.copy()
                args["lr"] = 10 ** lr
                args["weight_decay"] = 10 ** w
                args["criterion"] = c
                subprocess.run(args_to_command(args), shell=True)

