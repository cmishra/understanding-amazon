
import subprocess
import numpy as np

ROOT_COMMANDS = "jbsub -queue x86_6h -cores 1+1 -mem 20G -proj understanding_amazon python3 runs/train.py "

most_args = {
    "data": "data_processed",
    "cache_filepath": "epoch_cache",
    "pretrained": "true",
    "model": "alexnet",
    "workers": "1",
    "epochs": "30",
    "batch_size": "64",
    "optimizer": "adam",
    "momentum": "0.0",
    "lr_schedule": "Constant",
    "lr": "0.01",
    "weight_decay": "0.0001",
    "criterion": "mse",
}


def args_to_command(args):
    debug = True
    cmd = ROOT_COMMANDS
    for k, v in args.items():
        cmd += "--%s %s " % (k, v)
    if debug:
        cmd += "--debug "
    return cmd


if __name__ == "__main__":
    num_runs = 5
    for i in range(0, num_runs):
        args = most_args.copy()
        args["lr"] = 10 ** np.random.uniform(-5, -1, 1)[0]
        args["weight_decay"] = 10 ** np.random.uniform(-6, -2, 1)[0]
        args["criterion"] = "mse" if np.random.random() > 0.5 else "l1"
        subprocess.run(args_to_command(args), shell=True)

