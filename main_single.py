import argparse
import importlib
import os
import sys
import shutil
import json
import tensorflow as tf
from time import gmtime, strftime

src_dir = './src_single'


def launch_training(**kwargs):

    # Deal with file and paths
    appendix = kwargs["resume_from"]
    if appendix is None or appendix == '':
        cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        log_dir = './log_skgan_' + cur_time
        ckpt_dir = './ckpt_skgan_' + cur_time
        if not os.path.isdir(log_dir) and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.isdir(ckpt_dir) and not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # copy current script in src folder to log dir for record
        if not os.path.exists(src_dir) or not os.path.isdir(src_dir):
            print("src folder does not exist.")
            return
        else:
            for file in os.listdir(src_dir):
                if file.endswith(".py"):
                    shutil.copy(os.path.join(src_dir, file), log_dir)

        kwargs['log_dir'] = log_dir
        kwargs['ckpt_dir'] = ckpt_dir
        appendix = cur_time
        kwargs["resume_from"] = appendix
        kwargs["iter_from"] = 0

        # Save parameters
        with open(os.path.join(log_dir, 'param_%d.json' % 0), 'w') as fp:
            json.dump(kwargs, fp, indent=4)

        sys.path.append(src_dir)
        entry_point_module = kwargs['entry_point']
        from config import Config
        Config.set_from_dict(kwargs)

        print("Launching new train: %s" % cur_time)
    else:
        if len(appendix.split('-')) != 6:
            print("Invalid resume folder")
            return

        log_dir = './log_skgan_' + appendix
        ckpt_dir = './ckpt_skgan_' + appendix

        # Get last parameters (recover entry point module name)
        json_files = [f for f in os.listdir(log_dir) if
                      os.path.isfile(os.path.join(log_dir, f)) and os.path.splitext(f)[1] == '.json']
        iter_starts = max([int(os.path.splitext(filename)[0].split('_')[1]) for filename in json_files])
        with open(os.path.join(log_dir, 'param_%d.json' % iter_starts), 'r') as fp:
            params = json.load(fp)
        entry_point_module = params['entry_point']

        # Recover parameters
        _ignored = ['num_gpu', 'iter_from']
        for k, v in params.items():
            if k not in _ignored:
                kwargs[k] = v

        sys.path.append(log_dir)

        # Get latest checkpoint filename
        # if stage == 1:
        #     ckpt_file = tf.train.latest_checkpoint(stage_1_ckpt_dir)
        # elif stage == 2:
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file is None:
            raise RuntimeError
        else:
            iter_from = int(os.path.split(ckpt_file)[1].split('-')[1]) + 1
        kwargs['log_dir'] = log_dir
        kwargs['ckpt_dir'] = ckpt_dir
        kwargs['iter_from'] = iter_from

        # Save new set of parameters
        with open(os.path.join(log_dir, 'param_%d.json' % iter_from), 'w') as fp:
            kwargs['entry_point'] = entry_point_module
            json.dump(kwargs, fp, indent=4)

        from config import Config
        Config.set_from_dict(kwargs)
        print("Launching train from checkpoint: %s" % appendix)

    # Launch train
    train_module = importlib.import_module(entry_point_module)
    # from train_paired_aug_multi_gpu import train
    status = train_module.train(**kwargs)

    return status, appendix


def launch_test(**kwargs):
    # Deal with file and paths
    appendix = kwargs["resume_from"]
    if appendix is None or appendix == '' or len(appendix.split('-')) != 6:
        print("Invalid resume folder")
        return

    log_dir = './log_skgan_' + appendix
    ckpt_dir = './ckpt_skgan_' + appendix

    sys.path.append(log_dir)

    # Get latest checkpoint filename
    kwargs['log_dir'] = log_dir
    kwargs['ckpt_dir'] = ckpt_dir

    # Get last parameters (recover entry point module name)
    # Assuming last json file
    json_files = [f for f in os.listdir(log_dir) if
                  os.path.isfile(os.path.join(log_dir, f)) and os.path.splitext(f)[1] == '.json']
    iter_starts = max([int(os.path.splitext(filename)[0].split('_')[1]) for filename in json_files])
    with open(os.path.join(log_dir, 'param_%d.json' % iter_starts), 'r') as fp:
        params = json.load(fp)
    entry_point_module = params['entry_point']

    # Recover parameters
    _ignored = ["num_gpu", 'iter_from']
    for k, v in params.items():
        if k not in _ignored:
            kwargs[k] = v

    from config import Config
    Config.set_from_dict(kwargs)
    print("Launching test from checkpoint: %s" % appendix)

    # Launch test
    train_module = importlib.import_module(entry_point_module)
    train_module.test(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or Test model')
    parser.add_argument('--mode', type=str, default="train", help="train or test")
    parser.add_argument('--resume_from', type=str, default='', help="Whether resume last checkpoint from a past run. Notice: you only need to fill in the string after skgan_, i.e. the part with yyyy-mm-dd-hr-min-sec")
    parser.add_argument('--entry_point', type=str, default='train_single', help="name of the training .py file")
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per gpu')
    parser.add_argument('--max_iter_step', default=300000, type=int, help="Max number of iterations")
    parser.add_argument('--disc_iterations', default=1, type=int, help="Number of discriminator iterations")
    parser.add_argument('--ld', default=10, type=float, help="Gradient penalty lambda hyperparameter")
    parser.add_argument('--optimizer', type=str, default="Adam", help="Optimizer for the graph")
    parser.add_argument('--lr_G', type=float, default=2e-4, help="learning rate for the generator")
    parser.add_argument('--lr_D', type=float, default=4e-4, help="learning rate for the discriminator")
    parser.add_argument('--num_gpu', default=2, type=int, help="Number of GPUs to use")
    parser.add_argument('--distance_map', default=1, type=int, help="Whether using distance maps for sketches")
    parser.add_argument('--small_img', default=1, type=int, help="Whether using 64x64 instead of 256x256")
    parser.add_argument('--extra_info', default="", type=str, help="Extra information saved for record")

    args = parser.parse_args()

    assert args.optimizer in ["RMSprop", "Adam", "AdaDelta", "AdaGrad"], "Unsupported optimizer"

    # Set default params
    d_params = {"resume_from": args.resume_from,
                "entry_point": args.entry_point,
                "batch_size": args.batch_size,
                "max_iter_step": args.max_iter_step,
                "disc_iterations": args.disc_iterations,
                "ld": args.ld,
                "optimizer": args.optimizer,
                "lr_G": args.lr_G,
                "lr_D": args.lr_D,
                "num_gpu": args.num_gpu,
                "distance_map": args.distance_map,
                "small_img": args.small_img,
                "extra_info": args.extra_info,
                }

    if args.mode == 'train':
        # Launch training
        status, appendix = launch_training(**d_params)
        while status == -1:  # NaN during training
            print("Training ended with status -1. Restarting..")
            d_params["resume_from"] = appendix
            status = launch_training(**d_params)
    elif args.mode == 'test':
        launch_test(**d_params)
