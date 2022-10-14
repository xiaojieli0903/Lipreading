import datetime
import json
import logging
import os
import shutil

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg

SAMPLERS = Registry('sampler')


def calculateNorm2(model):
    para_norm = 0.
    for p in model.parameters():
        para_norm += p.data.norm(2)
    print('2-norm of the neural network: {:.4f}'.format(para_norm**.5))


def showLR(optimizer):
    return optimizer.param_groups[0]['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile(
        filepath
    ), "Error when trying to read txt file, path does not exist: {}".format(
        filepath)
    with open(filepath) as myfile:
        content = myfile.read().splitlines()
    return content


def save_as_json(d, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def load_json(json_fp):
    assert os.path.isfile(
        json_fp
    ), "Error loading JSON. File provided does not exist, cannot read: {}".format(
        json_fp)
    with open(json_fp, 'r') as f:
        json_content = json.load(f)
    return json_content


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)


# -- checkpoints
class CheckpointSaver:
    def __init__(self,
                 save_dir,
                 checkpoint_fn='ckpt.pth',
                 best_fn='ckpt.best.pth',
                 best_step_fn='ckpt.best.step{}.pth',
                 save_best_step=False,
                 lr_steps=[]):
        """
        Only mandatory: save_dir
            Can configure naming of checkpoint files through checkpoint_fn, best_fn and best_stage_fn
            If you want to keep best-performing checkpoint per step
        """

        self.save_dir = save_dir

        # checkpoint names
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.best_step_fn = best_step_fn

        # save best per step?
        self.save_best_step = save_best_step
        self.lr_steps = []

        # init var to keep track of best performing checkpoint
        self.current_best = 0

        # save best at each step?
        if self.save_best_step:
            assert lr_steps != [], "Since save_best_step=True, need proper value for lr_steps. Current: {}".format(
                lr_steps)
            self.best_for_stage = [0] * (len(lr_steps) + 1)

    def save(self, save_dict, current_perf, epoch=-1):
        """
            Save checkpoint and keeps copy if current perf is best overall or [optional] best for current LR step
        """

        # save last checkpoint
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)

        # keep track of best model
        self.is_best = current_perf > self.current_best
        if self.is_best:
            self.current_best = current_perf
            best_fp = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_prec'] = self.current_best

        # keep track of best-performing model per step [optional]
        if self.save_best_step:

            assert epoch >= 0, "Since save_best_step=True, need proper value for 'epoch'. Current: {}".format(
                epoch)
            s_idx = sum(epoch >= l for l in lr_steps)
            self.is_best_for_stage = current_perf > self.best_for_stage[s_idx]

            if self.is_best_for_stage:
                self.best_for_stage[s_idx] = current_perf
                best_stage_fp = os.path.join(self.save_dir,
                                             self.best_stage_fn.format(s_idx))
            save_dict['best_prec_per_stage'] = self.best_for_stage

        # save
        torch.save(save_dict, checkpoint_fp)
        print("Checkpoint saved at {}".format(checkpoint_fp))
        if self.is_best:
            shutil.copyfile(checkpoint_fp, best_fp)
        if self.save_best_step and self.is_best_for_stage:
            shutil.copyfile(checkpoint_fp, best_stage_fp)

    def set_best_from_ckpt(self, ckpt_dict):
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage', None)


def load_model(load_path, model, optimizer=None, allow_size_mismatch=False):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile(
        load_path
    ), "Error when loading the model, provided path not found: {}".format(
        load_path)
    checkpoint = torch.load(load_path)
    loaded_state_dict = checkpoint['model_state_dict']

    if allow_size_mismatch:
        loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
        model_state_dict = model.state_dict()
        model_sizes = {k: v.shape for k, v in model_state_dict.items()}
        mismatched_params = []
        for k in loaded_sizes:
            if loaded_sizes[k] != model_sizes[k]:
                mismatched_params.append(k)
        for k in mismatched_params:
            del loaded_state_dict[k]

    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch_idx'], checkpoint
    return model


# -- logging utils
def get_logger(args, save_path):
    log_path = '{}/{}_{}_{}classes_log.txt'.format(save_path,
                                                   args.training_mode, args.lr,
                                                   args.num_classes)
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s]'
        '[line:%(lineno)4d][%(levelname)8s]%(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def update_logger_batch(args, logger, dset_loader, batch_idx, running_loss,
                        loss_dict, loss_weight, running_corrects, running_all,
                        batch_time, data_time, lr, mem, global_iter, dataset_num):
    perc_epoch = 100. * batch_idx / (len(dset_loader) - 1)
    all_iters = args.epochs * len(dset_loader)
    eta = batch_time.avg * (all_iters - global_iter) / 3600

    logger.info(
        f"[{global_iter}/{all_iters} | {running_all * args.world_size:5.0f}/{dataset_num:5.0f} | "
        f"{batch_idx:5.0f}/{len(dset_loader.loader):5.0f} ({perc_epoch:.0f}%)] | "
        f"Loss: {running_loss / running_all:.4f} | Acc:{running_corrects / running_all:.4f} | "
        f"Cost time:{batch_time.val:1.3f} ({batch_time.avg:1.3f})s | "
        f"lr:{lr:.6f} | "
        f"Mem:{mem:.3f}M | "
        f"Data time:{data_time.val:1.3f} ({data_time.avg:1.3f}) | "
        f"Instances per second: {args.batch_size*args.world_size/batch_time.avg:.2f} | ETA: {eta:.2f}h"
    )
    for key in loss_dict:
        logger.info(
            f"-----{key}: {loss_weight[key]:.4f} * {loss_dict[key].item():.4f}"
        )


def get_save_folder(args):
    # create save and log folder
    save_path = '{}/{}/{}'.format(
        args.logging_dir, args.training_mode,
        args.config_path.split('/')[-1].replace('.json', '') + args.exp_name)
    save_path += '/' + datetime.datetime.now().isoformat().split('.')[0]
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    return save_path


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.

    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_items = next(self.loader)
        except StopIteration:
            if isinstance(self.next_items, dict):
                self.next_items = {key: None for key in self.next_items.keys()}
            elif isinstance(self.next_items, list):
                self.next_items = [None for item in self.next_items]
            return self.next_items
        except:
            raise RuntimeError('load data error')

        with torch.cuda.stream(self.stream):
            self.next_items = self.tocuda(self.next_items)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_items = self.next_items
        self.preload()
        return next_items

    def tocuda(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda(non_blocking=True)
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, list):
            return [self.tocuda(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self.tocuda(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return tuple([self.tocuda(i) for i in obj])

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        return self.next()
