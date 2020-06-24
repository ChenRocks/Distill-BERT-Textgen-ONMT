#!/usr/bin/env python
"""Train models."""
import json
import os
from os.path import abspath, dirname, join
import signal
import subprocess
import torch

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.logging import logger
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        mp = torch.multiprocessing.get_context('spawn')
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, ), daemon=False))  # TODO check this change
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def run(opt, device_id, error_queue):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=False)  # TODO check this change
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    # BERT KD options
    parser.add_argument('--bert_kd', action='store_true',
                        help='use BERT KD for training')
    # BERT model
    parser.add_argument("--bert_dump",
                        help="path to BERT model dump")
    # KD hyper params
    parser.add_argument("--kd_alpha", default=0.5, type=float,
                        help="ratio between label and teacher loss")
    parser.add_argument("--kd_temperature", default=10.0, type=float,
                        help="temperature of teacher logits")
    parser.add_argument("--kd_topk", default=-1, type=int,
                        help="to use only topk teacher logits (-1: all)")
    # special preprocessed DB
    parser.add_argument("--data_db", default=None, type=str,
                        help="path to shelve DB (used for BERT KD only)")
    parser.add_argument('--local_rank', default=-1, type=int)

    opt = parser.parse_args()

    # check for BERT KD
    if opt.bert_kd:
        assert opt.data_db, opt.bert_dump
        assert opt.batch_type == 'tokens'
        assert opt.normalization == 'tokens'

    # make output dir
    if opt.local_rank == -1 or opt.local_rank == 0:
        exp_root = opt.save_model
        os.makedirs(join(exp_root, 'log'))
        os.makedirs(join(exp_root, 'ckpt'))
        opt.save_model = join(exp_root, 'ckpt', 'model')
        opt.log_file = join(exp_root, 'log', 'log')
        opt.tensorboard_log_dir = join(exp_root, 'log')
        with open(join(exp_root, 'log', 'hps.json'), 'w') as writer:
            json.dump(vars(opt), writer, indent=4)

        # git info
        try:
            logger.info("Waiting on git info....")
            c = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                               timeout=10, stdout=subprocess.PIPE)
            git_branch_name = c.stdout.decode().strip()
            logger.info("Git branch: %s", git_branch_name)
            c = subprocess.run(["git", "rev-parse", "HEAD"],
                               timeout=10, stdout=subprocess.PIPE)
            git_sha = c.stdout.decode().strip()
            logger.info("Git SHA: %s", git_sha)
            git_dir = abspath(dirname(__file__))
            git_status = subprocess.check_output(
                ['git', 'status', '--short'],
                cwd=git_dir, universal_newlines=True).strip()
            with open(join(exp_root, 'log', 'git_info.json'), 'w') as writer:
                json.dump({'branch': git_branch_name,
                           'is_dirty': bool(git_status),
                           'status': git_status,
                           'sha': git_sha},
                          writer, indent=4)
        except subprocess.TimeoutExpired as e:
            logger.exception(e)
            logger.warn("Git info not found. Moving right along...")

    main(opt)
