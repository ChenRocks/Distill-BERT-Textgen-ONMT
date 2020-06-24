#!/usr/bin/env python
"""Training on a single process."""
import os

import torch
from torch.utils.data import DataLoader

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.inputters.bert_kd_dataset import (
    BertKdDataset, TokenBucketSampler, DistributedTokenBucketSampler)


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def cycle_loader(loader, device):
    """ yield batches indefinitely from the loader"""
    while True:
        for batch in loader:
            # NOTE this is an adhoc solution
            batch.src = (batch.src[0].to(device), batch.src[1].to(device))
            batch.tgt = batch.tgt.to(device)
            logit, indices = batch.bert_topk
            batch.bert_topk = (logit.to(device), indices.to(device))
            yield batch


def main(opt, device_id):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device_id = opt.local_rank
        world_size = torch.distributed.get_world_size()
    else:
        if device_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", device_id)
    if opt.local_rank > 0:
        logger.disabled = True
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)

    if opt.bert_kd:
        src_vocab = vocab['src'].fields[0][1].vocab.stoi
        tgt_vocab = vocab['tgt'].fields[0][1].vocab.stoi
        assert 0 < opt.kd_topk <= 128
        train_dataset = BertKdDataset(opt.data_db, opt.bert_dump,
                                      src_vocab, tgt_vocab,
                                      max_len=150, k=opt.kd_topk)
        BUCKET_SIZE = 8192
        if True or opt.local_rank == -1 and opt.world_size == 1:
            train_sampler = TokenBucketSampler(
                train_dataset.keys, BUCKET_SIZE, opt.batch_size,
                batch_multiple=1)
        else:
            assert False  # seems like it's handled in training loop
            train_sampler = DistributedTokenBucketSampler(
                world_size, device_id,
                train_dataset.keys, BUCKET_SIZE, opt.batch_size,
                batch_multiple=1)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=4,
                                  collate_fn=BertKdDataset.pad_collate)
        train_iter = cycle_loader(train_loader, device)
    else:
        train_iter = build_dataset_iter("train", fields, opt)
    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if opt.tensorboard:
        if trainer.report_manager.tensorboard_writer:
            trainer.report_manager.tensorboard_writer.close()
