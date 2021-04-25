"""
    date:20210409
    function: 训练一个CNN+transformer，其backbone是Res50，transformer有两层encoder和两层decoder，使用交叉熵损失
                decoder的输入为enconder第四个输入的强度和输出label前三个的拼接
"""

from model.resnet import resnet50
from model.tranformer import Transformer
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from tqdm import tqdm
from model.tranformer import MyTransformer, get_onehot_label, MyCrossEntropyLoss, FocalLoss
from model.bart import MyMseLoss
from config.Parser import Args
import numpy as np

logger = logging.getLogger(__name__)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    print('begin to save model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup(args):
    transformer = MyTransformer(args)
    transformer.to(args.device)
    return transformer


# def change_label_to_vector(y):
def valid(args, model, writer, test_loader, global_step):
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating...(loss=X.X",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    # criterion = MyCrossEntropyLoss()
    # criterion = MyMseLoss(args)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    eval_losses = AverageMeter()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        imgs, intensity, labels = batch
        bs, src_len = labels.size()
        test_target = torch.ones((bs, src_len - 1), dtype=labels.dtype).to(labels.device)
        with torch.no_grad():
            # size = imgs.size()
            for i in range(1, args.src_len + 1):
                outputs = model(imgs, intensity, labels, test_target)  # output[0].size = (B, N, C)
                # outputs = outputs.view(-1, args.class_number)
                # outputs = F.softmax(outputs, dim=-1)
                # labels = labels.view(-1)
                outputs_class = torch.argmax(outputs, dim=-1)
                test_target = outputs_class[:, :src_len - 1]
                # if i < 4:
                #     test_target = torch.cat((outputs_class[:, :i],
                #                              torch.ones((bs, src_len - i - 1), dtype=outputs_class.dtype).to(
                #                                  outputs.device)), dim=-1)
            preds = outputs_class
            outputs = outputs.view(-1, args.class_number)
            labels = labels.view(-1)
            eval_loss = criterion(outputs, labels)
            eval_losses.update(eval_loss)
        # preds = pred.view(args.train_batch_size,-1)
        # if len(all_preds) == 0:
        # print(type(preds.cpu().numpy()))
        all_preds.append(list(preds.cpu().numpy()))
        all_label.append(list(labels.cpu().numpy()))

        global_step += 1
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % (eval_losses.val))
    all_preds, all_label = np.array(all_preds), np.array(all_label)
    accuracy = simple_accuracy(all_preds, all_label)
    # print('val accuracy:', accuracy)
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    writer.add_scalar("test/losses", scalar_value=eval_losses.avg, global_step=global_step)
    return accuracy, eval_losses.avg


def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        # writer = SummaryWriter(log_dir=os.path.join("logs", 'transformer'))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader, test_loader = get_loader(args)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc, best_losses = 0, 0, np.inf
    while True:
        model.train()
        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss()
        # criterion = MyCrossEntropyLoss()
        # criterion = MyMseLoss(args)
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            imgs, intensity, labels = batch
            outputs = model(imgs, intensity, labels)  # output[0].size = (B, N, C)
            outputs = outputs.view(-1, args.class_number)
            # outputs = F.softmax(outputs, dim=-1)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.cpu())
            )
            if args.local_rank in [-1, 0]:
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
            if global_step % args.eval_every == 0:
                accuracy, eval_losses = valid(args, model, writer, test_loader, global_step)
                if eval_losses <= best_losses:
                    save_model(args, model)
                    best_losses = eval_losses
                model.train()

            if global_step % t_total == 0:
                break
        losses.reset()
        if global_step % t_total == 0:
            break
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


if __name__ == '__main__':
    args = Args
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model = setup(args)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))
    # Set seed
    set_seed(args)
    train(args, model)
    print('End Training!')
