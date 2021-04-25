from model.resnet import resnet50
from model.tranformer import Transformer
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
from torch.utils.tensorboard import SummaryWriter
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from tqdm import tqdm
from model.bart import MyBart, BartConfig
import numpy as np
from model.tranformer import MyTransformer
from config.Parser import Args


logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    # return transformer

    # transformer = MyBart(config, args)
    pthfile = './output/CNTrans_2_2_50_MyCE02_checkpoint.pth'
    # state_dict = torch.load(pthfile, map_location=torch.device('cpu'))
    transformer.load_state_dict(torch.load(pthfile, map_location=args.device))
    transformer.to(args.device)
    return transformer


# def change_label_to_vector(y):
def valid(args, model, test_loader, global_step):
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating...(loss=X.X",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    eval_losses = AverageMeter()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        size = x.size()
        decoder_input_embeds = torch.zeros((size[0], 1, args.d_model)).to(args.device)
        outputs = model(x, decoder_input_embeds=decoder_input_embeds)
        outputs = outputs[0].view(-1, args.class_number)
        y = y.view(-1)
        # outputs = torch.argmax(outputs, dim=-1).to(torch.float)
        # outputs = outputs.requires_grad = True
        eval_loss = criterion(outputs, y)
        eval_losses.update(eval_loss.item())
        preds = outputs.argmax(-1)
        # preds = pred.view(args.train_batch_size,-1)
        # if len(all_preds) == 0:
        # print(type(preds.cpu().numpy()))
        all_preds.append(list(preds.cpu().numpy()))
        all_label.append(list(y.cpu().numpy()))

        global_step += 1
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    all_preds, all_label = np.array(all_preds), np.array(all_label)
    accuracy = simple_accuracy(all_preds, all_label)
    print('val accuracy:', accuracy)
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    # logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    return accuracy


if __name__ == '__main__':
    args = Args
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model = setup(args)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
    #                (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    # Set seed
    set_seed(args)
    mse0, mse1, mse2, mse3 = 0, 0, 0, 0
    mae0, mae1, mae2, mae3 = 0, 0, 0, 0
    pre = []
    lab = []
    acc = 0.0
    l2 = [10, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 27, 28, 30, 32, 33, 35, 38, 40, 42, 43, 45, 48, 50,
          51, 52, 53, 55, 57, 58, 60, 62, 65, 68, 70, 72]
    l3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37]
    all_preds, all_label = [], []
    train_loader, test_loader = get_loader(args)
    epoch_iterator = tqdm(test_loader,
                          desc="Validating...(loss=X.X",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # f = open('./pred_result.txt', 'a')
    test_num = 0
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
            preds = outputs_class.view(-1)
            outputs = outputs.view(-1, args.class_number)
            labels = labels.view(-1)

        test_num += len(labels)
        batch = len(labels) / 4
        for i in range(len(labels)):
            # f.write(str(l2[int(preds[i])]) + '\n')
            if i % 4 == 0:
                mse0 += np.square(l2[int(labels[i])] - l2[int(preds[i])])
                mae0 += np.abs(l2[int(labels[i])] - l2[int(preds[i])])
            elif i % 4 == 1:
                mse1 += np.square(l2[int(labels[i])] - l2[int(preds[i])])
                mae1 += np.abs(l2[int(labels[i])] - l2[int(preds[i])])
            elif i % 4 == 2:
                mse2 += np.square(l2[int(labels[i])] - l2[int(preds[i])])
                mae2 += np.abs(l2[int(labels[i])] - l2[int(preds[i])])
            else:
                mse3 += np.square(l2[int(labels[i])] - l2[int(preds[i])])
                mae3 += np.abs(l2[int(labels[i])] - l2[int(preds[i])])
            # print(l2[int(y[i])])
        # print(preds)
    # f.close()
    print(test_num)
    test_num = test_num / 4
    rmse0 = np.sqrt(mse0 / test_num)
    rmse1 = np.sqrt(mse1 / test_num)
    rmse2 = np.sqrt(mse2 / test_num)
    rmse3 = np.sqrt(mse3 / test_num)
    mae0 = mae0 / test_num
    mae1 = mae1 / test_num
    mae2 = mae2 / test_num
    mae3 = mae3 / test_num
    print('rmse0:', rmse0)
    print('mae0:', mae0)
    print('rmse1:', rmse1)
    print('mae1:', mae1)
    print('rmse2:', rmse2)
    print('mae2:', mae2)
    print('rmse3:', rmse3)
    print('mae3:', mae3)
    print('rmse:', np.sqrt((mse0 + mse1 + mse2 + mse3) / (4 * test_num)))
    print('mae:', (mae0 + mae1 + mae2 + mae3) / 4)
    print('zzz')
