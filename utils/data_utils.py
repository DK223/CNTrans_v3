import logging
import argparse
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
from model.resnet import resnet50

logger = logging.getLogger(__name__)


def default_loader(paths):
    images = []
    for path in paths:
        image = Image.open(path).convert('RGB')
        images.append(image)
    return images  # 对于彩色图像返回RGB,对于灰度图像模式为L


class MyDataset(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt, mulu, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        imgs = []
        # img0 = []
        intensity2index = {}
        index2intensity = {}
        l2 = [10, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 27, 28, 30, 32, 33, 35, 38, 40, 42, 43, 45, 48, 50,
              51, 52, 53, 55, 57, 58, 60, 62, 65, 68, 70, 72]
        for i, intensity in enumerate(l2):
            intensity2index[intensity] = i
            index2intensity[i] = intensity
        # l3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        #       29, 30, 31, 32, 33, 34, 35, 36, 37]
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            # line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()
            words[0] = './Data/' + str(mulu) + '/' + str(words[0]) + '.JPG'
            words[2] = './Data/' + str(mulu) + '/' + str(words[2]) + '.JPG'
            words[4] = './Data/' + str(mulu) + '/' + str(words[4]) + '.JPG'
            words[6] = './Data/' + str(mulu) + '/' + str(words[6]) + '.JPG'
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append(([words[0], words[2], words[4], words[6]],
                         # 台风图片对应的强度
                         [float(intensity2index[int(words[1])]),
                          float(intensity2index[int(words[3])]),
                          float(intensity2index[int(words[5])]),
                          float(intensity2index[int(words[7])])],
                         # 后四个时刻的台风强度
                         [float(intensity2index[int(words[8])]),
                          float(intensity2index[int(words[9])]),
                          float(intensity2index[int(words[10])]),
                          float(intensity2index[int(words[11])])]))
            # 根据原本txt的内容，words[0]是图片信息，words[4]是label
        # for i in range(len(l2)):
        # for j in range(len(img0)):
        #     intensity = img0[j][1]
        #     index = l2.index(intensity)
        #     imgs.append((img0[j][0:4], float(l3[index])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, intensity, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[4]的信息
        # print(fn)
        # img = Image.open(fn)
        intensity = torch.tensor(intensity).long()
        label = torch.tensor(label).long()
        imgs = self.loader(fn)  # 按照路径读取图片

        # imgs_tensor = []
        imgs_list = []
        if self.transform is not None:
            for img in imgs:
                img = self.transform(img).unsqueeze(0)  # 数据标签转换为Tensor
                imgs_list.append(img)
        imgs_tensor = torch.cat(imgs_list, dim=0)
        # label_tensor = torch.cat(label, dim=0)
        return imgs_tensor, intensity, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# def get_loader(args):
def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 导入训练集并进行预处理
    trainset = MyDataset(txt='./Data/train.txt', mulu='img_05-17',
                         transform=transform_train)
    train_num = len(trainset)
    print(train_num)

    # 导入测试集并进行预处理
    testset = MyDataset(txt='./Data/test.txt', mulu='img_2018',
                        transform=transform_test)
    test_num = len(testset)

    # print(test_num)
    # if args.local_rank == 0:
    #     torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              # sampler=train_sampler,
                              shuffle=False,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              # drop_last=False,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             # sampler=test_sampler,
                             shuffle=False,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             # drop_last=False,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    args = parser.parse_args()
    train_loader = get_loader(args)
    resnet = resnet50(args, pretrained=True)

    print('zzzz')
