import os
import torch
import torch.distributed
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as t_F
from torch.nn import functional as F
from torchvision.datasets.folder import ImageFolder
from torch.nn.modules import loss
from torchvision.transforms import InterpolationMode
import random
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict

class Soft_CrossEntropy(loss._Loss):
    def forward(self, model_output, soft_output):

        size_average = True

        model_output_log_prob = F.log_softmax(model_output, dim=1)

        soft_output = soft_output.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        cross_entropy_loss = -torch.bmm(soft_output, model_output_log_prob)
        if size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()

        return cross_entropy_loss


class RandomResizedCrop_FKD(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCrop_FKD, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        i = coords[0].item() * img.size[1]
        j = coords[1].item() * img.size[0]
        h = coords[2].item() * img.size[1]
        w = coords[3].item() * img.size[0]

        if self.interpolation == 'bilinear':
            inter = InterpolationMode.BILINEAR
        elif self.interpolation == 'bicubic':
            inter = InterpolationMode.BICUBIC
        return t_F.resized_crop(img, i, j, h, w, self.size, inter)


class RandomHorizontalFlip_FKD(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, coords, status):
    
        if status == True:
            return t_F.hflip(img)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Compose_FKD(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(Compose_FKD, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCrop_FKD':
                img = t(img, coords, status)
            elif type(t).__name__ == 'RandomCrop_FKD':
                img, coords = t(img)
            elif type(t).__name__ == 'RandomHorizontalFlip_FKD':
                img = t(img, coords, status)
            else:
                img = t(img)
        return img


def compute_iou(bbox1, bbox2):
    # 获取bbox点坐标
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 计算交集(bbox交集的左上角坐标和右下角坐标)
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # 计算交集面积，若无交集，面积为0
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # 计算并集面积
    bbox1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    bbox2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    union_area = bbox1_area + bbox2_area - inter_area

    # 计算交并比
    iou = inter_area / union_area
    return iou
def calculate_entropy(max_values):
    """
    计算特定构成的1000维向量的熵。
    假设除了5个最大值外，其余995个值是相同的，且向量元素和为1.0。

    参数:
        max_values (list or numpy.ndarray): 概率向量中5个最大值的列表。

    返回:
        entropy (float): 给定概率分布的熵。
    """
    # 计算5个最大值的总和
    sum_max_values = sum(max_values)

    # 假设其余的995个值平均分配剩余的概率
    remaining_value = (1.0 - sum_max_values) / 995

    # 计算熵, 熵的计算公式是：-∑(p(x) * log(p(x)))，其中p(x)是概率分布
    # 对于5个最大值
    entropy_max_values = -np.sum(max_values * np.log(max_values))

    # 对于其余的995个值。因为这些值都相同，所以可以直接计算一个值的熵，然后乘以995
    entropy_remaining = -995 * remaining_value * np.log(remaining_value)

    # 总熵是两部分的和
    total_entropy = entropy_max_values + entropy_remaining

    return total_entropy

class ImageFolder_FKD(torchvision.datasets.ImageFolder):
    def __init__(self, **kwargs):
        self.num_crops = kwargs['num_crops']
        self.softlabel_path = kwargs['softlabel_path']
        kwargs.pop('num_crops')
        kwargs.pop('softlabel_path')
        super(ImageFolder_FKD, self).__init__(**kwargs)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.my_transform_common_1 = transforms.Compose([
            # transforms.Resize(256),

            transforms.RandomVerticalFlip(p = 0.3),
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomRotation(),
            # transforms.ToTensor(),
            # normalize,
        ])

        self.my_transform_common_2 = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def __getitem__(self, index):
            path, target = self.samples[index]

            label_path = os.path.join(self.softlabel_path, '/'.join(path.split('/')[-3:]).split('.')[0] + '.tar')

            label = torch.load(label_path, map_location=torch.device('cpu'))

            coords, flip_status, output = label

            rand_index = torch.randperm(len(output))
            soft_target = []
            no_confi = []
            num_classes = 1000
            threshold = np.log2(num_classes) * 0.5
            sample = self.loader(path)
            # sample_ori = self.my_transform(sample)
            # sample_ori_common = transforms.RandomResizedCrop(224)(sample)
            # sample_ori_weak = self.my_transform_common_2(sample_ori_common)
            # sample_ori_v2 = self.my_transform_common_1(sample_ori_common)
            # sample_ori_stronger = self.my_transform_common_2(sample_ori_v2)
            sample_all = []
            hard_target = []
            bbox = []
            for i in range(len(output)):
                tmp_soft = output[rand_index[i]]
                entropys = calculate_entropy(tmp_soft[1])
                # is_low_confidence = entropys > threshold
                if entropys > threshold:
                    continue
                if len(soft_target) == self.num_crops - 1 and self.num_crops > 1:
                    break
                if self.transform is not None:
                    soft_target.append(output[rand_index[i]])
                    coord = coords[rand_index[i]]
                    x1 = coord[0].item() * sample.size[1]
                    y1 = coord[1].item() * sample.size[0]
                    x2 = coord[2].item() * sample.size[1] + x1
                    y2 = coord[3].item() * sample.size[0] + y1
                    bbox.append((x1, y1, x2, y2))
                    sample_trans = self.transform(sample, coord, flip_status[rand_index[i]])
                    sample_all.append(sample_trans)
                    hard_target.append(target)
                else:
                    coords = None
                    flip_status = None
                if self.target_transform is not None:
                    target = self.target_transform(target)
                if self.num_crops == 1:
                    return sample_all, hard_target, soft_target
            # if len(sample_all) < self.num_crops - 1:
            #     # last_S = sample_all[-1]
            #     # last_H = hard_target[-1]
            #     # last_T = soft_target[-1]
            #     for idx in range(self.num_crops - 1):
            #         coord = coords[rand_index[idx]]
            #         sample_trans = self.transform(sample, coord, flip_status[rand_index[idx]])
            #         sample_all.append(sample_trans)
            #         hard_target.append(target)
            #         soft_target.append(output[rand_index[idx]])

            sample_in_conf = []
            for i in range(len(output)):
                tmp_soft = output[rand_index[i]]
                entropys = calculate_entropy(tmp_soft[1])
                if len(sample_in_conf) == 1:
                    break
                if entropys > threshold:
                    # sample_in_conf_label.append(output[rand_index[i]])
                    coord = coords[rand_index[i]]
                    sample_trans = self.transform(sample, coord, flip_status[rand_index[i]])
                    sample_in_conf.append(sample_trans)
                    # hard_target.append(target)
                else:
                    continue

            n = len(bbox)
            # iou_matrix = np.zeros((n, n))
            is_consistent = np.full((3, 3), False, dtype=bool)

            # for i in range(n):
            #     for j in range(i + 1, n):
            #         iou = compute_iou(bbox[i], bbox[j])
            #         iou_matrix[i, j] = iou
            #         iou_matrix[j, i] = iou  # IoU is symmetric
            # is_consistent = iou_matrix > 0.6

            is_consistent = torch.from_numpy(is_consistent)
            # in_conf_dict = OrderedDict()
            # in_conf_dict["sample_in_conf"] = sample_in_conf
            is_have_neg = False
            if len(sample_in_conf) == 1:
                is_have_neg = True
                sample_in_conf = sample_in_conf[0]
            else:
                is_have_neg = False
                sample_in_conf = torch.zeros_like(sample_trans)
            # if  np.random.rand() < 0.5:
            #     sample_all = []
            #     hard_target = []
            #     soft_target = []
            return sample_all, hard_target, soft_target, is_have_neg, sample_in_conf, is_consistent
            #     in_conf_dict["sample_in_conf"] = sample_in_conf
            # else:
            #     # print("h")
            # in_conf_dict["sample_in_conf"] = torch.zeros_like(sample_trans)




def Recover_soft_label(label, label_type, n_classes):
    # recover quantized soft label to n_classes dimension.
    if label_type == 'hard':

        return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1), 1)

    elif label_type == 'smoothing':
        index = label[:,0].to(dtype=int)
        value = label[:,1]
        minor_value = (torch.ones_like(value) - value)/(n_classes-1)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index.view(-1, 1), value.view(-1, 1))

        return soft_label

    elif label_type == 'marginal_smoothing_k5':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        minor_value = (torch.ones(label.size(0),1) - torch.sum(value, dim=1, keepdim=True))/(n_classes-5)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index, value)

        return soft_label

    elif label_type == 'marginal_renorm':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        soft_label = torch.zeros(index.size(0), n_classes).scatter_(1, index, value)
        soft_label = F.normalize(soft_label, p=1.0, dim=1, eps=1e-12)

        return soft_label

    elif label_type == 'marginal_smoothing_k10':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        minor_value = (torch.ones(label.size(0),1) - torch.sum(value, dim=1, keepdim=True))/(n_classes-10)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index, value)

        return soft_label


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(images, soft_label, args):
    enable_p = np.random.rand(1)
    if enable_p < args.mixup_cutmix_prob:
        switch_p = np.random.rand(1)
        if switch_p < args.mixup_switch_prob:
            lam = np.random.beta(args.mixup, args.mixup)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = soft_label
            target_b = soft_label[rand_index]
            mixed_x = lam * images + (1 - lam) * images[rand_index]
            target_mix = target_a * lam + target_b * (1 - lam)
            return mixed_x, target_mix
        else:
            lam = np.random.beta(args.cutmix, args.cutmix)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = soft_label
            target_b = soft_label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            target_mix = target_a * lam + target_b * (1 - lam)
    else:
        target_mix = soft_label

    return images, target_mix

def my_mixup_cutmix(images, soft_label, lam):
    # return False, images, soft_label
    enable_p = np.random.rand(1)
    if enable_p < 0.2:
        rand_index = torch.randperm(images.size()[0]).cuda()
        weight_1 = lam.unsqueeze(1).detach()
        weight_2 = lam[rand_index].unsqueeze(1).detach()
        weight = torch.cat((weight_1, weight_2), dim=1).softmax(dim=-1).to(images.device)
        target_a = soft_label
        target_b = soft_label[rand_index]

        # 0-1  reverse
        mixed_x = weight[:,1,None,None,None]  * images + weight[:,0,None,None,None] * images[rand_index]
        target_mix = target_a * weight[:,1,None] + target_b * weight[:,0,None]

        return True, mixed_x, target_mix

    else:
        return False, images, soft_label


if __name__ == "__main__":
    import torchvision.transforms as transforms
    softlabel_path = "/work/zhangzherui/data/FKD_soft_label_500_crops_marginal_smoothing_k_5/imagenet"
    traindir = os.path.join("/work/data/imagenet", 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_FKD(
        num_crops=4,
        softlabel_path=softlabel_path,
        root=traindir,
        transform=Compose_FKD(transforms=[
            RandomResizedCrop_FKD(size=224,
                                  interpolation='bilinear'),
            RandomHorizontalFlip_FKD(),
            transforms.ToTensor(),
            normalize,
        ]))

    ret = train_dataset[50]