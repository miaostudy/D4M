import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from utils import AverageMeter, accuracy, get_parameters

sys.path.append('..')
from matching.utils_fkd import ImageFolder_FKD_MIX, ComposeWithCoords, RandomResizedCropWithCoords, \
    RandomHorizontalFlipWithRes, mix_aug


def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')

    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')

    # 添加特征可视化和标签分析参数
    parser.add_argument('--vis_frequency', type=int, default=2,
                        help='每隔多少个epoch可视化一次训练特征')
    parser.add_argument('--vis_samples', type=int, default=500,
                        help='用于可视化的样本数量')
    parser.add_argument('--analyze-labels', action='store_true', default=True,
                        help='是否分析标签分布')
    parser.add_argument('--label-analysis-samples', type=int, default=1000,
                        help='用于标签分析的样本数量')

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')

    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')

    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--wandb-project', type=str,
                        default='Temperature', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')

    args = parser.parse_args()

    args.mode = 'fkd_load'
    return args


def analyze_training_labels(args):
    """分析训练集中的硬标签和软标签"""
    print("\n===== 开始分析训练集标签 =====")

    # 创建标签分析目录
    label_analysis_dir = os.path.join(args.output_dir, 'label_analysis')
    if not os.path.exists(label_analysis_dir):
        os.makedirs(label_analysis_dir)

    # 保存当前epoch以便恢复
    train_dataset = args.train_loader.dataset
    current_epoch = train_dataset.epoch
    train_dataset.set_epoch(0)  # 临时设置为0以便加载数据

    # 收集标签数据
    hard_labels = []
    soft_labels_list = []
    collected = 0

    # 使用单独的DataLoader以避免干扰训练
    temp_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=args.train_loader.sampler,
        num_workers=args.workers, pin_memory=True
    )

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(temp_loader):
            if collected >= args.label_analysis_samples:
                break

            # 提取硬标签和软标签
            _, target, _, _ = batch_data[0]  # 硬标签
            _, _, _, soft_label = batch_data[1:]  # 软标签

            # 转换为numpy并收集
            batch_size = target.size(0)
            take = min(batch_size, args.label_analysis_samples - collected)

            hard_labels.extend(target[:take].numpy())
            soft_labels_list.append(soft_label[:take].cpu().numpy())

            collected += take
            print(f"已收集 {collected}/{args.label_analysis_samples} 个样本用于标签分析...", end='\r')

    # 恢复原始epoch设置
    train_dataset.set_epoch(current_epoch)

    # 转换为numpy数组
    hard_labels = np.array(hard_labels)
    soft_labels = np.concatenate(soft_labels_list, axis=0)

    # 1. 分析硬标签分布
    unique_hard, counts_hard = np.unique(hard_labels, return_counts=True)
    print(f"\n\n硬标签分析:")
    print(f"  类别数量: {len(unique_hard)}")
    print(f"  样本数量: {len(hard_labels)}")
    print(f"  前10个类别分布:")
    for i in range(min(10, len(unique_hard))):
        print(f"    类别 {unique_hard[i]}: {counts_hard[i]} 个样本 ({counts_hard[i] / len(hard_labels):.2%})")

    # 2. 分析软标签特性
    print(f"\n软标签分析:")
    print(f"  软标签维度: {soft_labels.shape[1]} (应等于类别数量)")

    # 计算每个样本的软标签熵 (熵越高表示不确定性越大)
    softmax_soft_labels = soft_labels  # 假设已经是softmax后的结果
    entropy = -np.sum(softmax_soft_labels * np.log(softmax_soft_labels + 1e-10), axis=1)
    print(f"  软标签平均熵: {np.mean(entropy):.4f} (值越高表示不确定性越大)")
    print(f"  软标签熵范围: [{np.min(entropy):.4f}, {np.max(entropy):.4f}]")

    # 计算每个样本的软标签最大值 (置信度)
    max_probs = np.max(softmax_soft_labels, axis=1)
    print(f"  软标签平均最大概率: {np.mean(max_probs):.4f} (值越低表示置信度越低)")

    # 检查硬标签与软标签的一致性
    soft_label_preds = np.argmax(softmax_soft_labels, axis=1)
    consistency = np.mean(soft_label_preds == hard_labels)
    print(f"  硬标签与软标签一致性: {consistency:.2%} (软标签预测与硬标签相同的比例)")

    # 3. 可视化软标签分布
    plt.figure(figsize=(12, 6))

    # 绘制软标签熵分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(entropy, bins=30, alpha=0.7)
    plt.title('软标签熵分布')
    plt.xlabel('熵值')
    plt.ylabel('样本数量')

    # 绘制软标签最大概率分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(max_probs, bins=30, alpha=0.7)
    plt.title('软标签最大概率分布')
    plt.xlabel('最大概率值')
    plt.ylabel('样本数量')

    plt.tight_layout()
    plt.savefig(os.path.join(label_analysis_dir, 'soft_label_distributions.png'), dpi=300)
    plt.close()
    print(f"  软标签分布可视化已保存至: {os.path.join(label_analysis_dir, 'soft_label_distributions.png')}")

    # 4. 分析几个样本的软标签详情
    print("\n样本软标签详情 (前5个样本):")
    for i in range(min(5, len(hard_labels))):
        hard_label = hard_labels[i]
        soft_label = softmax_soft_labels[i]
        top3_idx = np.argsort(soft_label)[-3:][::-1]  # 取概率最高的3个类别
        top3_probs = soft_label[top3_idx]

        print(f"  样本 {i}: 硬标签={hard_label}")
        for idx, prob in zip(top3_idx, top3_probs):
            print(f"    类别 {idx}: {prob:.4f} {'(硬标签)' if idx == hard_label else ''}")

    print("===== 训练集标签分析结束 =====\n")


def main():
    args = get_args()

    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, name=args.output_dir.split('/')[-1])

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # 创建特征可视化目录
    vis_dir = os.path.join(args.output_dir, 'feature_visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    args.vis_dir = vis_dir

    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.train_dir,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=224,
                                        scale=(0.08, 1),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_dataset = datasets.ImageFolder(args.val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(args.batch_size / 4), shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')

    # 输出训练和验证数据集的统计信息
    print("\n===== 数据集统计信息 =====")
    # 训练集信息
    train_num_classes = len(train_dataset.classes)
    train_num_samples = len(train_dataset)
    print(f"训练集: {train_num_samples} 个样本, {train_num_classes} 个类别")

    # 验证集信息
    val_num_classes = len(val_dataset.classes)
    val_num_samples = len(val_dataset)
    print(f"验证集: {val_num_samples} 个样本, {val_num_classes} 个类别")
    print("==========================\n")

    # 分析标签（如果启用）
    if args.analyze_labels:
        args.train_loader = train_loader  # 传递给分析函数
        analyze_training_labels(args)

    # load student model
    print("=> loading student model '{}'".format(args.model))
    model = torchvision.models.__dict__[args.model](weights=None)
    model = nn.DataParallel(model).cuda()
    model.train()

    # 创建特征提取器（通用结构，适用于ResNet等）
    args.feature_extractor = nn.Sequential(
        *list(model.module.children())[:-1]  # 排除最后的全连接层
    )

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0,
                             last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0 - step / args.epochs) if step <= args.epochs else 0, last_epoch=-1)

    args.best_acc1 = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader
    train_dataset.set_epoch(0)
    # 训练前可视化初始特征（随机权重）
    visualize_train_features(model, args, -1)  # epoch=-1表示初始状态

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        train(model, args, epoch)

        # 定期可视化训练特征
        if epoch % args.vis_frequency == 0 or epoch == args.epochs - 1:
            visualize_train_features(model, args, epoch)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            top1 = validate(model, args, epoch)
        else:
            top1 = 0

        wandb.log(wandb_metrics)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status = batch_data[0]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            output = F.log_softmax(output / args.temperature, dim=1)
            partial_soft_label = F.softmax(partial_soft_label / args.temperature, dim=1)
            loss = loss_function_kl(output, partial_soft_label)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        optimizer.step()

    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
    }
    wandb_metrics.update(metrics)

    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()


def visualize_train_features(model, args, epoch):
    """可视化训练集特征"""
    model.eval()
    feature_extractor = args.feature_extractor
    all_features = []
    all_labels = []

    # 限制采样数量以提高速度
    max_samples = args.vis_samples
    collected = 0

    # 保存当前epoch以便恢复
    current_epoch = args.train_loader.dataset.epoch
    # 临时设置为0，确保数据加载正常
    args.train_loader.dataset.set_epoch(0)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(args.train_loader):
            if collected >= max_samples:
                break

            images, target, _, _ = batch_data[0]
            images = images.cuda()
            target = target.cpu().numpy()

            # 提取特征
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)  # 展平特征
            features = features.cpu().numpy()

            # 收集特征和标签
            batch_samples = min(images.size(0), max_samples - collected)
            all_features.append(features[:batch_samples])
            all_labels.append(target[:batch_samples])

            collected += batch_samples

    # 恢复原始epoch设置
    args.train_loader.dataset.set_epoch(current_epoch)

    # 合并所有数据
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 应用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
    tsne_results = tsne.fit_transform(all_features)

    # 创建可视化
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    scatter = sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=all_labels,
        palette=sns.color_palette("hsv", len(unique_labels)),
        legend="full",
        alpha=0.7
    )

    plt.title(f'Train Feature t-SNE (Epoch {epoch})')
    plt.legend(title="类别", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(args.vis_dir, f'train_tsne_epoch_{epoch}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    model.train()  # 恢复训练模式
    print(f"已保存训练特征可视化: {os.path.join(args.vis_dir, f'train_tsne_epoch_{epoch}.png')}")


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()

    all_probs = []
    all_labels = []
    all_features = []

    feature_extractor = nn.Sequential(
        *list(model.module.children())[:-1]
    )

    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            features = feature_extractor(data)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu().numpy())

            output = model(data)
            loss = loss_function(output, target)

            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    # 合并所有数据
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_features = np.concatenate(all_features, axis=0)

    # 输出验证集标签分布（仅在第一个验证epoch输出）
    if epoch == 0 or epoch == args.start_epoch:
        print("\n===== 验证集标签分布 =====")
        unique, counts = np.unique(all_labels, return_counts=True)
        # 只显示前10个类别和总数
        for i in range(min(10, len(unique))):
            print(f"类别 {unique[i]}: {counts[i]} 个样本")
        if len(unique) > 10:
            print(f"... 共 {len(unique)} 个类别")
        print("==========================\n")

    # 执行t-SNE降维 - 为了加速，只使用部分样本
    sample_indices = np.random.choice(len(all_features), min(1000, len(all_features)), replace=False)
    sample_features = all_features[sample_indices]
    sample_labels = all_labels[sample_indices]

    # 应用t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
    tsne_results = tsne.fit_transform(sample_features)

    # 创建可视化
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(sample_labels)
    scatter = sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=sample_labels,
        palette=sns.color_palette("hsv", len(unique_labels)),
        legend="full",
        alpha=0.7
    )

    plt.title(f'Validation Feature t-SNE (Epoch {epoch})')
    plt.legend(title="类别", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(args.output_dir, f'val_tsne_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 精简输出：只显示关键指标
    print("\n验证集关键指标:")
    print(f"类别平均概率 (前5个类别):")
    class_avg_probs = np.mean(all_probs, axis=0)
    for i in range(min(5, len(class_avg_probs))):
        print(f"类别 {i}: {class_avg_probs[i]:.6f}")

    predictions = np.argmax(all_probs, axis=1)
    correct_mask = (predictions == all_labels)
    accuracy = np.mean(correct_mask) * 100
    print(f"\n整体准确率: {accuracy:.2f}%")

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def save_checkpoint(state, is_best, output_dir=None, epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method('spawn')
    main()
    wandb.finish()
