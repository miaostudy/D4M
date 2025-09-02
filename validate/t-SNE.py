import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR

from utils import AverageMeter, accuracy, get_parameters

sys.path.append('..')
from matching.utils_fkd import ImageFolder_FKD_MIX, ComposeWithCoords, RandomResizedCropWithCoords, \
    RandomHorizontalFlipWithRes, mix_aug


# 钩子函数用于提取特征
class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.features = None

        # 注册钩子
        self.hook_handle = self._register_hook()

    def _register_hook(self):
        # 根据模型类型和层名找到目标层
        if isinstance(self.model, nn.DataParallel):
            modules = dict(self.model.module.named_modules())
        else:
            modules = dict(self.model.named_modules())

        if self.layer_name not in modules:
            raise ValueError(f"Layer {self.layer_name} not found in the model")

        layer = modules[self.layer_name]

        # 定义钩子函数
        def hook_fn(module, input, output):
            # 对输出特征进行平均池化，将空间维度压缩
            if len(output.shape) == 4:  # 卷积层输出 (batch, channels, height, width)
                self.features = F.adaptive_avg_pool2d(output, 1).squeeze()  # 变为 (batch, channels)
            else:
                self.features = output  # 全连接层输出直接使用

        return layer.register_forward_hook(hook_fn)

    def get_features(self):
        return self.features

    def remove_hook(self):
        self.hook_handle.remove()


def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K with t-SNE visualization")
    parser.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=8, type=int,  # 减少worker数量
                        help='number of data loading workers')

    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')

    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')
    parser.add_argument('--feature-layer', type=str,
                        default='layer4.1.conv2', help='layer name for feature extraction')
    parser.add_argument('--tsne-samples', type=int,
                        default=1000, help='number of samples for t-SNE visualization')
    parser.add_argument('--tsne-perplexity', type=int,
                        default=30, help='perplexity parameter for t-SNE')
    parser.add_argument('--tsne-max-iter', type=int,
                        default=1000, help='maximum iterations for t-SNE')
    parser.add_argument('--tsne-max-classes', type=int,
                        default=10, help='maximum number of classes to display in t-SNE')

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


def visualize_tsne(features, labels, epoch, output_dir, max_classes=10):
    """使用t-SNE可视化特征，添加严格的长度检查"""
    # 严格检查特征和标签长度是否一致
    if len(features) != len(labels):
        print(f"错误: 特征数量 ({len(features)}) 与标签数量 ({len(labels)}) 不匹配!")
        # 尝试修正：取较短的长度
        min_length = min(len(features), len(labels))
        features = features[:min_length]
        labels = labels[:min_length]
        print(f"已修正为共同长度: {min_length}")

    print(f"正在进行t-SNE可视化，特征形状: {features.shape}, 标签数量: {len(labels)}")

    # 先使用PCA进行预处理，加速t-SNE
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    print(f"PCA降维后特征形状: {features_pca.shape}")

    # 应用t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        random_state=42,
        max_iter=args.tsne_max_iter,
        n_jobs=-1  # 使用所有可用CPU
    )
    features_tsne = tsne.fit_transform(features_pca)
    print(f"t-SNE降维后特征形状: {features_tsne.shape}")

    # 再次检查长度是否一致
    if len(features_tsne) != len(labels):
        print(f"错误: t-SNE特征数量 ({len(features_tsne)}) 与标签数量 ({len(labels)}) 不匹配!")
        min_length = min(len(features_tsne), len(labels))
        features_tsne = features_tsne[:min_length]
        labels = labels[:min_length]
        print(f"已修正为共同长度: {min_length}")

    # 创建可视化图像
    plt.figure(figsize=(12, 10))

    # 选择要显示的类别（如果类别太多）
    unique_labels = np.unique(labels)
    if len(unique_labels) > max_classes:
        print(f"类别数量 ({len(unique_labels)}) 超过最大显示数量 ({max_classes})，将只显示部分类别")
        selected_labels = unique_labels[:max_classes]
        # 创建掩码并确保同步筛选
        mask = np.isin(labels, selected_labels)
        # 应用掩码
        features_tsne = features_tsne[mask]
        labels = labels[mask]
        unique_labels = selected_labels

        # 再次检查长度
        if len(features_tsne) != len(labels):
            print(f"错误: 筛选后特征数量 ({len(features_tsne)}) 与标签数量 ({len(labels)}) 不匹配!")
            min_length = min(len(features_tsne), len(labels))
            features_tsne = features_tsne[:min_length]
            labels = labels[:min_length]

    # 绘制散点图 - 使用更稳定的方式传递数据
    # 创建DataFrame确保数据对齐
    import pandas as pd
    tsne_df = pd.DataFrame({
        'tsne_dim1': features_tsne[:, 0],
        'tsne_dim2': features_tsne[:, 1],
        'label': labels
    })

    # 检查DataFrame是否完整
    if tsne_df.isnull().any().any():
        print("警告: 数据中存在空值，将被删除")
        tsne_df = tsne_df.dropna()

    scatter = sns.scatterplot(
        data=tsne_df,
        x='tsne_dim1',
        y='tsne_dim2',
        hue='label',
        palette=sns.color_palette("hsv", len(np.unique(tsne_df['label']))),
        legend="full",
        s=10,
        alpha=0.7
    )

    plt.title(f't-SNE Visualization of Features (Epoch {epoch})', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 保存图像
    tsne_dir = os.path.join(output_dir, 'tsne_visualizations')
    os.makedirs(tsne_dir, exist_ok=True)
    save_path = os.path.join(tsne_dir, f'tsne_epoch_{epoch}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"t-SNE可视化已保存至: {save_path}")

    # 返回可视化结果供wandb记录
    return save_path


def main():
    global args
    args = get_args()

    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, name=args.output_dir.split('/')[-1])

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 数据加载
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
        num_workers=args.workers, pin_memory=True,
        multiprocessing_context='forkserver',
        persistent_workers=True)

    # 加载验证数据
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size / 4), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        multiprocessing_context='forkserver',
        persistent_workers=True)
    print('数据加载成功')

    # 加载学生模型
    print(f"=> 加载学生模型 '{args.model}'")
    model = torchvision.models.__dict__[args.model](weights=None)
    model = nn.DataParallel(model).cuda()
    model.train()

    # 打印模型层信息，帮助选择特征层
    print("\n模型层信息:")
    for name, _ in model.named_modules():
        if name and (name.endswith('conv1') or name.endswith('conv2') or
                     name.endswith('fc') or 'layer' in name):
            print(f"  {name}")
    print(f"\n将使用 '{args.feature_layer}' 层提取特征用于t-SNE可视化\n")

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

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        train(model, args, epoch)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            top1 = validate(model, args, epoch)
            # 提取特征并进行t-SNE可视化
            extract_and_visualize_features(model, args, epoch)
        else:
            top1 = 0

        wandb.log(wandb_metrics)

        scheduler.step()

        # 保存最佳模型
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
        soft_label = soft_label.cuda().float()  # 转换为float32
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # 最后一个批次可能不满足batch_size
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

    printInfo = f'TRAIN Iter {epoch}: lr = {scheduler.get_last_lr()[0]:.6f},\tloss = {objs.avg:.6f},\t' + \
                f'Top-1 err = {100 - top1.avg:.6f},\t' + \
                f'Top-5 err = {100 - top5.avg:.6f},\t' + \
                f'train_time = {time.time() - t1:.6f}'
    print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            # 计算概率分布
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if len(all_probs) == 1 and epoch % 50 == 0:  # 只在特定epoch打印
                print("\n示例概率分布 (前5个样本):")
                for i in range(min(5, len(probs))):
                    print(f"样本 {i + 1} (真实标签: {target[i]})")
                    top5_probs, top5_indices = torch.topk(probs[i], 5)
                    for p, idx in zip(top5_probs, top5_indices):
                        print(f"  类别 {idx}: {p.item():.4f}")
                print()

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if epoch % 50 == 0:  # 只在特定epoch打印
        class_avg_probs = np.mean(all_probs, axis=0)
        print("类别平均概率 (前10个类别):")
        for i in range(min(10, len(class_avg_probs))):
            print(f"类别 {i}: {class_avg_probs[i]:.6f}")

        predictions = np.argmax(all_probs, axis=1)
        correct_mask = (predictions == all_labels)

        if np.sum(correct_mask) > 0:
            correct_probs = all_probs[correct_mask]
            correct_avg = np.mean(correct_probs, axis=0)
            print("\n正确预测的平均概率 (前10个类别):")
            for i in range(min(10, len(correct_avg))):
                print(f"类别 {i}: {correct_avg[i]:.6f}")

        if np.sum(~correct_mask) > 0:
            incorrect_probs = all_probs[~correct_mask]
            incorrect_avg = np.mean(incorrect_probs, axis=0)
            print("\n错误预测的平均概率 (前10个类别):")
            for i in range(min(10, len(incorrect_avg))):
                print(f"类别 {i}: {incorrect_avg[i]:.6f}")

    logInfo = f'TEST Iter {epoch}: loss = {objs.avg:.6f},\t' + \
              f'Top-1 err = {100 - top1.avg:.6f},\t' + \
              f'Top-5 err = {100 - top5.avg:.6f},\t' + \
              f'val_time = {time.time() - t1:.6f}'
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def extract_and_visualize_features(model, args, epoch):
    """提取特征并进行t-SNE可视化"""
    print(f"\n开始提取特征并进行t-SNE可视化 (Epoch {epoch})")

    # 创建特征提取器
    feature_extractor = FeatureExtractor(model, args.feature_layer)

    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(args.val_loader):
            data, target = data.cuda(), target.cuda()

            # 确保数据和标签长度一致
            if len(data) != len(target):
                print(f"警告: 批次 {batch_idx} 中数据数量 ({len(data)}) 与标签数量 ({len(target)}) 不匹配")
                min_len = min(len(data), len(target))
                data = data[:min_len]
                target = target[:min_len]

            # 前向传播，钩子函数会自动提取特征
            model(data)

            # 获取提取的特征
            features = feature_extractor.get_features()
            if features is None:
                print(f"警告: 未能从层 {args.feature_layer} 提取特征")
                break

            # 检查特征和标签数量是否一致
            if len(features) != len(target):
                print(f"警告: 批次 {batch_idx} 中特征数量 ({len(features)}) 与标签数量 ({len(target)}) 不匹配")
                min_len = min(len(features), len(target))
                features = features[:min_len]
                target = target[:min_len]

            all_features.append(features.cpu().numpy())
            all_labels.append(target.cpu().numpy())

            # 限制样本数量以加速t-SNE
            total_samples = sum(len(labels) for labels in all_labels)
            if total_samples >= args.tsne_samples:
                break

    # 移除钩子
    feature_extractor.remove_hook()

    if not all_features:
        print("未能提取到任何特征，跳过t-SNE可视化")
        return

    # 合并所有特征和标签
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 再次检查长度一致性
    if len(all_features) != len(all_labels):
        print(f"错误: 合并后特征数量 ({len(all_features)}) 与标签数量 ({len(all_labels)}) 不匹配!")
        min_length = min(len(all_features), len(all_labels))
        all_features = all_features[:min_length]
        all_labels = all_labels[:min_length]
        print(f"已修正为共同长度: {min_length}")

    # 限制样本数量
    if len(all_features) > args.tsne_samples:
        indices = np.random.choice(len(all_features), args.tsne_samples, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]

        # 检查随机选择后的长度
        if len(all_features) != len(all_labels):
            print(f"错误: 随机选择后特征数量 ({len(all_features)}) 与标签数量 ({len(all_labels)}) 不匹配!")
            min_length = min(len(all_features), len(all_labels))
            all_features = all_features[:min_length]
            all_labels = all_labels[:min_length]

    # 进行t-SNE可视化
    tsne_image_path = visualize_tsne(all_features, all_labels, epoch, args.output_dir, args.tsne_max_classes)

    # 将可视化结果上传到wandb
    if tsne_image_path and os.path.exists(tsne_image_path):
        wandb.log({f"tsne/epoch_{epoch}": wandb.Image(tsne_image_path)})


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

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
    wandb.finish()
