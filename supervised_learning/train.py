import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import time
from tqdm import tqdm
import logging

# SwanLab 可视化
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    print("SwanLab not installed. Training will continue without visualization.")

# 引入本地模块
from models.c6transnet import C6TransNet
from data.dataset import Connect6Dataset

# 设置日志
# 确保 output 目录存在
os.makedirs('output', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 默认数据路径: script_dir/data/processed/connect6_cleaned.csv
    default_data_path = os.path.join(script_dir, 'data/processed/connect6_cleaned.csv')
    
    parser = argparse.ArgumentParser(description="Train C6-TransNet for Connect6")
    parser.add_argument('--data', type=str, default=default_data_path, help='Path to training data CSV')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Total epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=str, default=None, help='GPU ids to use (e.g. "0,1"), None for all available')
    parser.add_argument('--workers', type=int, default=4, help='Num data loader workers')
    return parser.parse_args()

def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    # 如果是最佳模型，可以另外保存一份
    # (这里简化逻辑，只保存最新的和按epoch命名的)

def main():
    args = parse_args()
    
    # SwanLab 初始化
    if HAS_SWANLAB:
        swanlab.init(
            project="connect6-experiment",
            experiment_name="supervised-learning",
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.wd,
                "model": "C6TransNet",
                "params": "11M"
            }
        )
    
    # 1. GPU 设置
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using device: {device}, Num GPUs: {num_gpus}")

    # 2. 数据集
    logger.info(f"Loading dataset from {args.data}...")
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return

    dataset = Connect6Dataset(args.data, mode='train')
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True
    )
    logger.info(f"Dataset loaded. Steps per epoch: {len(dataloader)}")

    # 3. 模型构建
    model = C6TransNet(input_planes=17)
    model = model.to(device)
    
    # 多 GPU 支持
    if num_gpus > 1:
        logger.info("Activating DataParallel...")
        model = nn.DataParallel(model)

    # 4. 优化器与损失
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss Functions
    # Policy: Multi-class classification (361 classes) -> CrossEntropyLoss
    # Value: Regression (-1 to 1) -> MSELoss
    criterion_policy = nn.CrossEntropyLoss(ignore_index=361) # 忽略 padding
    criterion_value = nn.MSELoss()
    
    scaler = GradScaler() # 混合精度训练

    # 5. 恢复断点
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            
            # 处理 DataParallel 的 state_dict key 问题
            state_dict = checkpoint['state_dict']
            if num_gpus > 1 and not list(state_dict.keys())[0].startswith('module.'):
                # 如果当前是多卡，但ckpt是单卡，加 module. 前缀
                new_state_dict = {'module.'+k: v for k, v in state_dict.items()}
                state_dict = new_state_dict
            elif num_gpus == 1 and list(state_dict.keys())[0].startswith('module.'):
                # 如果当前是单卡，但ckpt是多卡，去 module. 前缀
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = new_state_dict
                
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            logger.info(f"No checkpoint found at '{args.resume}'")

    # 6. 训练循环
    logger.info("Start Training...")
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        
        # 新增：准确率统计
        correct_move1 = 0
        correct_move2 = 0
        total_move1 = 0
        total_move2 = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (inputs, target_move1, target_move2, target_value) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True) # (B, 17, 19, 19)
            target_move1 = target_move1.to(device, non_blocking=True) # (B,)
            target_move2 = target_move2.to(device, non_blocking=True) # (B,)
            target_value = target_value.to(device, non_blocking=True) # (B, 1)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                # Teacher Forcing: 传入真实的 move1 让模型预测 move2
                pred_p1, pred_p2, pred_value = model(inputs, move1_idx=target_move1)
                
                # Calculate Loss
                loss_p1 = criterion_policy(pred_p1, target_move1)
                
                if pred_p2 is not None:
                    loss_p2 = criterion_policy(pred_p2, target_move2)
                else:
                    loss_p2 = 0.0
                
                loss_p = loss_p1 + loss_p2
                loss_v = criterion_value(pred_value, target_value)
                loss = loss_p + 0.5 * loss_v
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 计算准确率
            with torch.no_grad():
                # Move1 准确率
                pred_move1 = pred_p1.argmax(dim=1)
                correct_move1 += (pred_move1 == target_move1).sum().item()
                total_move1 += target_move1.size(0)
                
                # Move2 准确率 (忽略 padding=361)
                if pred_p2 is not None:
                    valid_mask = target_move2 != 361
                    if valid_mask.sum() > 0:
                        pred_move2 = pred_p2.argmax(dim=1)
                        correct_move2 += (pred_move2[valid_mask] == target_move2[valid_mask]).sum().item()
                        total_move2 += valid_mask.sum().item()
            
            # 记录
            total_loss += loss.item()
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            
            # 显示实时准确率
            acc1 = correct_move1 / total_move1 * 100 if total_move1 > 0 else 0
            acc2 = correct_move2 / total_move2 * 100 if total_move2 > 0 else 0
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}", 
                'acc1': f"{acc1:.1f}%",
                'acc2': f"{acc2:.1f}%"
            })
        
        scheduler.step()
        duration = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        final_acc1 = correct_move1 / total_move1 * 100 if total_move1 > 0 else 0
        final_acc2 = correct_move2 / total_move2 * 100 if total_move2 > 0 else 0
        
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Move1 Acc: {final_acc1:.2f}% | Move2 Acc: {final_acc2:.2f}% | Time: {duration:.1f}s")
        
        # SwanLab 记录
        if HAS_SWANLAB:
            avg_p_loss = total_p_loss / len(dataloader)
            avg_v_loss = total_v_loss / len(dataloader)
            swanlab.log({
                "loss": avg_loss,
                "policy_loss": avg_p_loss,
                "value_loss": avg_v_loss,
                "move1_acc": final_acc1,
                "move2_acc": final_acc2,
                "lr": scheduler.get_last_lr()[0]
            })
        
        # 保存 Checkpoint
        if (epoch + 1) % 1 == 0: # 每个 epoch 都存
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, args.save_dir, filename=f'checkpoint_epoch_{epoch+1}.pth')
            
            # 更新 'latest.pth'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, args.save_dir, filename='checkpoint_latest.pth')

    logger.info("Training Complete!")

if __name__ == "__main__":
    main()
