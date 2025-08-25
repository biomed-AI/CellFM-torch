import scanpy as sc
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from layers.utils import *
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import json

import pickle

import warnings
warnings.filterwarnings("ignore")

from model import Finetune_Cell_FM


def basic():
    cfg2 = Config_80M()
    cfg2.ecs_threshold = 0.8
    cfg2.ecs = True
    cfg2.add_zero = True
    cfg2.pad_zero = True
    cfg2.use_bs = 16
    cfg2.mask_ratio = 0.5
    
    class config_pt:
        dataset = "Pancrm4"
        feature_col = "cell_type" 
        sample_batch_size = 4 # each batch contain 2 sample
        cell_batch_size = 8 # each porcess contain 8 cells -> to LLM
        construct_data = True
        att_hidden = 1536 # process to attention pooling
        gene_hidden = 2048
        gene_ls = None
        device = "cuda:7"
        epoch = 5
        ckpt_path = "/bigdat2/user/shanggny/checkpoint/para80m/6300w_18000_19479-1_38071.ckpt"
        num_cls = 1
    
    cfg = config_pt()
    #### A lots of Path ####
    PT_PATH = f"../data_pt/{cfg.dataset}"
    MODEL_PATH = f"../model_checkpoint/{cfg.dataset}" # for 40000 cells
    
    #### Make dir ####
    os.makedirs(PT_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    def load_data(adata_path):
        adata = read_h5ad(adata_path)
        # adata.var_names = adata.var['gene_name']
        adata.obs['celltype'] = adata.obs['cell_type']
        adata.obs['feat'] = adata.obs[cfg.feature_col].cat.codes.values
        cfg.num_cls = len(adata.obs['feat'].unique())
        
        adata.obs['batch_id'] = 0
        adata.obs['train'] = 0

        dataset = SCrna(adata, mode="train")
        prep = Prepare(cfg2.nonz_len, pad=0, mask_ratio=cfg2.mask_ratio)
        loader = build_dataset(
            dataset,
            prep=prep,
            batch_size=cfg2.use_bs,
            pad_zero=cfg2.pad_zero,
            drop=True,
            shuffle=True
        )
        return loader
    ################### training ###################
    train_adata_path = f"/data/user/liwb/project/CellFM/datasets/cell_annotion/Inter/{cfg.dataset}/train.h5ad"
    test_adata_path = f"/data/user/liwb/project/CellFM/datasets/cell_annotion/Inter/{cfg.dataset}/test.h5ad"
    
    train_loader = load_data(train_adata_path)
    test_loader = load_data(test_adata_path)
    

    net = Finetune_Cell_FM(cfg) # 27855

    for name, param in net.named_parameters():
        param.requires_grad = "cls." in name or "encoder" in name
    
    print("Trainable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    net = net.to(cfg.device)
    net.extractor.load_model(weight=True, moment=False)
    
    optimizer = AdamW([p for p in net.parameters() if p.requires_grad], 
                      lr=1e-4,
                      weight_decay=1e-5)
    
    scaler = GradScaler() 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    
    criterion_cls = nn.CrossEntropyLoss()
    for epoch in range(cfg.epoch):
        net.train()
        print("training...")
        running_loss = 0.0
        running_acc = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epoch}")
        
        for step, batch in enumerate(progress):    
            
            # === 数据输入 ===
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)
            celltype_label = batch['celltype_label'].to(cfg.device)
            batch_id = batch['batch_id'].to(cfg.device)
            feat = batch['feat'].long().to(cfg.device)

            # === 清空梯度 ===
            optimizer.zero_grad()
            # === 前向 + 反向（AMP混合精度） ===
            with torch.cuda.amp.autocast():
                cls, mask_loss, cls_token = net(
                    raw_nzdata=raw_nzdata,
                    dw_nzdata=dw_nzdata,
                    ST_feat=ST_feat,
                    nonz_gene=nonz_gene,
                    mask_gene=mask_gene,
                    zero_idx=zero_idx
                ) 
                
                cls_loss = criterion_cls(cls, feat)
                loss = mask_loss + cls_loss
            
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            accuracy = (cls.argmax(1) == feat).sum().item()
            accuracy = accuracy / len(batch_id)
            
            running_loss += loss.item()
            running_acc += accuracy
            
            avg_loss = running_loss / (step + 1)
            avg_acc = running_acc / (step + 1)
            
            progress.set_postfix(loss=avg_loss, acc=avg_acc)
        
        scheduler.step()
        print(f"Epoch {epoch+1} 完成,平均loss: {avg_loss:.6f}")
        torch.save(net.state_dict(), f"{MODEL_PATH}/checkpoint_epoch_{epoch+1}.pth")

        net.eval()
        print("testing...")
        running_loss = 0.0
        running_acc = 0.0
    
        progress = tqdm(test_loader, desc="Testing")
        with torch.no_grad():  # 测试阶段不需要计算梯度
            for step, batch in enumerate(progress):    
                
                raw_nzdata = batch['raw_nzdata'].to(cfg.device)
                dw_nzdata = batch['dw_nzdata'].to(cfg.device)
                ST_feat = batch['ST_feat'].to(cfg.device)
                nonz_gene = batch['nonz_gene'].to(cfg.device)
                mask_gene = batch['mask_gene'].to(cfg.device)
                zero_idx = batch['zero_idx'].to(cfg.device)
                celltype_label = batch['celltype_label'].to(cfg.device)
                batch_id = batch['batch_id'].to(cfg.device)
                feat = batch['feat'].long().to(cfg.device)

                # 测试阶段不需要清空梯度和反向传播
                with torch.cuda.amp.autocast():
                    cls, mask_loss, cls_token = net(
                        raw_nzdata=raw_nzdata,
                        dw_nzdata=dw_nzdata,
                        ST_feat=ST_feat,
                        nonz_gene=nonz_gene,
                        mask_gene=mask_gene,
                        zero_idx=zero_idx
                    ) 
                    
                    cls_loss = criterion_cls(cls, feat)
                    loss = mask_loss[0] + cls_loss

                pred = cls.argmax(1)
                accuracy = (pred == feat).sum().item()
                accuracy = accuracy / len(batch_id)
                
                running_loss += loss.item()
                running_acc += accuracy
                
                avg_loss = running_loss / (step + 1)
                avg_acc = running_acc / (step + 1)
                
                progress.set_postfix(loss=avg_loss, acc=avg_acc)
        
        print(f"Testing {epoch+1} 完成,平均loss: {avg_loss:.6f}, 平均准确率: {avg_acc:.6f}")
    
if __name__ == "__main__":
    basic()