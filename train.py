import argparse
import os
import yaml
import datasets
import models
import utils
import torch
import torch.distributed as dist
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from eval_iou import SegmentationMetric
from statistics import mean
from prettytable import PrettyTable

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def onehot_to_index_label(mask):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0).numpy()
    x = np.argmax(mask, axis=-1)
    #colour_codes = np.array(palette)
    #x = np.uint8(colour_codes[x.astype(np.uint8)])*255
    #x=x.permute(2,0,1)
    #x=x.numpy()
    #x = np.around
    return x


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler,drop_last=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()
    eval_type =config.get('eval_type')
    class_num = config['model']['args']['num_classes']
    ignore_background =  config['val_dataset']['dataset']['args']['ignore_bg']

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'kvasir':
        metric_fn = utils.calc_kvasir
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'
    elif eval_type == 'seg':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

        metric_seg = SegmentationMetric(class_num,  ignore_background)

    # val_metric1 = utils.Averager()
    # val_metric2 = utils.Averager()
    # val_metric3 = utils.Averager()
    # val_metric4 = utils.Averager()


    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    
    val_metric1 = 0
    val_metric2 = 0
    val_metric3 = 0
    val_metric4 = 0
    cnt = 0
    
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        output_masks = model.infer(inp)
        pred = torch.sigmoid(output_masks)

        # pred = torch.sigmoid(model.infer(inp))

        batch_pred = [torch.zeros_like(output_masks) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]
        
        dist.all_gather(batch_pred, pred)

        for i in range(len(batch_pred)):
            #print(len(batch_pred))
            batch_pred[i]=batch_pred[i].to('cpu')
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        for i in range(len(batch_gt)):
            batch_gt[i]=batch_gt[i].to('cpu')
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

        #print(len(batch_gt))
        for i in range(len(batch_gt)):
            output_mask = batch_pred[i][0]
        #output_mask = output_masks[0].cpu().detach()
        #binary_mask = onehot_to_mask(output_mask)
            mask_index_label = onehot_to_index_label(output_mask).flatten()

            gt_mask = batch_gt[i][0]
            #gt_mask= batch['gt'][0].cpu().detach()
        #gt_mask_rgb = onehot_to_mask(gt_mask)
            gt_index_label = onehot_to_index_label(gt_mask).flatten()

            if eval_type == 'seg':
                metric_seg.addBatch(mask_index_label,gt_index_label)

        #pred = torch.sigmoid(model.infer(inp))

        # result1, result2, result3, result4 = metric_fn(pred, batch['gt'])

        # val_metric1.add(result1.item(), inp.shape[0])
        # val_metric2.add(result2.item(), inp.shape[0])
        # val_metric3.add(result3.item(), inp.shape[0])
        # val_metric4.add(result4.item(), inp.shape[0])
    

        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1 += (result1 * pred.shape[0])
        val_metric2 += (result2 * pred.shape[0])
        val_metric3 += (result3 * pred.shape[0])
        val_metric4 += (result4 * pred.shape[0])     
        cnt += pred.shape[0]
        
    if pbar is not None:
            pbar.update(1)
    val_metric1 = torch.tensor(val_metric1).cuda()
    val_metric2 = torch.tensor(val_metric2).cuda()
    val_metric3 = torch.tensor(val_metric3).cuda()
    val_metric4 = torch.tensor(val_metric4).cuda()
    cnt = torch.tensor(cnt).cuda()
    dist.all_reduce(val_metric1)
    dist.all_reduce(val_metric2)
    dist.all_reduce(val_metric3)
    dist.all_reduce(val_metric4)
    dist.all_reduce(cnt)
          
    # if pbar is not None:
    #     pbar.close()
    
    # return val_metric1.item()/cnt, val_metric2.item()/cnt, val_metric3.item()/cnt, val_metric4.item()/cnt, metric1, metric2, metric3, metric4

    oa = metric_seg.overallAccuracy()
    oa = np.around(oa,decimals=4)
    mIoU ,IoU= metric_seg.meanIntersectionOverUnion()
    mIoU = np.around(mIoU,decimals=4)
    IoU = np.around(IoU,decimals=4)
    p = metric_seg.precision()
    p = np.around(p,decimals=4)
    mp = np.nanmean(p)
    mp = np.around(mp,decimals=4)
    r = metric_seg.recall()
    r=np.around(r,decimals=4)
    mr = np.nanmean(r)
    mr = np.around(mr,decimals=4)
    f1 = (2*p*r) / (p + r)
    f1 = np.around(f1,decimals=4)
    mf1 = np.nanmean(f1)
    mf1 = np.around(mf1,decimals=4)
    normed_confusionMatrix = metric_seg.confusionMatrix / metric_seg.confusionMatrix.sum(axis=0)
    normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
    fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()
    fwIOU= np.around(fwIOU,decimals=4)

    classes_list = config['train_dataset']['dataset']['args']['classes']
    if ignore_background:
        axis_labels=classes_list[:-1]
    else:
        axis_labels=classes_list


    IOU_row = ['IOU',mIoU]
    IOU_row.extend(IoU.tolist())
    Precision_row = ['Precision',mp]
    Precision_row.extend(p.tolist())
    Recall_row = ['Recall',mr]
    Recall_row.extend(r.tolist())
    F1_row = ['F1',mf1]
    F1_row.extend(f1.tolist())
    title_row = ['metrics','average']
    title_row.extend(axis_labels)
    OA_row = ['OA',oa]#,' ',' ',' ',' ']

    fwIOU_row = ['FWIOU', fwIOU]#,' ',' ',' ',' ']
    for i in range(len(axis_labels)):
        OA_row.append(' ')
        fwIOU_row.append(' ')

    table = PrettyTable(title_row)
    table.add_row(IOU_row)
    table.add_row(Precision_row)
    table.add_row(Recall_row)
    table.add_row(F1_row)
    table.add_row(OA_row)
    table.add_row(fwIOU_row)

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), metric1, metric2, metric3, metric4, table, normed_confusionMatrix



def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint['model'], strict=False)
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4, seg_eval_table, normed_confusionMatrix = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                else:
                    if result2 < max_val_v:
                        max_val_v = result2
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log_info.append(str(seg_eval_table))
                log_info.append('Confusion Matrix:')
                log_info.append(str(normed_confusionMatrix))
                
                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/home/diani/diani/SAM2-Adapter/configs/demo.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
