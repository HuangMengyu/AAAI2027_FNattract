import numpy as np
import argparse
import random
import gc
from datetime import datetime

from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

from model import Model as tfcc_model
from common_utils.dataset_loader import get_path_loader_new, get_idx, get_loader
from base_models import SSLDataSet, FTDataSet
from models.prior_bmm import load_or_build_magnitude_prior

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import os


def parse_warm_epochs(value):
    value = str(value).strip()
    if value.lower() == "adaptive":
        return "adaptive"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--warm_epochs must be an integer or 'adaptive'") from exc


def parse_prior_hard_neg_weight(value):
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--prior_hard_neg_weight must be a float or 'auto'") from exc


def self_supervised_learning(dataset_name, fold, model, X, batch_size, device, model_save_path, use_iteration):
    start_datetime = datetime.now()

    model.to(device)
    model.train()

    dataset = SSLDataSet(X, dataset_name=dataset_name)
    print('SSL dataset:', len(dataset))

    if not use_iteration:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)
    else:
        print("Fix randomness for iteration training by setting num_workers=0.")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    model(dataloader, ssl=True)

    torch.save(model.state_dict(), os.path.join(model_save_path, f'Pretrained_Model_{fold}.pkl'))
    end_datetime = datetime.now()
    time_diff = end_datetime - start_datetime
    time_diff_seconds = time_diff.total_seconds()
    print(f"Pretraining Time for fold {fold}: {time_diff_seconds} seconds")
    

def finetuning(fold, model, train_set, valid_set, n_epoch, lr, batch_size, device, model_save_path, multi_label=True, n_classes=5):
    # multi_label: whether the classification task is a multi-label task.
    start_datetime = datetime.now()
    model.train()
    model.to(device)
    
    print("Fine-tuning train set:", len(train_set))
    print("Fine-tuning valid set:", len(valid_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    loss_func = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    
    best_auc = 0
    step = 0

    min_lr = 1e-8

    optimizer = optim.AdamW(model.parameters(), lr, betas=(0.5, 0.99), weight_decay=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode = 'max', factor=0.8, min_lr=min_lr)
    pbar = trange(n_epoch)
    for _ in pbar:
        for batch_idx, batch in enumerate(train_loader):
            step += 1
            *modalities, y = tuple(t.to(device) for t in batch)
            if y.shape[0] == 1:
                continue  # skip the batch if only one sample
            x = tuple(modalities)
            pred = model(x)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            gap = max(1, len(train_loader) // 5)
            if step % gap == 0 :
                valid_auc, val_label = val(model, valid_set, batch_size, multi_label, n_classes=n_classes)
                pbar.set_description('Best Validation AUC: {:.4f} --------- AUC on this step: {:.4f}'.format(best_auc, valid_auc))
                if valid_auc > best_auc:
                    best_auc = valid_auc
                    full_state = model.state_dict()
                    torch.save(full_state, os.path.join(model_save_path, f'Finetuned_Model_{fold}.pkl'))
                    scheduler.step(best_auc)
                
    end_datetime = datetime.now()
    time_diff = end_datetime - start_datetime
    time_diff_seconds = time_diff.total_seconds()
    print(f"Finetune Time for fold {fold}: {time_diff_seconds} seconds")
    

def multiclass_auc_with_all_classes(y_true, y_score, n_classes=8):
    y_score = np.array(y_score)
    
    aucs = []
    for c in range(n_classes):
        if np.any(y_true == c):  # class present
            auc = roc_auc_score((y_true == c).astype(int), y_score[:, c])
        else:  # class missing
            auc = np.nan
        aucs.append(auc)
    return aucs, np.nanmean(aucs)  # mean across all classes

def test(model, dataset, batch_size, multi_label, n_classes = 5):
    print("Start testing...")
    model.to(device)
    model.eval()
    testloader = DataLoader(dataset, batch_size=batch_size)

    pred_prob = []
    # label = []
    with torch.no_grad():
        for batch in testloader:
            *modalities, y = tuple(t.to(device) for t in batch)
            x = tuple(modalities)
            pred = model(x)
            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])

    pred_prob = np.array(pred_prob)

    _, auc = multiclass_auc_with_all_classes(dataset.label, pred_prob, n_classes=n_classes)
    
    
    pred_label = np.argmax(pred_prob, axis=1)
    
    F1 = f1_score(dataset.label, pred_label, average='macro', )
    acc = balanced_accuracy_score(dataset.label, pred_label, )
    
    print('AUC is {:.2f}'.format(auc * 100))
    print('F1 is', F1)
    print('acc is', acc)
    model.train()
    return auc, F1, acc, dataset.label

def val(model, dataset, batch_size, multi_label, n_classes = 5):
    model.to(device)
    model.eval()
    testloader = DataLoader(dataset, batch_size=batch_size)

    pred_prob = []
    with torch.no_grad():
        for batch in testloader:
            *modalities, y = tuple(t.to(device) for t in batch)
            x = tuple(modalities)
            pred = model(x)
            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])

    pred_prob = np.array(pred_prob)

    _, auc = multiclass_auc_with_all_classes(dataset.label, pred_prob, n_classes= n_classes)
    
    model.train()
    return auc, dataset.label

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", type=str2bool, default=False)
    parser.add_argument("--sl", type=str2bool, default=True)

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--in_dim", type=int, default=12)
    parser.add_argument("--dataset_name", type=str, default='SleepEDFx', choices=['SleepEDFx', 'SleepEDFx_3', 'PAMAP2', 'PAMAP2_3', 'UCI-HAR', 'UCI-HAR_total'])
    parser.add_argument("--model_save_path", type=str, default='.')
    parser.add_argument("--current_num_fold", type=int, default=1)
    parser.add_argument("--seeds", nargs='*', type=int, default=[0, 20, 40])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warm_epochs", "--warm_epoch", type=parse_warm_epochs, default=50)
    parser.add_argument("--adaptive_warmup_threshold", type=float, default=0.08, help='relative loss-drop threshold for ending adaptive warmup')
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument('--use_iteration', type=str2bool, default=False, help='whether to use iteration instead of epoches for training.')
    parser.add_argument('--num_iter', type=int, default=600, help='number of iterations for training')

    parser.add_argument('--final_out_channels', type=int, default=128)
    parser.add_argument('--output_dims', type=int, default=127)
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='validation ratio for splitting training and validation sets.')
    parser.add_argument('--filter_temporal', type=str2bool, default=True, help='whether to apply binary-guided filtering in temporal contrastive losses')
    parser.add_argument('--filter_intra', type=str2bool, default=True, help='whether to apply binary-guided filtering in intra-modal contrastive losses')
    parser.add_argument('--filter_inter', type=str2bool, default=True, help='whether to apply binary-guided filtering in inter-modal contrastive losses')
    parser.add_argument('--fn_filter_use_binary', type=str2bool, default=True, help='whether binary/BMM scores are used for false-negative attraction')
    parser.add_argument('--fn_filter_use_prior', type=str2bool, default=True, help='whether magnitude-prior probabilities are used for false-negative attraction when a prior is available')
    parser.add_argument('--replace_binary_with_bmm', type=str2bool, default=False, help='whether to replace neural binary classifiers with per-batch BMMs fitted on all pairwise similarities')
    parser.add_argument('--temporal_binary_mode', type=str, default=None, choices=['binary', 'bmm'], help='temporal branch selector: binary classifier or per-batch BMM')
    parser.add_argument('--intra_binary_mode', type=str, default=None, choices=['binary', 'bmm'], help='intra-modal branch selector: binary classifier or per-batch BMM')
    parser.add_argument('--inter_binary_mode', type=str, default=None, choices=['binary', 'bmm'], help='inter-modal branch selector: binary classifier or per-batch BMM')
    parser.add_argument('--three_mod_contrast', type=str, default='pairwise', choices=['pairwise', '1vsall'], help='inter-modality contrast mode for 3-modality inputs')
    parser.add_argument('--use_intra_sample_for_temporal_filter', type=str2bool, default=False, help='whether temporal filtering uses intra-modal sample-level binary/prior decisions instead of temporal segment-level binary/prior decisions')
    parser.add_argument('--use_prior', type=str2bool, default=False, help='whether to use log-RMS magnitude Euclidean-prior mixture filtering')
    parser.add_argument('--prior_mode', type=str, default='combined', choices=['combined', 'separate'], help='whether to fit/use a combined-modality prior or separate modality priors')
    parser.add_argument('--prior_model', type=str, default='bmm', choices=['bmm', 'gmm'], help='mixture model for prior distances; BMM scales distances to [0, 1], GMM uses raw distances')
    parser.add_argument('--prior_gmm_metric', type=str, default='euclidean', choices=['euclidean', 'cosine', 'cosine_euclidean'], help='pair feature used for prior mixture fitting: Euclidean distance, cosine similarity, or their concatenation')
    parser.add_argument('--prior_delta_mode', type=str, default='concat', choices=['none', 'delta', 'concat', 'both'], help='whether to use magnitude only, delta only, concatenated magnitude+delta, or separate magnitude/delta pair variables')
    parser.add_argument('--prior_fit_max_iter', type=int, default=200, help='maximum EM iterations for BMM/GMM prior fitting')
    parser.add_argument('--prior_plot', type=str2bool, default=False, help='whether to save distance histogram and fitted prior mixture overlay plots after fitting')
    parser.add_argument('--prior_save_dir', type=str, default=None, help='directory for saving/loading magnitude prior artifacts')
    parser.add_argument('--prior_hard_neg_weight', type=parse_prior_hard_neg_weight, default=1.0, help="negative weight for binary-positive pairs rejected by the prior mixture model; use 'auto' for 1 + 0.001 * binary_output * (1 - prior_prob)")
    parser.add_argument('--prior_cancel_weighting', type=str2bool, default=False, help='whether prior cancel pairs use 1 - prior_prob as weight; if false, use weight 1')
    parser.add_argument('--prior_num_random_pairs', type=int, default=3500, help='number of random off-diagonal pairs used to fit the prior mixture model')
    parser.add_argument('--prior_num_self_pairs', type=int, default=500, help='number of self-distance pairs used to fit the prior mixture model')
    parser.add_argument('--prior_segment_len', type=int, default=4, help='raw segment length used for RMS magnitude prior extraction')
    parser.add_argument('--fn_analysis', type=str2bool, default=False, help='whether to save offline false-negative attraction analysis during SSL pretraining')
    parser.add_argument('--fn_analysis_dir', type=str, default=None, help='root directory for false-negative attraction analysis outputs; each dataset/fold/seed gets its own subdirectory')
    parser.add_argument('--fn_analysis_every', type=int, default=1, help='analyze every N filtering-active epochs')
    parser.add_argument('--fn_analysis_threshold', type=float, default=0.5, help='probability threshold used for offline attraction analysis')
    
    opt = parser.parse_args()
    fallback_binary_mode = 'bmm' if opt.replace_binary_with_bmm else 'binary'
    opt.temporal_binary_mode = opt.temporal_binary_mode or fallback_binary_mode
    opt.intra_binary_mode = opt.intra_binary_mode or fallback_binary_mode
    opt.inter_binary_mode = opt.inter_binary_mode or fallback_binary_mode
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    dim = opt.dim
    in_dim = opt.in_dim
    root_save_path = opt.model_save_path


    print("final_out_channels", opt.final_out_channels)
    print("output_dims", opt.output_dims)

    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path, exist_ok=True)

    n_epoch = opt.epochs
    warm_epoch = opt.warm_epochs
    n_epoch_ft = 10
    lr = opt.lr

    use_iteration = opt.use_iteration
    print('Use iterations:', use_iteration)
    num_iter = opt.num_iter
    print('Number of iterations:', num_iter)

    batch_size = opt.batch_size
    if use_iteration:
        batch_size_ssl = 128
    else:
        batch_size_ssl = batch_size
    
    print("Batch size for pretraining:", batch_size_ssl)
    print("Batch size for finetuning:", batch_size)


    print("n_epoch: ", n_epoch)
    print("warm_epoch: ", warm_epoch)
    print("adaptive_warmup_threshold: ", opt.adaptive_warmup_threshold)
    print("n_epoch_ft: ", n_epoch_ft)
    print('lr', lr)
    print("batch_size", batch_size)
    print("filter_temporal:", opt.filter_temporal)
    print("filter_intra:", opt.filter_intra)
    print("filter_inter:", opt.filter_inter)
    print("fn_filter_use_binary:", opt.fn_filter_use_binary)
    print("fn_filter_use_prior:", opt.fn_filter_use_prior)
    print("replace_binary_with_bmm:", opt.replace_binary_with_bmm)
    print("temporal_binary_mode:", opt.temporal_binary_mode)
    print("intra_binary_mode:", opt.intra_binary_mode)
    print("inter_binary_mode:", opt.inter_binary_mode)
    print("three_mod_contrast:", opt.three_mod_contrast)
    print("use_intra_sample_for_temporal_filter:", opt.use_intra_sample_for_temporal_filter)
    print("use_prior:", opt.use_prior)
    print("prior_mode:", opt.prior_mode)
    print("prior_model:", opt.prior_model)
    print("prior_gmm_metric:", opt.prior_gmm_metric)
    print("prior_delta_mode:", opt.prior_delta_mode)
    print("prior_fit_max_iter:", opt.prior_fit_max_iter)
    print("prior_plot:", opt.prior_plot)
    print("prior_save_dir:", opt.prior_save_dir)
    print("prior_hard_neg_weight:", opt.prior_hard_neg_weight)
    print("prior_cancel_weighting:", opt.prior_cancel_weighting)
    print("prior_num_random_pairs:", opt.prior_num_random_pairs)
    print("prior_num_self_pairs:", opt.prior_num_self_pairs)
    print("prior_segment_len:", opt.prior_segment_len)
    print("fn_analysis:", opt.fn_analysis)
    print("fn_analysis_dir:", opt.fn_analysis_dir)
    print("fn_analysis_every:", opt.fn_analysis_every)
    print("fn_analysis_threshold:", opt.fn_analysis_threshold)

    if opt.dataset_name == 'SleepEDFx_3' or opt.dataset_name == 'PAMAP2_3':
        print("Using 3-modality input for dataset:", opt.dataset_name)
        # seeds = [0, 20, 42, 60, 80, 100, 120, 140, 160, 180]
        seeds = [0, 20, 42, 60, 80]
    else:
        seeds = opt.seeds  # use 3 seeds
    print(seeds)

    ssl = opt.ssl
    if not ssl:
        print("Only Finetuning!")
    
    file_parser = dict()

    # set dataset path and current fold
    dataset_name = opt.dataset_name
    loader_dataset_name = dataset_name.replace('_3', '')
    small_classifier = False
    transformer_depth = 4
    mod3_dims = None
    if loader_dataset_name == 'SleepEDFx':
        if dataset_name == 'SleepEDFx_3':
            print("Using 3-modality input for SleepEDFx, which includes EEG, EOG and EMG.")
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/SleepEDFx/SleepTelemetry_preprocessed'
            mod1_dims, mod2_dims, mod3_dims = 2, 1, 1
        else:
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette'
            mod1_dims, mod2_dims = 2, 1
        n_classes = 5
        time_steps = 50
        print("n_classes:", n_classes, "input_dims", [dim for dim in [mod1_dims, mod2_dims, mod3_dims] if dim is not None], "time_steps", time_steps)
    elif loader_dataset_name == 'PAMAP2':
        if dataset_name == 'PAMAP2_3':
            print("Using 3-modality input for PAMAP2, which includes IMU on hand, IMU on chest and IMU on ankle.")
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_processed_3modality'
            mod1_dims, mod2_dims, mod3_dims = 9, 9, 9
        else:
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_normalized_9classes'
            mod1_dims, mod2_dims = 9, 9
        n_classes = 9
        time_steps = 5 #1
        print("n_classes:", n_classes, "input_dims", [dim for dim in [mod1_dims, mod2_dims, mod3_dims] if dim is not None], "time_steps", time_steps)
    elif loader_dataset_name in ['UCI-HAR', 'UCI-HAR_total']:
        if dataset_name == 'UCI-HAR_total':
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR_total'
        else:
            dataset_path = '/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR'
        n_classes = 6
        mod1_dims = 3
        mod2_dims = 3
        time_steps = 3 #1
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
    
    print("Dataset path:", dataset_path)

    file_parser["filepath"] = dataset_path
    file_parser["Fold"] = opt.current_num_fold
    print("Current Fold: ", file_parser["Fold"])

    for seed in seeds:
        print(f'=================Current Seed: {seed}=================')
        set_seed(seed)
        model_save_path = os.path.join(root_save_path, str(seed))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        fn_analysis_root = opt.fn_analysis_dir or os.path.join(model_save_path, "fn_analysis")
        fn_analysis_dir = os.path.join(
            fn_analysis_root,
            f"{dataset_name}_fold{opt.current_num_fold}_seed{seed}",
        )
        

        i = opt.current_num_fold


        path, path_name = get_path_loader_new(file_parser["filepath"], dataset_name=dataset_name)

        
        pretrain_test_auc = []
        pretrain_test_F1 = []
        pretrain_test_acc = []
        finetune_test_auc = []
        finetune_test_F1 = []
        finetune_test_acc = []
        
        # Data initialization
        pretrain_data, pretrain_label, finetune_train_data, finetune_train_label, finetune_val_data, finetune_val_label, test_data, test_label = get_loader(file_parser, path, path_name, dataset_name, valid_ratio=opt.valid_ratio)
        prior_info = None
        if opt.use_prior:
            prior_save_dir = opt.prior_save_dir or os.path.join(model_save_path, "prior_cache")
            prior_info = load_or_build_magnitude_prior(
                pretrain_data,
                dataset_name,
                fold=i,
                seed=seed,
                save_dir=prior_save_dir,
                segment_len=opt.prior_segment_len,
                num_random_pairs=opt.prior_num_random_pairs,
                num_self_pairs=opt.prior_num_self_pairs,
                prior_mode=opt.prior_mode,
                prior_model=opt.prior_model,
                prior_gmm_metric=opt.prior_gmm_metric,
                prior_delta_mode=opt.prior_delta_mode,
                prior_fit_max_iter=opt.prior_fit_max_iter,
                three_mod_contrast=opt.three_mod_contrast,
                plot=opt.prior_plot,
            )
            
        # model initialization
        print("n_classes:", n_classes, "input_dims", [dim for dim in [mod1_dims, mod2_dims, mod3_dims] if dim is not None])
        model = tfcc_model(
            mod1_dims=mod1_dims,
            mod2_dims=mod2_dims,
            mod3_dims=mod3_dims,
            output_dims=opt.output_dims,
            device=device,
            num_class=n_classes,
            final_out_channels = opt.final_out_channels, 
            kernel_size = 25, 
            stride = 3, 
            timesteps=time_steps, 
            hidden_dim=64,
            batch_size=batch_size,
            num_epochs=n_epoch,
            warm_epochs=warm_epoch,
            adaptive_warmup_threshold=opt.adaptive_warmup_threshold,
            lr = lr,
            use_iteration=use_iteration,
            num_iter=num_iter,
            depth=4,
            filter_temporal=opt.filter_temporal,
            filter_intra=opt.filter_intra,
            filter_inter=opt.filter_inter,
            fn_filter_use_binary=opt.fn_filter_use_binary,
            fn_filter_use_prior=opt.fn_filter_use_prior,
            replace_binary_with_bmm=opt.replace_binary_with_bmm,
            temporal_binary_mode=opt.temporal_binary_mode,
            intra_binary_mode=opt.intra_binary_mode,
            inter_binary_mode=opt.inter_binary_mode,
            use_intra_sample_for_temporal_filter=opt.use_intra_sample_for_temporal_filter,
            use_prior=opt.use_prior,
            prior_info=prior_info,
            prior_hard_neg_weight=opt.prior_hard_neg_weight,
            prior_cancel_weighting=opt.prior_cancel_weighting,
            pretrain_labels=pretrain_label,
            contrast_mode=opt.three_mod_contrast,
            fn_analysis=opt.fn_analysis,
            fn_analysis_dir=fn_analysis_dir,
            fn_analysis_every=opt.fn_analysis_every,
            fn_analysis_threshold=opt.fn_analysis_threshold,
            fn_analysis_context={
                "dataset_name": dataset_name,
                "fold": i,
                "seed": seed,
            },
        )
        
        if ssl:
            # Pretraining
            self_supervised_learning(dataset_name, i, model, pretrain_data, batch_size_ssl, device, model_save_path, use_iteration=use_iteration)
            
        
        pretrained_weights = os.path.join(model_save_path, f'Pretrained_Model_{i}.pkl')
        model.load_state_dict(torch.load(pretrained_weights))
        
        # # Testing before finetuning
        model.eval()
        TestSet = FTDataSet(test_data, test_label, dataset_name, multi_label=False)
        auc, F1, acc, _ = test(model, TestSet, batch_size=100, multi_label=False, n_classes=n_classes)
        pretrain_test_auc.append(auc)
        pretrain_test_F1.append(F1)
        pretrain_test_acc.append(acc)
        
        model.train()


        # Finetuning
        TrainSet = FTDataSet(finetune_train_data, finetune_train_label, dataset_name, multi_label=False) # multi_label = True when the dataset is PTBXL
        ValidSet = FTDataSet(finetune_val_data, finetune_val_label, dataset_name, multi_label=False)
        finetuning(i, model, TrainSet, ValidSet, n_epoch_ft, lr, batch_size, device, model_save_path, multi_label=False, n_classes=n_classes)
        
        # Testing
        finetune_weights = torch.load(os.path.join(model_save_path, f'Finetuned_Model_{i}.pkl'), map_location=device)
        model.load_state_dict(finetune_weights, strict=False)
        
        model.eval()
        auc1, F11, acc1, _ = test(model, TestSet, batch_size=100, multi_label=False, n_classes=n_classes)
        finetune_test_auc.append(auc1)
        finetune_test_F1.append(F11)
        finetune_test_acc.append(acc1)
        

        # print out the 10 fold results
        
        print(f"=================Fold {i} Seed {seed} Results=================")
        print(f"Pretraining Test AUC: {pretrain_test_auc}, F1: {pretrain_test_F1}, Acc: {pretrain_test_acc}")
        print(f"Finetuning Test AUC: {finetune_test_auc}, F1: {finetune_test_F1}, Acc: {finetune_test_acc}")
        print(f"===================================================")

        
