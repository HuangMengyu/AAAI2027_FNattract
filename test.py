import os
import numpy as np
from torch.utils.data import DataLoader
import torch
from common_utils.dataset_loader import get_path_loader_new, get_idx
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from base_models import FTDataSet
import argparse
from datetime import datetime

from model import Model as tfcc_model

import torch.nn as nn
import torch.nn.functional as F



def metric_summary(values):
    values = np.array(values, dtype=float)
    return float(np.mean(values)), float(np.std(values))


def make_unique_names(names):
    counts = {}
    unique_names = []
    for name in names:
        base_name = name or "checkpoint"
        counts[base_name] = counts.get(base_name, 0) + 1
        if counts[base_name] == 1:
            unique_names.append(base_name)
        else:
            unique_names.append(f"{base_name}_{counts[base_name]}")
    return unique_names


def resolve_checkpoint_specs(opt):
    if not opt.checkpoint_path:
        raise ValueError("Please pass --checkpoint_path.")

    checkpoint_name = opt.checkpoint_name or os.path.basename(os.path.normpath(opt.checkpoint_path)) or "checkpoint"
    checkpoint_specs = [(checkpoint_name, opt.checkpoint_path)]

    if opt.baseline_checkpoint_path:
        baseline_name = opt.baseline_checkpoint_name or os.path.basename(os.path.normpath(opt.baseline_checkpoint_path)) or "baseline"
        checkpoint_specs.append((baseline_name, opt.baseline_checkpoint_path))

    unique_names = make_unique_names([name for name, _ in checkpoint_specs])
    return list(zip(unique_names, [path for _, path in checkpoint_specs]))


def run_paired_ttest(a_scores, b_scores, alternative="greater"):
    a_scores = np.array(a_scores, dtype=float)
    b_scores = np.array(b_scores, dtype=float)
    if a_scores.shape != b_scores.shape:
        raise ValueError(f"Paired scores must have the same shape, got {a_scores.shape} and {b_scores.shape}.")

    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"Unsupported alternative '{alternative}'. Use 'greater', 'less', or 'two-sided'.")

    valid_mask = np.isfinite(a_scores) & np.isfinite(b_scores)
    diff = a_scores[valid_mask] - b_scores[valid_mask]
    n = int(diff.size)
    mean_diff = float(np.mean(diff)) if n else np.nan
    if n < 2:
        return {"n": n, "mean_diff": mean_diff, "t": np.nan, "p": np.nan}

    diff_std = float(np.std(diff, ddof=1))
    if np.allclose(diff, 0):
        p_value = 1.0 if alternative == "two-sided" else 0.5
        return {"n": n, "mean_diff": 0.0, "t": 0.0, "p": p_value}

    if np.isclose(diff_std, 0.0):
        t_stat = np.inf if mean_diff > 0 else -np.inf
    else:
        t_stat = mean_diff / (diff_std / np.sqrt(n))

    df = n - 1
    if alternative == "greater":
        p_value = student_t.sf(t_stat, df)
    elif alternative == "less":
        p_value = student_t.cdf(t_stat, df)
    else:
        p_value = 2 * student_t.sf(abs(t_stat), df)

    return {
        "n": n,
        "mean_diff": mean_diff,
        "t": float(t_stat),
        "p": float(p_value),
    }


def run_wilcoxon_signed_rank(a_scores, b_scores, alternative="greater"):
    a_scores = np.array(a_scores, dtype=float)
    b_scores = np.array(b_scores, dtype=float)
    if a_scores.shape != b_scores.shape:
        raise ValueError(f"Paired scores must have the same shape, got {a_scores.shape} and {b_scores.shape}.")

    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"Unsupported alternative '{alternative}'. Use 'greater', 'less', or 'two-sided'.")

    valid_mask = np.isfinite(a_scores) & np.isfinite(b_scores)
    diff = a_scores[valid_mask] - b_scores[valid_mask]
    nonzero_diff = diff[~np.isclose(diff, 0.0)]
    n = int(nonzero_diff.size)
    mean_diff = float(np.mean(diff)) if diff.size else np.nan
    if n == 0:
        return {"n": n, "statistic": 0.0, "p": 1.0}

    result = wilcoxon(
        nonzero_diff,
        alternative=alternative,
        zero_method="wilcox",
        correction=False,
    )
    return {
        "n": n,
        "statistic": float(result.statistic),
        "p": float(result.pvalue),
        "mean_diff": mean_diff,
    }


def multiclass_auc_with_all_classes(y_true, y_score, n_classes=5):
    y_score = np.array(y_score)
    aucs = []
    for c in range(n_classes):
        if np.any(y_true == c):  # class present
            auc = roc_auc_score((y_true == c).astype(int), y_score[:, c])
        else:  # class missing
            auc = np.nan
        aucs.append(auc)
    return aucs, np.nanmean(aucs)  # mean across all classes


def get_test_loader_batch(path_name, test_idx):
    test_data = []
    test_label = []

    for t_idx in test_idx:
        for i in range(len(path_name[t_idx])):
            temp_data = np.load(path_name[t_idx][i][0], mmap_mode='r')
            temp_label = np.load(path_name[t_idx][i][1], mmap_mode='r')
            test_data.append(temp_data)
            test_label.append(temp_label)
    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    assert test_data.shape[0] == test_label.shape[0], "test data and test size do not match!"
    print("Number of test samples:", test_data.shape)

    return test_data, test_label

def test(model, dataset, batch_size, multi_label, device, n_classes=5):
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
        
    _, auc = multiclass_auc_with_all_classes(dataset.label, pred_prob, n_classes=n_classes )
    
    pred_label = np.argmax(pred_prob, axis=1)

    F1 = f1_score(dataset.label, pred_label, average='macro', )
    acc = balanced_accuracy_score(dataset.label, pred_label, )
    
    print('AUC is {:.2f}'.format(auc * 100))
    print('F1 is', F1)
    print('acc is', acc)
    model.train()
    return auc, F1, acc


def evaluate_checkpoint_root(
    checkpoint_name,
    checkpoint_path,
    opt,
    parser,
    path,
    path_name,
    model,
    device,
    n_classes,
    dataset_name,
    num_folds,
    seeds
):
    print(f"Evaluating Checkpoint path [{checkpoint_name}]: {checkpoint_path}")
    acc_results = []
    f1_results = []
    auc_results = []
    fold_scores = {metric: [] for metric in ["auc", "acc", "F1"]}
    seed_scores = {metric: [] for metric in ["auc", "acc", "F1"]}

    for seed in seeds:
        models_save_path = os.path.join(checkpoint_path, f"{seed}")
        acc_results_seed = []
        f1_results_seed = []
        auc_results_seed = []

        for fold in range(1, num_folds + 1):  # only evaluate the finetuned model here
            print(f"Evaluating checkpoint {checkpoint_name}, seed {seed}, fold {fold}...")
            parser["Fold"] = fold
            _, _, test_idx = get_idx(parser, path, dataset_name)
            test_data, test_label = get_test_loader_batch(path_name, test_idx)

            TestSet = FTDataSet(test_data, test_label, dataset_name, multi_label=False)
            checkpoint_file = os.path.join(models_save_path, f'Finetuned_Model_{fold}.pkl')
            weights = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(weights, strict=False)  # in case some keys are missing

            auc, F1, acc = test(model, TestSet, batch_size=opt.batch_size, multi_label=False, device=device, n_classes=n_classes)

            print("Checkpoint: {}, Seed: {}, Fold: {}, Test Auc: {:.4f}, Test Acc: {:.4f}, Test MF1: {:.4f}".format(checkpoint_name, seed, fold, auc, acc, F1))
            
            acc_results_seed.append(acc)
            f1_results_seed.append(F1)
            auc_results_seed.append(auc)
            fold_scores["auc"].append(auc)
            fold_scores["acc"].append(acc)
            fold_scores["F1"].append(F1)

        seed_acc_mean = float(np.mean(acc_results_seed))
        seed_f1_mean = float(np.mean(f1_results_seed))
        seed_auc_mean = float(np.mean(auc_results_seed))
        acc_results.append(seed_acc_mean)
        f1_results.append(seed_f1_mean)
        auc_results.append(seed_auc_mean)
        seed_scores["auc"].append(seed_auc_mean)
        seed_scores["acc"].append(seed_acc_mean)
        seed_scores["F1"].append(seed_f1_mean)

        print(f"Checkpoint {checkpoint_name}, seed {seed} average results: Acc {seed_acc_mean:.4f}, MF1 {seed_f1_mean:.4f}, AUC {seed_auc_mean:.4f}")
        

    acc_mean, acc_std = metric_summary(acc_results)
    f1_mean, f1_std = metric_summary(f1_results)
    auc_mean, auc_std = metric_summary(auc_results)
    print(f"Checkpoint {checkpoint_name} all seeds average results: Acc {acc_mean:.4f} ± {acc_std:.4f}, MF1 {f1_mean:.4f} ± {f1_std:.4f}, AUC {auc_mean:.4f} ± {auc_std:.4f}")
    

    return {
        "fold_scores": fold_scores,
        "seed_scores": seed_scores,
        "summary": {
            "acc": {"mean": acc_mean, "std": acc_std},
            "F1": {"mean": f1_mean, "std": f1_std},
            "auc": {"mean": auc_mean, "std": auc_std},
        },
    }


def run_pairwise_significance(results_by_checkpoint, alpha=0.05):
    checkpoint_names = list(results_by_checkpoint.keys())
    if len(checkpoint_names) < 2:
        print("Pairwise significance test skipped: at least two checkpoint paths are required.")
        return

    print(f"Pairwise paired significance results, one-sided alpha={alpha}")
    for score_level, level_label in [("fold_scores", "seed-fold scores"), ("seed_scores", "seed-mean scores")]:
        print(f"Score level: {level_label}")
        for i in range(len(checkpoint_names)):
            for j in range(i + 1, len(checkpoint_names)):
                name_a = checkpoint_names[i]
                name_b = checkpoint_names[j]
                print(f"Directional alternative: {name_a} > {name_b}")
                for metric in ["auc", "acc", "F1"]:
                    scores_a = results_by_checkpoint[name_a][score_level][metric]
                    scores_b = results_by_checkpoint[name_b][score_level][metric]
                    ttest_result = run_paired_ttest(scores_a, scores_b, alternative="greater")
                    wilcoxon_result = run_wilcoxon_signed_rank(scores_a, scores_b, alternative="greater")
                    improved = ttest_result["mean_diff"] > 0
                    ttest_significant = bool(improved and np.isfinite(ttest_result["p"]) and ttest_result["p"] < alpha)
                    wilcoxon_significant = bool(improved and np.isfinite(wilcoxon_result["p"]) and wilcoxon_result["p"] < alpha)
                    mean_a = float(np.mean(scores_a))
                    mean_b = float(np.mean(scores_b))
                    print(
                        f"{name_a} vs {name_b} | {metric} | n={ttest_result['n']} | "
                        f"mean_a={mean_a:.4f} | mean_b={mean_b:.4f} | "
                        f"diff={ttest_result['mean_diff']:.4f} | "
                        f"t={ttest_result['t']:.4f} | t_p_greater={ttest_result['p']:.6f} | "
                        f"t_significant={ttest_significant} | "
                        f"wilcoxon_n={wilcoxon_result['n']} | "
                        f"wilcoxon_W={wilcoxon_result['statistic']:.4f} | "
                        f"wilcoxon_p_greater={wilcoxon_result['p']:.6f} | "
                        f"wilcoxon_significant={wilcoxon_significant}"
                    )
                    

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def main():
    # initialize the args
    parser = dict()
    parser["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser["Fold"] = 1
    in_parser = argparse.ArgumentParser()
    in_parser.add_argument('--dataset_name', type=str, default='SleepEDFx', help='dataset name')
    in_parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path for one model checkpoint root')
    in_parser.add_argument('--checkpoint_name', type=str, default=None, help='optional display name for --checkpoint_path')
    in_parser.add_argument('--baseline_checkpoint_path', type=str, default=None, help='optional baseline checkpoint root for paired t-test')
    in_parser.add_argument('--baseline_checkpoint_name', type=str, default=None, help='optional display name for --baseline_checkpoint_path')
    in_parser.add_argument('--seeds', nargs='*', type=int, default=[0, 20, 40], help='seed folders to evaluate')
    in_parser.add_argument('--batch_size', type=int, default=100, help='batch size for testing')
    in_parser.add_argument('--significance_alpha', type=float, default=0.05, help='alpha threshold for paired t-tests')
    

    opt = in_parser.parse_args()
    
    if opt.dataset_name == 'SleepEDFx_3' or opt.dataset_name == 'PAMAP2_3':
        print("Using 3-modality input for dataset:", opt.dataset_name)
        # opt.seeds = [0, 20, 42, 60, 80, 100, 120, 140, 160, 180]
        opt.seeds = [0, 20, 42, 60, 80]
    print(opt.seeds)
    
    checkpoint_specs = resolve_checkpoint_specs(opt)


    small_classifier = False
    dataset_name = opt.dataset_name
    loader_dataset_name = dataset_name.replace('_3', '')
    mod3_dims = None
    if loader_dataset_name == 'SleepEDFx':
        if dataset_name == 'SleepEDFx_3':
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/SleepEDFx/SleepTelemetry_preprocessed"
            mod1_dims, mod2_dims, mod3_dims = 2, 1, 1
        else:
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette"
            mod1_dims, mod2_dims = 2, 1
        n_classes = 5
        time_steps = 50
        print("n_classes:", n_classes, "input_dims", [dim for dim in [mod1_dims, mod2_dims, mod3_dims] if dim is not None], "time_steps", time_steps)
    elif loader_dataset_name == 'PAMAP2':
        if dataset_name == 'PAMAP2_3':
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_processed_3modality"
            mod1_dims, mod2_dims, mod3_dims = 9, 9, 9
        else:
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_normalized_9classes"
            mod1_dims, mod2_dims = 9, 9
        n_classes = 9
        time_steps = 5
        print("n_classes:", n_classes, "input_dims", [dim for dim in [mod1_dims, mod2_dims, mod3_dims] if dim is not None], "time_steps", time_steps)
    elif loader_dataset_name in ['UCI-HAR', 'UCI-HAR_total']:
        if dataset_name == 'UCI-HAR_total':
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR_total"
        else:
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR"
        n_classes = 6
        mod1_dims = 3
        mod2_dims = 3
        time_steps = 3
        print("n_classes:", n_classes, "input_dims", mod1_dims, mod2_dims, "time_steps", time_steps)
    
    print("Evaluating Checkpoint paths:")
    for checkpoint_name, checkpoint_path in checkpoint_specs:
        print(f"  {checkpoint_name}: {checkpoint_path}")
    

    if dataset_name in ['PAMAP2', 'PAMAP2_3']:
        num_folds = 4
    else:
        num_folds = 5
    
    device = parser["device"]
    path, path_name = get_path_loader_new(input_path, dataset_name)

    model = tfcc_model(
            mod1_dims=mod1_dims,
            mod2_dims=mod2_dims,
            mod3_dims=mod3_dims,
            device=device,
            num_class=n_classes,
            timesteps=time_steps, 
        ).to(device)


    results_by_checkpoint = {}
    for checkpoint_name, checkpoint_path in checkpoint_specs:
        results_by_checkpoint[checkpoint_name] = evaluate_checkpoint_root(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            opt=opt,
            parser=parser,
            path=path,
            path_name=path_name,
            model=model,
            device=device,
            n_classes=n_classes,
            dataset_name=dataset_name,
            num_folds=num_folds,
            seeds=opt.seeds
        )

    if opt.baseline_checkpoint_path:
        run_pairwise_significance(results_by_checkpoint, alpha=opt.significance_alpha)



if __name__ == '__main__':
    main()
