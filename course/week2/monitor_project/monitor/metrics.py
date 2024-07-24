import torch
import numpy as np
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_ks_score(tr_probs, te_probs):
    logging.info("Computing KS score")
    tr_probs_np = tr_probs.numpy()
    te_probs_np = te_probs.numpy()
    ks_stat, p_value = ks_2samp(tr_probs_np, te_probs_np)
    logging.info(f"KS score computed: {p_value}")
    return p_value

def get_hist_score(tr_probs, te_probs, bins=10):
    logging.info("Computing histogram score")
    tr_heights, bin_edges = np.histogram(tr_probs.numpy(), bins=bins, density=True)
    te_heights, _ = np.histogram(te_probs.numpy(), bins=bin_edges, density=True)
    score = 0
    for i in range(len(tr_heights)):
        bin_diff = bin_edges[i+1] - bin_edges[i]
        tr_area = bin_diff * tr_heights[i]
        te_area = bin_diff * te_heights[i]
        intersect = min(tr_area, te_area)
        score += intersect
    logging.info(f"Histogram score computed: {score}")
    return score

def get_vocab_outlier(tr_vocab, te_vocab):
    logging.info("Computing vocabulary outlier score")
    # num_seen = sum(1 for word in te_vocab.keys() if word in tr_vocab.keys())
    # num_seen = sum(1 for word in te_vocab if word in tr_vocab)
    num_seen = len(set(te_vocab.keys()).intersection(set(tr_vocab.keys())))
    num_total = len(te_vocab)
    
    if num_total == 0:
        logging.warning("Test vocabulary is empty, returning score of 0.0")
        return 0.0
    
    score = 1 - (num_seen / num_total)
    logging.info(f"Vocabulary outlier score computed: {score}")
    return score



class MonitoringSystem:

    def __init__(self, tr_vocab, tr_probs, tr_labels):
        self.tr_vocab = tr_vocab
        self.tr_probs = tr_probs
        self.tr_labels = tr_labels
        logging.info("MonitoringSystem initialized")

    def calibrate(self, tr_probs, tr_labels, te_probs):
        logging.info("Calibrating probabilities")
        ir = IsotonicRegression(out_of_bounds='clip')
        tr_probs_cal_np = ir.fit_transform(tr_probs.numpy(), tr_labels.numpy())
        te_probs_cal_np = ir.transform(te_probs.numpy())
        tr_probs_cal = torch.tensor(tr_probs_cal_np, dtype=torch.float32)
        te_probs_cal = torch.tensor(te_probs_cal_np, dtype=torch.float32)
        logging.info("Calibration complete")
        return tr_probs_cal, te_probs_cal

    def monitor(self, te_vocab, te_probs):
        logging.info("Starting monitoring")
        tr_probs, te_probs = self.calibrate(self.tr_probs, self.tr_labels, te_probs)
        ks_score = get_ks_score(tr_probs, te_probs)
        hist_score = get_hist_score(tr_probs, te_probs)
        outlier_score = get_vocab_outlier(self.tr_vocab, te_vocab)
        metrics = {
            'ks_score': ks_score,
            'hist_score': hist_score,
            'outlier_score': outlier_score,
        }
        logging.info(f"Monitoring complete: {metrics}")
        return metrics
