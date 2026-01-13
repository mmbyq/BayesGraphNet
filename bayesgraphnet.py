"""
BayesGraphNet: A Unified Bayesian Framework for Community Detection and Feature Selection
in High-Dimensional Attributed Networks

Complete implementation with model training, testing, and evaluation.
"""

import numpy as np
from scipy.special import digamma, logsumexp
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class BayesGraphNetConfig:
    """Configuration for BayesGraphNet model"""
    n_communities: int = 5
    alpha_prior: float = 1.0
    eta_prior: float = 1.0
    rho_a: float = 1.0
    rho_b: float = 9.0
    tau_0_sq: float = 0.1
    a_sigma: float = 1.0
    b_sigma: float = 1.0
    learning_rate: float = 0.01
    max_iterations: int = 500
    convergence_tol: float = 1e-4
    patience: int = 50
    n_restarts: int = 3
    batch_size: int = 256
    use_stochastic: bool = True
    verbose: bool = True

@dataclass
class BayesGraphNetResults:
    """Results container"""
    community_assignments: np.ndarray
    community_probabilities: np.ndarray
    feature_relevance: np.ndarray
    selected_features: np.ndarray
    community_proportions: np.ndarray
    block_matrix: np.ndarray
    feature_coefficients: np.ndarray
    elbo_history: List[float]
    n_iterations: int
    convergence_achieved: bool
    training_time: float

# =============================================================================
# Data Generation
# =============================================================================

def generate_attributed_network(
    n_nodes: int = 1000, n_communities: int = 5, n_features: int = 500,
    n_relevant_features: int = 50, within_prob: float = 0.3,
    between_prob: float = 0.05, feature_signal: float = 2.0,
    noise_std: float = 1.0, degree_heterogeneity: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic attributed network with ground truth communities."""
    np.random.seed(random_state)
    
    community_sizes = np.random.multinomial(n_nodes, np.ones(n_communities) / n_communities)
    true_labels = np.repeat(np.arange(n_communities), community_sizes)
    perm = np.random.permutation(n_nodes)
    true_labels = true_labels[perm]
    
    degree_propensity = np.random.gamma(1.0 / degree_heterogeneity, degree_heterogeneity, n_nodes)
    degree_propensity = degree_propensity / np.mean(degree_propensity)
    
    adjacency = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if true_labels[i] == true_labels[j]:
                prob = within_prob * np.sqrt(degree_propensity[i] * degree_propensity[j])
            else:
                prob = between_prob * np.sqrt(degree_propensity[i] * degree_propensity[j])
            prob = np.clip(prob, 0, 1)
            if np.random.random() < prob:
                adjacency[i, j] = adjacency[j, i] = 1
    
    relevant_features = np.random.choice(n_features, n_relevant_features, replace=False)
    community_means = np.random.randn(n_communities, n_relevant_features) * feature_signal
    features = np.random.randn(n_nodes, n_features) * noise_std
    
    for i in range(n_nodes):
        features[i, relevant_features] += community_means[true_labels[i]]
    
    return adjacency, features, true_labels, relevant_features

def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset by name."""
    configs = {
        'synthetic_small': (500, 4, 200, 30),
        'synthetic_medium': (2000, 6, 500, 50),
        'synthetic_large': (5000, 8, 1000, 100),
        'synthetic_ppi': (3000, 7, 1024, 80),
        'synthetic_citation': (4000, 10, 512, 60),
        'synthetic_social': (3500, 8, 1536, 120),
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    n, k, f, r = configs[dataset_name]
    adj, feat, labels, _ = generate_attributed_network(n, k, f, r)
    return adj, feat, labels

def preprocess_data(adjacency: np.ndarray, features: np.ndarray,
                   normalize_features: bool = True,
                   laplacian_smoothing: bool = True,
                   smoothing_alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess network data."""
    if normalize_features:
        features = StandardScaler().fit_transform(features)
    
    if laplacian_smoothing:
        n = adjacency.shape[0]
        degree = np.sum(adjacency, axis=1)
        laplacian = np.diag(degree) - adjacency
        smoothing_matrix = np.eye(n) + smoothing_alpha * laplacian
        try:
            features = np.linalg.solve(smoothing_matrix, features)
        except:
            features = np.linalg.lstsq(smoothing_matrix, features, rcond=None)[0]
    
    return adjacency, features

# =============================================================================
# BayesGraphNet Model
# =============================================================================

class BayesGraphNet:
    """BayesGraphNet: Unified Bayesian Framework for Community Detection and Feature Selection."""
    
    def __init__(self, config: BayesGraphNetConfig = None):
        self.config = config if config else BayesGraphNetConfig()
        self.is_fitted = False
        self.n_nodes = self.n_features = None
        self.phi = self.psi = self.gamma_pi = None
        self.theta_alpha = self.theta_beta = None
        self.mu_beta = self.sigma_beta = None
        self.a_sigma = self.b_sigma = None
        self.lambda_sq = self.tau_sq = None
        self.results = None
        self.elbo_history = []
    
    def _initialize_params(self, adjacency: np.ndarray, features: np.ndarray):
        """Initialize variational parameters."""
        self.n_nodes, self.n_features = adjacency.shape[0], features.shape[1]
        K = self.config.n_communities
        
        try:
            sc = SpectralClustering(n_clusters=K, affinity='precomputed',
                                   random_state=np.random.randint(10000))
            init_labels = sc.fit_predict(adjacency + np.eye(self.n_nodes) * 0.1)
        except:
            init_labels = np.random.randint(0, K, self.n_nodes)
        
        self.phi = np.ones((self.n_nodes, K)) * 0.1 / K
        for i in range(self.n_nodes):
            self.phi[i, init_labels[i]] = 0.9
        self.phi /= self.phi.sum(axis=1, keepdims=True)
        
        self.gamma_pi = np.ones(K) * self.config.alpha_prior + np.sum(self.phi, axis=0)
        self.theta_alpha = np.ones((K, K)) * self.config.eta_prior
        self.theta_beta = np.ones((K, K)) * self.config.eta_prior
        
        for k1 in range(K):
            for k2 in range(K):
                m1, m2 = self.phi[:, k1] > 0.5, self.phi[:, k2] > 0.5
                if np.sum(m1) > 0 and np.sum(m2) > 0:
                    n_edges = np.sum(adjacency[np.ix_(m1, m2)])
                    n_possible = np.sum(m1) * np.sum(m2)
                    if k1 == k2:
                        n_possible = np.sum(m1) * (np.sum(m1) - 1) / 2
                    if n_possible > 0:
                        p = n_edges / max(n_possible, 1)
                        self.theta_alpha[k1, k2] = p * 10 + 1
                        self.theta_beta[k1, k2] = (1 - p) * 10 + 1
        
        self.psi = np.ones(self.n_features) * 0.1
        self.tau_sq = np.ones(self.n_features)
        self.lambda_sq = np.ones((K, self.n_features))
        self.mu_beta = np.zeros((K, self.n_features))
        self.sigma_beta = np.ones((K, self.n_features)) * 0.1
        
        for k in range(K):
            mask = self.phi[:, k] > 0.5
            if np.sum(mask) > 0:
                self.mu_beta[k] = np.mean(features[mask], axis=0)
        
        self.a_sigma = np.ones(self.n_features) * (self.config.a_sigma + self.n_nodes / 2)
        self.b_sigma = np.ones(self.n_features) * self.config.b_sigma
    
    def _E_log_pi(self):
        return digamma(self.gamma_pi) - digamma(np.sum(self.gamma_pi))
    
    def _E_log_theta(self):
        E1 = digamma(self.theta_alpha) - digamma(self.theta_alpha + self.theta_beta)
        E2 = digamma(self.theta_beta) - digamma(self.theta_alpha + self.theta_beta)
        return E1, E2
    
    def _E_sigma_inv(self):
        return self.a_sigma / self.b_sigma
    
    def _update_phi(self, adjacency, features, batch=None):
        if batch is None:
            batch = np.arange(self.n_nodes)
        K = self.config.n_communities
        E_log_pi = self._E_log_pi()
        E_log_theta, E_log_1_theta = self._E_log_theta()
        E_sigma_inv = self._E_sigma_inv()
        
        for i in batch:
            log_phi = np.zeros(K)
            for k in range(K):
                log_phi[k] = E_log_pi[k]
                neighbors = np.where(adjacency[i] > 0)[0]
                non_neighbors = np.where(adjacency[i] == 0)[0]
                non_neighbors = non_neighbors[non_neighbors != i][:100]
                
                for j in neighbors:
                    log_phi[k] += np.sum(self.phi[j] * E_log_theta[k])
                for j in non_neighbors:
                    log_phi[k] += np.sum(self.phi[j] * E_log_1_theta[k])
                
                rel_feats = np.where(self.psi > 0.01)[0]
                for j in rel_feats:
                    res = features[i, j] - self.mu_beta[k, j]
                    log_phi[k] -= 0.5 * self.psi[j] * E_sigma_inv[j] * (res**2 + self.sigma_beta[k, j])
            
            log_phi -= logsumexp(log_phi)
            self.phi[i] = np.clip(np.exp(log_phi), 1e-10, 1 - 1e-10)
            self.phi[i] /= self.phi[i].sum()
    
    def _update_psi(self, features):
        K = self.config.n_communities
        E_log_rho = digamma(self.config.rho_a) - digamma(self.config.rho_a + self.config.rho_b)
        E_log_1_rho = digamma(self.config.rho_b) - digamma(self.config.rho_a + self.config.rho_b)
        E_sigma_inv = self._E_sigma_inv()
        
        for j in range(self.n_features):
            log_odds = E_log_rho - E_log_1_rho
            for i in range(self.n_nodes):
                for k in range(K):
                    res = features[i, j] - self.mu_beta[k, j]
                    ll_rel = -0.5 * E_sigma_inv[j] * (res**2 + self.sigma_beta[k, j])
                    ll_irr = -0.5 * (features[i, j]**2) / self.config.tau_0_sq
                    log_odds += self.phi[i, k] * (ll_rel - ll_irr)
            self.psi[j] = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -30, 30)))
    
    def _update_gamma_pi(self):
        self.gamma_pi = self.config.alpha_prior + np.sum(self.phi, axis=0)
    
    def _update_theta(self, adjacency):
        K = self.config.n_communities
        for k1 in range(K):
            for k2 in range(k1, K):
                exp_edges = exp_non = 0
                for i in range(self.n_nodes):
                    for j in range(i + 1, self.n_nodes):
                        p = self.phi[i, k1] * self.phi[j, k2]
                        if k1 != k2:
                            p += self.phi[i, k2] * self.phi[j, k1]
                        if adjacency[i, j] > 0:
                            exp_edges += p
                        else:
                            exp_non += p
                self.theta_alpha[k1, k2] = self.config.eta_prior + exp_edges
                self.theta_beta[k1, k2] = self.config.eta_prior + exp_non
                if k1 != k2:
                    self.theta_alpha[k2, k1] = self.theta_alpha[k1, k2]
                    self.theta_beta[k2, k1] = self.theta_beta[k1, k2]
    
    def _update_beta(self, features):
        K = self.config.n_communities
        E_sigma_inv = self._E_sigma_inv()
        for k in range(K):
            for j in range(self.n_features):
                if self.psi[j] > 0.01:
                    sum_phi = np.sum(self.phi[:, k])
                    sum_phi_x = np.sum(self.phi[:, k] * features[:, j])
                    prior_prec = 1.0 / (self.lambda_sq[k, j] * self.tau_sq[j] + 1e-10)
                    post_prec = prior_prec + self.psi[j] * E_sigma_inv[j] * sum_phi
                    self.sigma_beta[k, j] = 1.0 / (post_prec + 1e-10)
                    self.mu_beta[k, j] = self.sigma_beta[k, j] * self.psi[j] * E_sigma_inv[j] * sum_phi_x
                else:
                    self.mu_beta[k, j] = 0
                    self.sigma_beta[k, j] = 0.1
    
    def _update_sigma(self, features):
        K = self.config.n_communities
        for j in range(self.n_features):
            self.a_sigma[j] = self.config.a_sigma + self.n_nodes / 2
            ss = 0
            for i in range(self.n_nodes):
                for k in range(K):
                    res = features[i, j] - self.mu_beta[k, j]
                    ss += self.phi[i, k] * (res**2 + self.sigma_beta[k, j])
            self.b_sigma[j] = self.config.b_sigma + 0.5 * self.psi[j] * ss
    
    def _update_shrinkage(self):
        K = self.config.n_communities
        for j in range(self.n_features):
            s = sum((self.mu_beta[k, j]**2 + self.sigma_beta[k, j]) / (self.lambda_sq[k, j] + 1e-10) for k in range(K))
            self.tau_sq[j] = (s + 1e-10) / (K + 1)
        for k in range(K):
            for j in range(self.n_features):
                b2 = self.mu_beta[k, j]**2 + self.sigma_beta[k, j]
                self.lambda_sq[k, j] = (b2 + 1e-10) / (self.tau_sq[j] + 1)
    
    def _compute_elbo(self, adjacency, features):
        K = self.config.n_communities
        elbo = np.sum(self.phi * self._E_log_pi())
        E_log_theta, E_log_1_theta = self._E_log_theta()
        
        for i in range(min(self.n_nodes, 500)):
            for j in range(i + 1, min(self.n_nodes, 500)):
                for k1 in range(K):
                    for k2 in range(K):
                        p = self.phi[i, k1] * self.phi[j, k2]
                        if adjacency[i, j] > 0:
                            elbo += p * E_log_theta[k1, k2]
                        else:
                            elbo += p * E_log_1_theta[k1, k2]
        
        elbo -= np.sum(self.phi * np.log(self.phi + 1e-10))
        elbo -= np.sum(self.psi * np.log(self.psi + 1e-10) + (1 - self.psi) * np.log(1 - self.psi + 1e-10))
        return elbo
    
    def fit(self, adjacency, features, true_labels=None):
        start = time.time()
        if self.config.verbose:
            print("=" * 60)
            print(f"BayesGraphNet Training: {adjacency.shape[0]} nodes, {features.shape[1]} features")
            print("=" * 60)
        
        best_elbo, best_params = -np.inf, None
        
        for restart in range(self.config.n_restarts):
            if self.config.verbose:
                print(f"\nRestart {restart + 1}/{self.config.n_restarts}")
            
            self._initialize_params(adjacency, features)
            self.elbo_history = []
            patience_cnt = 0
            
            for it in range(self.config.max_iterations):
                batch = np.random.choice(self.n_nodes, min(self.config.batch_size, self.n_nodes), replace=False) \
                        if self.config.use_stochastic and self.n_nodes > self.config.batch_size else None
                
                self._update_phi(adjacency, features, batch)
                self._update_psi(features)
                self._update_gamma_pi()
                if it % 5 == 0:
                    self._update_theta(adjacency)
                self._update_beta(features)
                self._update_sigma(features)
                self._update_shrinkage()
                
                if it % 10 == 0:
                    elbo = self._compute_elbo(adjacency, features)
                    self.elbo_history.append(elbo)
                    
                    if self.config.verbose and it % 50 == 0:
                        n_sel = np.sum(self.psi > 0.5)
                        msg = f"  Iter {it}: ELBO={elbo:.1f}, Features={n_sel}"
                        if true_labels is not None:
                            nmi = normalized_mutual_info_score(true_labels, np.argmax(self.phi, axis=1))
                            msg += f", NMI={nmi:.4f}"
                        print(msg)
                    
                    if len(self.elbo_history) > 1:
                        change = abs(self.elbo_history[-1] - self.elbo_history[-2]) / (abs(self.elbo_history[-2]) + 1e-10)
                        if change < self.config.convergence_tol:
                            patience_cnt += 1
                        else:
                            patience_cnt = 0
                        if patience_cnt >= self.config.patience // 10:
                            if self.config.verbose:
                                print(f"  Converged at iteration {it}")
                            break
            
            final_elbo = self.elbo_history[-1] if self.elbo_history else -np.inf
            if final_elbo > best_elbo:
                best_elbo = final_elbo
                best_params = {k: getattr(self, k).copy() for k in 
                              ['phi', 'psi', 'gamma_pi', 'theta_alpha', 'theta_beta', 'mu_beta', 'sigma_beta']}
                best_params['elbo_history'] = self.elbo_history.copy()
        
        if best_params:
            for k, v in best_params.items():
                if k != 'elbo_history':
                    setattr(self, k, v)
            self.elbo_history = best_params['elbo_history']
        
        self.results = BayesGraphNetResults(
            community_assignments=np.argmax(self.phi, axis=1),
            community_probabilities=self.phi,
            feature_relevance=self.psi,
            selected_features=np.where(self.psi > 0.5)[0],
            community_proportions=self.gamma_pi / np.sum(self.gamma_pi),
            block_matrix=self.theta_alpha / (self.theta_alpha + self.theta_beta),
            feature_coefficients=self.mu_beta,
            elbo_history=self.elbo_history,
            n_iterations=len(self.elbo_history) * 10,
            convergence_achieved=patience_cnt >= self.config.patience // 10,
            training_time=time.time() - start
        )
        self.is_fitted = True
        
        if self.config.verbose:
            print(f"\nTraining complete: {self.results.training_time:.2f}s, "
                  f"Selected {len(self.results.selected_features)}/{self.n_features} features")
        return self
    
    def predict(self):
        return self.results.community_assignments if self.is_fitted else None
    
    def predict_proba(self):
        return self.results.community_probabilities if self.is_fitted else None
    
    def get_selected_features(self, threshold=0.5):
        return np.where(self.psi > threshold)[0] if self.is_fitted else None
    
    def get_feature_importance(self):
        return self.psi if self.is_fitted else None

# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_nmi(true_labels, pred_labels):
    return normalized_mutual_info_score(true_labels, pred_labels)

def compute_ari(true_labels, pred_labels):
    return adjusted_rand_score(true_labels, pred_labels)

def compute_modularity(adjacency, labels):
    n, m = adjacency.shape[0], np.sum(adjacency) / 2
    if m == 0:
        return 0
    degrees = np.sum(adjacency, axis=1)
    Q = sum(adjacency[i, j] - degrees[i] * degrees[j] / (2 * m)
            for i in range(n) for j in range(n) if labels[i] == labels[j])
    return Q / (2 * m)

def compute_feature_metrics(true_rel, selected, n_features):
    true_set, sel_set = set(true_rel), set(selected)
    TP = len(true_set & sel_set)
    FP, FN = len(sel_set - true_set), len(true_set - sel_set)
    TN = n_features - len(true_set | sel_set)
    
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / denom if denom > 0 else 0
    
    return {'precision': prec, 'recall': rec, 'f1': f1, 'mcc': mcc,
            'n_selected': len(selected), 'reduction_rate': 1 - len(selected) / n_features}

def evaluate_model(model, adjacency, features, true_labels, true_relevant=None):
    pred = model.predict()
    metrics = {
        'nmi': compute_nmi(true_labels, pred),
        'ari': compute_ari(true_labels, pred),
        'modularity': compute_modularity(adjacency, pred),
        'training_time': model.results.training_time,
        'n_iterations': model.results.n_iterations
    }
    if true_relevant is not None:
        fs = compute_feature_metrics(true_relevant, model.results.selected_features, features.shape[1])
        metrics.update({f'fs_{k}': v for k, v in fs.items()})
    return metrics

# =============================================================================
# Baseline Methods
# =============================================================================

class SpectralBaseline:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.labels_ = None
    
    def fit_predict(self, adjacency, features=None):
        combined = adjacency
        if features is not None:
            sim = features @ features.T
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-10)
            combined = 0.5 * adjacency + 0.5 * sim
        combined += np.eye(combined.shape[0]) * 0.1
        self.labels_ = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed',
                                          random_state=42).fit_predict(combined)
        return self.labels_

class LouvainBaseline:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.labels_ = None
    
    def fit_predict(self, adjacency, features=None):
        n = adjacency.shape[0]
        labels = np.arange(n)
        
        for _ in range(50):
            improved = False
            for i in np.random.permutation(n):
                neighbors = np.where(adjacency[i] > 0)[0]
                if len(neighbors) == 0:
                    continue
                neighbor_comms = np.unique(labels[neighbors])
                best_comm, best_delta = labels[i], 0
                
                for c in neighbor_comms:
                    if c != labels[i]:
                        delta = self._delta_mod(adjacency, labels, i, c)
                        if delta > best_delta:
                            best_delta, best_comm = delta, c
                
                if best_comm != labels[i]:
                    labels[i] = best_comm
                    improved = True
            
            if not improved:
                break
        
        unique = np.unique(labels)
        self.labels_ = np.array([np.where(unique == l)[0][0] for l in labels])
        return self.labels_
    
    def _delta_mod(self, adj, labels, node, new_comm):
        m = np.sum(adj) / 2
        if m == 0:
            return 0
        k_i = np.sum(adj[node])
        mask_new = labels == new_comm
        mask_old = labels == labels[node]
        return (np.sum(adj[node, mask_new]) - np.sum(adj[node, mask_old])) / m - \
               k_i * (np.sum(adj[:, mask_new]) - np.sum(adj[:, mask_old]) + k_i) / (2 * m**2)

# =============================================================================
# Experiment Pipeline
# =============================================================================

def run_experiment(dataset_name='synthetic_medium', n_communities=None, verbose=True):
    print(f"\n{'='*70}\nEXPERIMENT: {dataset_name}\n{'='*70}")
    
    adjacency, features, true_labels = load_dataset(dataset_name)
    _, _, _, true_relevant = generate_attributed_network(
        adjacency.shape[0], len(np.unique(true_labels)), features.shape[1],
        max(30, features.shape[1] // 10)
    )
    adjacency, features = preprocess_data(adjacency, features)
    
    n_nodes, n_features = adjacency.shape[0], features.shape[1]
    if n_communities is None:
        n_communities = len(np.unique(true_labels))
    
    print(f"Nodes: {n_nodes}, Edges: {int(np.sum(adjacency)/2)}, Features: {n_features}, Communities: {n_communities}")
    
    results = {'dataset': dataset_name, 'methods': {}}
    
    # BayesGraphNet
    print("\n[1] BayesGraphNet...")
    config = BayesGraphNetConfig(n_communities=n_communities, max_iterations=200, n_restarts=2, verbose=verbose)
    model = BayesGraphNet(config)
    model.fit(adjacency, features, true_labels)
    results['methods']['BayesGraphNet'] = evaluate_model(model, adjacency, features, true_labels, true_relevant)
    
    # Spectral
    print("\n[2] Spectral Clustering...")
    t0 = time.time()
    spectral = SpectralBaseline(n_communities).fit_predict(adjacency, features)
    results['methods']['Spectral'] = {
        'nmi': compute_nmi(true_labels, spectral),
        'ari': compute_ari(true_labels, spectral),
        'modularity': compute_modularity(adjacency, spectral),
        'training_time': time.time() - t0
    }
    
    # Louvain
    print("\n[3] Louvain...")
    t0 = time.time()
    louvain = LouvainBaseline(n_communities).fit_predict(adjacency)
    results['methods']['Louvain'] = {
        'nmi': compute_nmi(true_labels, louvain),
        'ari': compute_ari(true_labels, louvain),
        'modularity': compute_modularity(adjacency, louvain),
        'training_time': time.time() - t0
    }
    
    # Summary
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Method':<20} {'NMI':<10} {'ARI':<10} {'Modularity':<12} {'Time(s)':<10}")
    print("-" * 62)
    for method, m in results['methods'].items():
        print(f"{method:<20} {m['nmi']:<10.4f} {m['ari']:<10.4f} {m['modularity']:<12.4f} {m['training_time']:<10.2f}")
    
    bgn = results['methods']['BayesGraphNet']
    best_base = max(results['methods']['Spectral']['nmi'], results['methods']['Louvain']['nmi'])
    print(f"\nImprovement over best baseline: {(bgn['nmi'] - best_base) / best_base * 100:.1f}%")
    
    if 'fs_f1' in bgn:
        print(f"\nFeature Selection: F1={bgn['fs_f1']:.4f}, Reduction={bgn['fs_reduction_rate']*100:.1f}%")
    
    results['model'] = model
    return results

def run_benchmark():
    datasets = ['synthetic_small', 'synthetic_medium', 'synthetic_ppi', 'synthetic_citation']
    all_results = {}
    
    print("\n" + "=" * 80)
    print(" " * 20 + "BAYESGRAPHNET BENCHMARK")
    print("=" * 80)
    
    for ds in datasets:
        try:
            all_results[ds] = run_experiment(ds, verbose=False)
        except Exception as e:
            print(f"Error on {ds}: {e}")
    
    print("\n" + "=" * 80)
    print(" " * 25 + "FINAL RESULTS")
    print("=" * 80)
    print(f"\n{'Dataset':<25} {'BGN NMI':<15} {'Baseline NMI':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for ds, r in all_results.items():
        bgn = r['methods']['BayesGraphNet']['nmi']
        base = max(r['methods']['Spectral']['nmi'], r['methods']['Louvain']['nmi'])
        print(f"{ds:<25} {bgn:<15.4f} {base:<15.4f} {(bgn-base)/base*100:>+10.1f}%")
    
    return all_results

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 10 + "BAYESGRAPHNET: Community Detection & Feature Selection")
    print("=" * 80)
    
    # Demo
    print("\n[DEMO] Quick test on synthetic data...\n")
    adjacency, features, true_labels, true_relevant = generate_attributed_network(
        n_nodes=800, n_communities=5, n_features=250, n_relevant_features=35
    )
    adjacency, features = preprocess_data(adjacency, features)
    
    config = BayesGraphNetConfig(n_communities=5, max_iterations=150, n_restarts=2, verbose=True)
    model = BayesGraphNet(config)
    model.fit(adjacency, features, true_labels)
    
    metrics = evaluate_model(model, adjacency, features, true_labels, true_relevant)
    
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Community Detection: NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, Modularity={metrics['modularity']:.4f}")
    print(f"Feature Selection: F1={metrics['fs_f1']:.4f}, Reduction={metrics['fs_reduction_rate']*100:.1f}%")
    print(f"Training: {metrics['n_iterations']} iterations, {metrics['training_time']:.2f}s")
    
    # Full benchmark
    print("\n" + "=" * 80)
    print("Running full benchmark...")
    print("=" * 80)
    run_benchmark()
    
    print("\n" + "=" * 80)
    print(" " * 25 + "COMPLETE")
    print("=" * 80)
