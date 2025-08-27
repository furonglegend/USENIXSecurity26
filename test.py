"""
SecFedGate: Complete Implementation of Resource-Aware Gating Architecture
Implements all core components from the paper:
1. Ontology-guided feature weighting (Sec 4.1)
2. Clinical dependency encoding with graph attention (Sec 4.4)
3. Prototype-consistent representation learning (Sec 4.3)
4. Reinforcement learning-based dynamic gating (Sec 4.7)
5. Formal differential privacy mechanism (Sec 4.8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from scipy.stats import norm

class OntologyGuidedFeatureWeighting(nn.Module):
    """
    Implements ontology-driven feature prioritization (Sec 4.1)
    Equation references from paper: Feature scaling (Eq 5)
    """
    def __init__(self, input_dim, ontology_dim, hidden_dim=128):
        super().__init__()
        # Ontology projection network
        self.ontology_projector = nn.Sequential(
            nn.Linear(ontology_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Feature gating mechanism
        self.gate_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Clinical relevance estimator
        self.relevance_estimator = nn.Sequential(
            nn.Linear(ontology_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features, ontology_emb):
        """
        Args:
            features: Input features [batch_size, input_dim]
            ontology_emb: Ontology embeddings [batch_size, ontology_dim]
        
        Returns:
            Weighted features [batch_size, input_dim]
        """
        # Ontology-based relevance projection (Eq 3)
        relevance = self.ontology_projector(ontology_emb)
        
        # Clinical relevance weighting (Eq 4)
        clinical_relevance = self.relevance_estimator(ontology_emb)
        
        # Feature gating mechanism (Eq 5)
        gate = self.gate_generator(features)
        
        # Apply weighted gating with clinical relevance
        weighted_features = features * gate * clinical_relevance + features * relevance
        return weighted_features

class ClinicalDependencyEncoder(nn.Module):
    """
    Clinical Dependency Encoding with multi-head graph attention (Sec 4.4)
    Implements Eqs 17-20 with topology-aware dropout
    """
    def __init__(self, embed_dim, num_heads=4, dropout_prob=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q,K,V
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        
        # Attention parameters
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Topology-aware dropout parameters
        self.dropout_beta = 0.35
        self.dropout_min = 0.1
        self.dropout_max = 0.5

    def forward(self, embeddings, adj_matrix):
        """
        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Refined embeddings [num_nodes, embed_dim]
        """
        # Project to query, key, value
        Q = self.Wq(embeddings)
        K = self.Wk(embeddings)
        V = self.Wv(embeddings)
        
        # Split into multiple heads
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores (Eq 18)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply adjacency mask
        adj_mask = adj_matrix.unsqueeze(1)  # Add head dimension
        attn_scores = attn_scores.masked_fill(adj_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply topology-aware dropout (Eq 19-20)
        if self.training:
            # Compute adaptive dropout probabilities
            deg = adj_matrix.sum(dim=1).float()
            deg_std = deg.std(dim=1, keepdim=True)
            p_i = torch.clamp(
                self.dropout_beta * deg_std / (deg.mean(dim=1, keepdim=True) + 1e-8),
                self.dropout_min, self.dropout_max
            )
            
            # Apply dropout
            dropout_mask = torch.rand_like(attn_weights) > p_i.unsqueeze(1)
            attn_weights = attn_weights * dropout_mask.float() / (1 - p_i.unsqueeze(1))
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, self.embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.norm(embeddings + attn_output)
        return attn_output

class PrototypeConsistentLearning(nn.Module):
    """
    Prototype-Consistent Representation Learning (Sec 4.3)
    Implements Eqs 19-20 with momentum-based prototype updates
    """
    def __init__(self, embed_dim, num_prototypes, temperature=0.5, momentum=0.9):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        self.temperature = temperature
        self.momentum = momentum

    def forward(self, embeddings, cluster_ids):
        """
        Args:
            embeddings: Sample embeddings [batch_size, embed_dim]
            cluster_ids: Cluster assignments [batch_size]
        
        Returns:
            Loss value and updated prototypes
        """
        # Normalize embeddings and prototypes
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # Calculate similarity matrix (Eq 19)
        sim_matrix = torch.matmul(emb_norm, proto_norm.t()) / self.temperature
        
        # Compute contrastive loss (Eq 20)
        pos_sim = sim_matrix.gather(1, cluster_ids.unsqueeze(1))
        neg_mask = torch.ones_like(sim_matrix).scatter_(1, cluster_ids.unsqueeze(1), 0)
        neg_sim = sim_matrix.masked_fill(neg_mask == 0, float('-inf'))
        
        # Log-sum-exp for stability
        max_val = neg_sim.max(dim=1, keepdim=True).values
        log_sum_exp = torch.log(torch.exp(neg_sim - max_val).sum(dim=1, keepdim=True)) + max_val
        loss = -pos_sim + log_sum_exp
        
        # Momentum update of prototypes
        with torch.no_grad():
            for i in range(len(cluster_ids)):
                cid = cluster_ids[i]
                self.prototypes.data[cid] = (
                    self.momentum * self.prototypes.data[cid] + 
                    (1 - self.momentum) * embeddings[i]
                )
                
        return loss.mean(), self.prototypes

class RLGatingController(nn.Module):
    """
    Reinforcement Learning-based Dynamic Gating (Sec 4.7)
    Implements Eqs 30-33 for resource-aware module activation
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Args:
            state: Device state vector [batch_size, state_dim]
        
        Returns:
            Action probabilities [batch_size, action_dim]
            State value estimate [batch_size, 1]
        """
        action_probs = self.policy_net(state)
        state_value = self.value_net(state)
        return action_probs, state_value

class DifferentialPrivacyMechanism:
    """
    Formal Privacy Guarantee Mechanism (Sec 4.8)
    Implements Eqs 35-38 with calibrated noise injection
    """
    def __init__(self, epsilon, delta, sensitivity=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
    def add_noise(self, tensor):
        """Add calibrated Gaussian noise (Eq 36)"""
        noise = torch.normal(
            mean=0.0, 
            std=self.sigma * self.sensitivity,
            size=tensor.shape
        ).to(tensor.device)
        return tensor + noise
        
    def clip_gradient(self, gradient, clip_norm):
        """Clip gradients to control sensitivity (Eq 35)"""
        norm = torch.norm(gradient)
        if norm > clip_norm:
            gradient = gradient * clip_norm / norm
        return gradient

class SecFedGateClient(nn.Module):
    """
    Complete Client Module (Sec 4.1-4.7)
    Integrates all components for local training
    """
    def __init__(self, input_dim, ontology_dim, num_prototypes, state_dim, action_dim):
        super().__init__()
        self.cafw = OntologyGuidedFeatureWeighting(input_dim, ontology_dim)
        self.cde = ClinicalDependencyEncoder(input_dim)
        self.pcrl = PrototypeConsistentLearning(input_dim, num_prototypes)
        self.rl_gate = RLGatingController(state_dim, action_dim)
        
        # Local model components
        self.local_model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # State tracking
        self.device_state = None

    def forward(self, x, ontology_emb, adj_matrix):
        """
        Args:
            x: Input features [batch_size, input_dim]
            ontology_emb: Ontology embeddings [batch_size, ontology_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Output predictions [batch_size, 64]
        """
        # Resource-aware gating (Sec 4.7)
        if self.device_state is not None:
            action_probs, _ = self.rl_gate(self.device_state)
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = 0  # Default to full processing
            
        # Feature processing pipeline
        if action == 0:  # Full processing
            x = self.cafw(x, ontology_emb)
            x = self.cde(x, adj_matrix)
        elif action == 1:  # Partial processing
            x = self.cafw(x, ontology_emb)
        # Else: Minimal processing (identity)
            
        # Local model computation
        output = self.local_model(x)
        return output

    def update_state(self, new_state):
        """Update device state vector"""
        self.device_state = new_state

class SecFedGateServer:
    """
    Server Module with Privacy-Preserving Aggregation (Sec 4.8)
    Implements Algorithm 2 from the paper
    """
    def __init__(self, global_model, dp_mechanism):
        self.global_model = global_model
        self.dp = dp_mechanism
        self.client_updates = {}

    def aggregate_updates(self):
        """Federated aggregation with differential privacy (Eq 36)"""
        global_state = self.global_model.state_dict()
        new_global_state = copy.deepcopy(global_state)
        
        # Weighted averaging of client updates
        total_samples = sum(client_data['num_samples'] for client_data in self.client_updates.values())
        
        for param_name in global_state.keys():
            param_sum = torch.zeros_like(global_state[param_name])
            
            for client_id, client_data in self.client_updates.items():
                client_weight = client_data['num_samples'] / total_samples
                client_update = client_data['model_update'][param_name]
                
                # Apply privacy-preserving modifications
                client_update = self.dp.clip_gradient(client_update, clip_norm=1.0)
                client_update = self.dp.add_noise(client_update)
                
                param_sum += client_weight * client_update
                
            new_global_state[param_name] = global_state[param_name] + param_sum
            
        # Update global model
        self.global_model.load_state_dict(new_global_state)
        return self.global_model

    def register_client_update(self, client_id, model_update, num_samples):
        """Store client updates for aggregation"""
        self.client_updates[client_id] = {
            'model_update': model_update,
            'num_samples': num_samples
        }

# ----------------------- Clinical Implementation Utilities -----------------------

def clinical_stratified_sampling(clients, clinical_groups):
    """
    Stratified client sampling based on clinical groups (Sec 4.8)
    Implements clinical stratification for participant selection
    """
    group_mapping = {}
    for client_id, group_id in clinical_groups.items():
        group_mapping.setdefault(group_id, []).append(client_id)
    
    selected = []
    for group_id, group_clients in group_mapping.items():
        n_select = max(1, int(len(group_clients) * 0.3))  # Select 30% per group
        selected.extend(np.random.choice(group_clients, n_select, replace=False))
    
    return selected

def initialize_prototypes(num_prototypes=100, embed_dim=512):
    """Initialize diagnostic prototypes (Sec 4.3)"""
    return torch.randn(num_prototypes, embed_dim)

def update_device_state(state, power_consumption, latency):
    """Update device state vector (Sec 4.7)"""
    # Update based on resource consumption metrics
    state[0] = min(1.0, state[0] + power_consumption * 0.01)  # CPU util
    state[1] = max(0.0, state[1] - power_consumption * 0.005)  # Battery
    state[3] = latency  # Update latency
    return state

# ----------------------- Federated Training Workflow -----------------------

def federated_training(server, clients, data_loaders, ontology_graph, clinical_groups, num_rounds=100):
    """
    Complete federated training loop (Sec 4.8)
    Implements Algorithm 2 with clinical stratification
    """
    # Initialize prototypes
    prototypes = initialize_prototypes()
    
    for round in range(num_rounds):
        # Clinical stratified sampling (Sec 4.8)
        selected_ids = clinical_stratified_sampling(clients.keys(), clinical_groups)
        
        # Client local training
        for client_id in selected_ids:
            client = clients[client_id]
            data_loader = data_loaders[client_id]
            
            # Simulate device state (CPU util, battery, memory, latency, capability)
            device_state = torch.tensor([0.5, 80.0, 1024, 50.0, 3.0]) 
            client.update_state(device_state)
            
            # Local training
            client.train()
            optimizer = torch.optim.SGD(client.parameters(), lr=0.01, momentum=0.9)
            
            for batch_idx, (data, target, ontology_emb, adj_matrix) in enumerate(data_loader):
                # Forward pass
                output = client(data, ontology_emb, adj_matrix)
                
                # Prototype assignment
                with torch.no_grad():
                    cluster_ids = torch.argmin(
                        torch.cdist(output, prototypes), 
                        dim=1
                    )
                
                # Compute prototype-consistent loss
                loss, prototypes = client.pcrl(output, cluster_ids)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update device state
                device_state = update_device_state(device_state, 0.5, 85.0)
                client.update_state(device_state)
            
            # Compute model update
            model_update = {
                name: param.grad.clone().detach()
                for name, param in client.named_parameters()
            }
            server.register_client_update(client_id, model_update, len(data_loader.dataset))
        
        # Global aggregation
        global_model = server.aggregate_updates()
        
        # Distribute updated model
        for client in clients.values():
            client.load_state_dict(global_model.state_dict())
            
    return global_model

# ----------------------- Example Initialization -----------------------

if __name__ == "__main__":
    # Initialize privacy mechanism (ε=1.0, δ=10e-5)
    dp_mech = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5)
    
    # Initialize global model
    global_model = SecFedGateClient(
        input_dim=512, 
        ontology_dim=256, 
        num_prototypes=100,
        state_dim=5,
        action_dim=3
    )
    
    # Initialize server
    server = SecFedGateServer(global_model, dp_mech)
    
    # Initialize clients (10 clients)
    clients = {
        i: SecFedGateClient(512, 256, 100, 5, 3)
        for i in range(10)
    }
    
    # Simulated data loaders (in real implementation, load from medical datasets)
    data_loaders = {
        i: MedicalDataLoader(f"client_{i}_data.h5") 
        for i in range(10)
    }
    
    # Clinical group assignments (e.g., ICU, ward, outpatient)
    clinical_groups = {i: np.random.randint(0, 3) for i in range(10)}
    
    # Run federated training
    trained_model = federated_training(
        server,
        clients,
        data_loaders,
        ontology_graph="medical_ontology.owl",
        clinical_groups=clinical_groups,
        num_rounds=100
    )
