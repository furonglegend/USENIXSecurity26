"""
Implementation of SecFedGate: Resource-Aware Gating Architecture for Federated Learning
Based on the paper: 
    "SecFedGate: Resource-Aware Gating Architecture for Adversary-Resilient Federated Learning with Formal Privacy Guarantees"
Adapted from reference TCR predictor framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

class ContextAwareFeatureWeighting(nn.Module):
    """
    Ontology-guided feature weighting module
    Prioritizes clinically significant patterns using medical knowledge graphs
    Implements Eq. (14) from the paper
    """
    def __init__(self, input_dim, ontology_dim, hidden_dim=128):
        super().__init__()
        self.ontology_projector = nn.Sequential(
            nn.Linear(ontology_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.gate_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features, ontology_emb):
        # Ontology-based relevance projection
        relevance = self.ontology_projector(ontology_emb)
        
        # Feature gating mechanism
        gate = self.gate_generator(features)
        
        # Apply weighted gating
        weighted_features = features * gate + features * relevance
        return weighted_features

class ClinicalDependencyEncoder(nn.Module):
    """
    Clinical Dependency Encoder with multi-head graph attention
    Models semantic relationships among healthcare entities
    Implements Eq. (17-18) from the paper
    """
    def __init__(self, embed_dim, num_heads=4, dropout_prob=0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_prob, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Topology-aware dropout parameters
        self.dropout_prob = dropout_prob
        self.dropout_beta = 0.35
        self.dropout_min = 0.1
        self.dropout_max = 0.5

    def forward(self, embeddings, adj_matrix):
        # Compute attention coefficients
        attn_output, _ = self.multihead_attn(
            embeddings, embeddings, embeddings, 
            key_padding_mask=(adj_matrix == 0)
        )
        
        # Apply residual connection and layer normalization
        embeddings = self.norm(embeddings + self.dropout(attn_output))
        
        # Dynamic topology-aware dropout
        if self.training:
            # Compute adaptive dropout probabilities
            deg = adj_matrix.sum(dim=1).float()
            deg_std = deg.std(dim=1, keepdim=True)
            p_i = torch.clamp(
                self.dropout_beta * deg_std / (deg.mean(dim=1, keepdim=True) + 1e-8),
                self.dropout_min, self.dropout_max
            )
            
            # Apply dropout
            dropout_mask = torch.rand_like(embeddings) > p_i
            embeddings = embeddings * dropout_mask.float() / (1 - p_i)
            
        return embeddings

class PrototypeConsistentLearning(nn.Module):
    """
    Prototype-Consistent Representation Learning
    Maintains diagnostic consistency across institutions
    Implements Eq. (19-20) from the paper
    """
    def __init__(self, embed_dim, num_prototypes, temperature=0.5, momentum=0.9):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        self.temperature = temperature
        self.momentum = momentum
        
    def forward(self, embeddings, cluster_ids):
        # Normalize embeddings and prototypes
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # Calculate similarity matrix
        sim_matrix = torch.matmul(emb_norm, proto_norm.t()) / self.temperature
        
        # Contrastive loss calculation
        pos_sim = sim_matrix.gather(1, cluster_ids.unsqueeze(1))
        neg_sim = sim_matrix.clone().scatter_(1, cluster_ids.unsqueeze(1), float('-inf'))
        
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
                
        return loss.mean()

class RLGatingController(nn.Module):
    """
    Reinforcement Learning-based Gating Mechanism
    Dynamically allocates computational resources based on constraints
    Implements Eq. (30-33) from the paper
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
        action_probs = self.policy_net(state)
        state_value = self.value_net(state)
        return action_probs, state_value

class DifferentialPrivacyMechanism:
    """
    Formal Privacy Guarantee Mechanism
    Implements calibrated noise injection for (ε,δ)-DP
    Implements Eq. (35-38) from the paper
    """
    def __init__(self, epsilon, delta, sensitivity=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
    def add_noise(self, tensor):
        noise = torch.normal(
            mean=0.0, 
            std=self.sigma * self.sensitivity,
            size=tensor.shape
        ).to(tensor.device)
        return tensor + noise
        
    def clip_gradient(self, gradient, clip_norm):
        """Clip gradients to control sensitivity"""
        norm = torch.norm(gradient)
        if norm > clip_norm:
            gradient = gradient * clip_norm / norm
        return gradient

class SecFedGateClient(nn.Module):
    """
    SecFedGate Client Module
    Combines all components for local training on edge devices
    """
    def __init__(self, input_dim, ontology_dim, num_prototypes, state_dim, action_dim):
        super().__init__()
        self.cafw = ContextAwareFeatureWeighting(input_dim, ontology_dim)
        self.cde = ClinicalDependencyEncoder(input_dim)
        self.pcrl = PrototypeConsistentLearning(input_dim, num_prototypes)
        self.rl_gate = RLGatingController(state_dim, action_dim)
        
        # Client-specific parameters
        self.local_model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, x, ontology_emb, adj_matrix, state):
        # Resource-aware gating
        action_probs, _ = self.rl_gate(state)
        action = torch.multinomial(action_probs, 1).item()
        
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

class SecFedGateServer:
    """
    SecFedGate Server Implementation
    Handles federated aggregation with privacy guarantees
    Implements Algorithm 2 from the paper
    """
    def __init__(self, global_model, dp_mechanism):
        self.global_model = global_model
        self.dp_mechanism = dp_mechanism
        self.client_models = {}
        
    def aggregate_updates(self):
        """Federated aggregation with differential privacy"""
        global_state = self.global_model.state_dict()
        new_global_state = copy.deepcopy(global_state)
        
        # Weighted averaging of client updates
        total_samples = sum(client_data['num_samples'] for client_data in self.client_models.values())
        
        for param_name in global_state.keys():
            param_sum = torch.zeros_like(global_state[param_name])
            
            for client_id, client_data in self.client_models.items():
                client_weight = client_data['num_samples'] / total_samples
                client_update = client_data['model_update'][param_name]
                
                # Apply privacy-preserving modifications
                client_update = self.dp_mechanism.clip_gradient(client_update, clip_norm=1.0)
                client_update = self.dp_mechanism.add_noise(client_update)
                
                param_sum += client_weight * client_update
                
            new_global_state[param_name] = global_state[param_name] + param_sum
            
        # Update global model
        self.global_model.load_state_dict(new_global_state)
        return self.global_model

    def register_client_update(self, client_id, model_update, num_samples):
        """Store client updates for aggregation"""
        self.client_models[client_id] = {
            'model_update': model_update,
            'num_samples': num_samples
        }

# ----------------------- Training Loop Implementation -----------------------

def federated_training(server, clients, data_loaders, ontology_graph, num_rounds=100):
    """
    Federated training loop implementing the SecFedGate framework
    Combines all components for end-to-end training
    """
    for round in range(num_rounds):
        logging.info(f"Federated Round {round+1}/{num_rounds}")
        
        # Select clients for this round (stratified sampling)
        selected_clients = select_clients(clients, strategy='clinical_stratification')
        
        # Client local training
        client_updates = []
        for client_id in selected_clients:
            client = clients[client_id]
            data_loader = data_loaders[client_id]
            
            # Get current resource constraints
            device_state = get_device_state(client_id)
            
            # Local training
            model_update = client_local_training(
                client, 
                data_loader,
                ontology_graph,
                device_state
            )
            
            # Store update for aggregation
            client_updates.append((client_id, model_update))
            
            # Register update with server
            server.register_client_update(
                client_id,
                model_update,
                num_samples=len(data_loader.dataset)
            )
        
        # Global aggregation
        global_model = server.aggregate_updates()
        
        # Distribute updated model to clients
        for client in clients.values():
            client.load_state_dict(global_model.state_dict())
            
        # RL policy update
        update_rl_policies(clients, client_updates)

    return global_model

def client_local_training(client, data_loader, ontology_graph, device_state):
    """
    Client-side training with resource-aware gating
    Implements local optimization with prototype refinement
    """
    client.train()
    optimizer = optim.SGD(client.parameters(), lr=0.01, momentum=0.9)
    prototypes = initialize_prototypes()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        # Get ontology embeddings for batch
        ontology_emb = get_ontology_embeddings(data, ontology_graph)
        
        # Get adjacency matrix for clinical dependencies
        adj_matrix = get_adjacency_matrix(data, ontology_graph)
        
        # Resource-aware forward pass
        output = client(data, ontology_emb, adj_matrix, device_state)
        
        # Calculate loss with prototype consistency
        cluster_ids = assign_to_prototypes(output, prototypes)
        loss = client.pcrl(output, cluster_ids)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update prototypes
        update_prototypes(prototypes, output, cluster_ids)
        
        # Update resource state
        update_device_state(device_state)
    
    # Calculate model update (Δθ)
    model_update = {
        name: param.grad.clone().detach()
        for name, param in client.named_parameters()
    }
    
    return model_update

# ----------------------- Utility Functions -----------------------

def initialize_prototypes(num_prototypes=100, embed_dim=512):
    """Initialize diagnostic prototypes"""
    return torch.randn(num_prototypes, embed_dim)

def update_prototypes(prototypes, embeddings, cluster_ids, momentum=0.9):
    """Update prototypes using momentum-based refinement"""
    with torch.no_grad():
        for i, cid in enumerate(cluster_ids):
            prototypes[cid] = (
                momentum * prototypes[cid] +
                (1 - momentum) * embeddings[i]
            )
    return prototypes

def get_device_state(device_id):
    """Simulate device resource monitoring"""
    return {
        'cpu_util': random.uniform(0.1, 0.9),
        'memory_avail': random.uniform(0.2, 0.8),
        'battery_level': random.uniform(0.1, 1.0),
        'network_latency': random.uniform(5, 100),
        'compute_capability': random.choice([1, 2, 3])
    }

def update_device_state(state):
    """Simulate device state changes during computation"""
    state['cpu_util'] += random.uniform(-0.1, 0.1)
    state['battery_level'] -= random.uniform(0.01, 0.05)
    state['battery_level'] = max(0.05, state['battery_level'])
    return state

def update_rl_policies(clients, client_updates):
    """Update RL gating policies based on performance metrics"""
    # This would implement the policy gradient update from Eq. (33)
    # In practice, we'd use advantage estimation and backpropagation
    pass

# ----------------------- Example Usage -----------------------

if __name__ == "__main__":
    # Initialize privacy mechanism
    dp_mech = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5)
    
    # Initialize global model
    global_model = SecFedGateClient(
        input_dim=512, 
        ontology_dim=256, 
        num_prototypes=100,
        state_dim=5,  # CPU, memory, battery, latency, capability
        action_dim=3   # Full, partial, minimal processing
    )
    
    # Initialize server
    server = SecFedGateServer(global_model, dp_mech)
    
    # Initialize clients
    clients = {
        i: SecFedGateClient(
            input_dim=512, 
            ontology_dim=256, 
            num_prototypes=100,
            state_dim=5,
            action_dim=3
        )
        for i in range(10)  # 10 clients
    }
    
    # Load medical data (simulated)
    data_loaders = {i: load_medical_data(f"client_{i}_data.h5") for i in range(10)}
    
    # Load medical ontology graph
    ontology_graph = load_ontology("medical_ontology.owl")
    
    # Run federated training
    trained_model = federated_training(
        server,
        clients,
        data_loaders,
        ontology_graph,
        num_rounds=100
    )
