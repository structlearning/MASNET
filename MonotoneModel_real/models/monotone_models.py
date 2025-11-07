import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hat_functions import *
import sys

class MLP(nn.Module):
    def __init__(self, args):
        """
        Multi-Layer Perceptron with BatchNorm.

        Args:
            args: args from task_config.yaml
        """
        super().__init__()
        layers = []
        input_dim = args.d
        hidden_dims = args.hidden_dims
        output_dim = args.d
        activation = get_act(args.mlp_act, args)
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(linear_layer.weight)  # Xavier initialization
            layers.append(linear_layer)
            layers.append(activation)
            prev_dim = hidden_dim
            
        output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(output_layer.weight)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
    
class MonotonicMLP(nn.Module):
    """
    Monotonic Multi-Layer Perceptron with non-negative weights and ReLU activation.
    
    This network ensures monotonicity by using torch.abs() on weights before
    matrix multiplication, combined with ReLU activation functions.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dims (list): List of hidden layer dimensions
        output_dim (int): Dimension of output
        bias (bool): Whether to use bias terms (default: True)
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, bias=True):
        super(MonotonicMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layer dimensions
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create linear layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                nn.Linear(layer_dims[i], layer_dims[i + 1], bias=bias)
            )
    
    def forward(self, x):
        """
        Forward pass with non-negative weights and ReLU activation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        for i, layer in enumerate(self.layers):
            # Apply absolute value to weights to ensure non-negativity
            weight = torch.abs(layer.weight)
            bias = torch.abs(layer.bias)
            # Manual matrix multiplication with non-negative weights
            x = F.linear(x, weight, bias)
            x = F.relu(x)
        
        return x

class SigmaLayer(nn.Module):
    def __init__(self, args):
        """
        Special layer computing sum_i σ(a * x_i / c + b) for num_channels.

        Args:
            args: args from task_config.yaml
        """
        super().__init__()
        self.num_channels = args.num_channels
        self.input_dim = args.d
        # Learnable parameters for each channel
        self.a = nn.Parameter(torch.rand(self.num_channels, self.input_dim, requires_grad=True))  # Learnable unit vectors
        self.c = nn.Parameter(torch.ones(self.num_channels, requires_grad=True))  # c in (0,2]
        self.b = nn.Parameter(torch.rand(self.num_channels, requires_grad=True))  # Bias in (-1, 1)
        activation_name = args.hat_act
        self.activation = get_act(activation_name, args)
        self.agg = args.agg
        self.divide_by_c = not args.no_divide_by_c
        if not self.divide_by_c: print("Divide by c set to False")
    
    def forward(self, x, mask):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, input_dim).
            mask (torch.Tensor): Mask of shape(batch size, num points)

        Returns:
            torch.Tensor: Output of shape (batch_size, num_channels).
        """
        # Normalize `a` to be a unit vector
        batch_size, num_pts, input_dim = x.shape
        input_dim = torch.tensor(input_dim)
        
        assert len(mask.shape) == 2, f"Supposed mask shape: {torch.Size([batch_size, num_pts])}, Actual mask shape: {mask.shape}"
        eps = 1e-6
        a_unit = self.a / (torch.norm(self.a, dim=1, keepdim=True) + eps)  # Shape: (num_channels, input_dim)

        # Compute projection of `x` onto `a_unit` for each channel
        try: proj = torch.matmul(x, a_unit.T)  # Shape: (batch_size, num_points, num_channels), broadcasted matrix multiplication 
        except:
            print(f"a shape: {a_unit.shape}")
            print(f"x shape: {x.shape}")
            sys.exit()
        # Apply the scaling and bias
        adjust_shape = lambda tensor : tensor.unsqueeze(0).unsqueeze(0).expand(proj.shape) #(batch size,  num pts, num channels)
        b, c = adjust_shape(self.b), adjust_shape(self.c)  
        if self.divide_by_c:
            scaled = torch.div(proj - b, c + eps)  # (batch_size, num_points, num_channels)
        else:
            scaled = proj - b
        scaled = scaled.contiguous().view(-1, self.num_channels) #(batch size * num pts, num channels)
        scaled_activation = self.activation(scaled).view(batch_size, num_pts, -1) * mask.unsqueeze(-1)  #(batch size , num points, num channels); restored
        # Apply activation function and sum over `num_points`
        if self.agg == 'max': aggregated, _ = torch.max(scaled_activation, dim = 1) 
        else: aggregated = torch.sum(scaled_activation, dim = 1)
        assert aggregated.shape == torch.Size([batch_size, self.num_channels]), f"Supposed agg shape: {torch.Size([batch_size, self.num_channels])}, Actual agg shape: {agg.shape}"
        
        return aggregated

class MonotneodeModel(nn.Module):
    def __init__(self, args):
        """
        Custom model processing two tensor sets.

        Args:
            args: args from task_config.yaml
            activation_key: activation fn name(eg-leaky relu)
        """
        super().__init__()
        self.encoder = nn.Identity()
        if args.DATASET_NAME == 'POINTCLOUD' and args.pointcloud_encoder == 'pointnet':
            self.encoder = PointNet(args)
        elif args.DATASET_NAME == 'POINTCLOUD' and args.pointcloud_encoder == 'dgcnn':
            self.encoder = DGCNNPointFeatures(args)
        self.mlp = MLP(args) if not args.use_shallow else nn.Identity()
        self.special_layer = SigmaLayer(args)
        self.outer_layer = nn.Identity()
        
        if args.monotone_m2:
            print("Using outer monotonic MLP")
            self.outer_layer = MonotonicMLP(args.num_channels, args.monotonic_m2_hidden, args.num_channels)

    def forward(self, x1, mask1, x2, mask2):
        """
        Forward pass for two input tensors.

        Args:
            x1 (torch.Tensor): First set of input tensors, shape (batch, num_points, input_dim).
            x2 (torch.Tensor): Second set of input tensors, shape (batch, num_points, input_dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, num_channels).
        """
        out1 = self.mlp(self.encoder(x1))  # Shape: (batch, num_points, output_dim)
        out1 = self.special_layer(out1, mask1)
        out2 = self.mlp(self.encoder(x2))  # Shape: (batch, num_points, output_dim)
        out2 = self.special_layer(out2, mask2)
        
        return self.outer_layer(out1), self.outer_layer(out1)


class InvariantDeepSets(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_dim, hidden_dim, out_dim = args.d, args.hidden_dim, args.num_channels
        self.use_outer = not args.no_outer_rho
        self.monotone_m2 = args.monotone_m2
        self.encoder = nn.Identity()
        if args.DATASET_NAME == 'POINTCLOUD' and args.pointcloud_encoder == 'pointnet':
            self.encoder = PointNet(args)
        elif args.DATASET_NAME == 'POINTCLOUD' and args.pointcloud_encoder == 'dgcnn':
            self.encoder = DGCNNPointFeatures(args)
        
        if not args.use_shallow: 
            print("DeepSets with M1")
            self.phi = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            print("Shallow ReLU")
            self.phi = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU()
            )
            
        if self.use_outer and self.monotone_m2:
            self.rho = MonotonicMLP(hidden_dim, [hidden_dim], out_dim)
            print("using DeepSets with outer monotonic MLP, ie MASNET ReLU with monotonic M2")
            
        elif self.use_outer:
            self.rho = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else: 
            self.rho = nn.Identity()
            print("using DeepSets with no outer, ie MASNET ReLU")
        
        self.aggregation_type = args.agg

    def forward(self, X, mask_X, Y, mask_Y):
        # x: (batch_size, set_size, in_dim)
        # mask: (batch size, set size)
        X = self.encoder(X)
        Y = self.encoder(Y)
            
        phi_x = self.phi(X)  # Apply phi function elementwise
        phi_y = self.phi(Y)  # Apply phi function elementwise
        
        if self.aggregation_type == 'max':
            phi_x_masked = phi_x * mask_X.unsqueeze(-1)
            agg_x, _ = torch.max(phi_x_masked, dim=1)  # Max aggregation
            
            phi_y_masked = phi_y * mask_Y.unsqueeze(-1)
            agg_y, _ = torch.max(phi_y_masked, dim=1)  # Max aggregation   
        elif self.aggregation_type == 'sum':
            # For sum aggregation, multiply by mask
            phi_x_masked = phi_x * mask_X.unsqueeze(-1)
            agg_x = phi_x_masked.sum(dim=1)  # Sum aggregation
            
            phi_y_masked = phi_y * mask_Y.unsqueeze(-1)
            agg_y = phi_y_masked.sum(dim=1)  # Sum aggregation  
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggregation_type}. Use 'sum' or 'max'.")
        
        out_x = self.rho(agg_x)
        out_y = self.rho(agg_y)  # Apply rho function
        return out_x, out_y
    
class IndepOneWayMonotone(nn.Module):
    """
    Class for One way monotone functions, with indep copies across multiple channels
    
    Args:
        args: args from task_config.yaml file
        activation_key: activation fn name(eg- relu)
    """
    def __init__(self, args):
        super().__init__()
        self.num_channels = args.num_channels
        self.input_dim = args.d
        self.base_mlp = MLP(args)
        init_std = torch.sqrt(torch.tensor(1/self.input_dim))
        self.indep_layer_wt = nn.Parameter(torch.randn(self.num_channels, self.input_dim, requires_grad=True)*init_std)
        self.indep_layer_bias = nn.Parameter(torch.rand(self.num_channels, requires_grad=True))
        self.inner_activation = get_act('relu')
        self.outer_activation = get_act(args.outer_activation, args)
        self.agg = args.agg
        
    def monitor(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.abs().mean()}")
        
    def forward(self, X, mask_X, Y, mask_Y):
        # x, y: (batch_size, max_set_size, in_dim)  
        # mask_x, mask_y: (batch_size, max_set_size)
        batch_size, max_set_size = X.shape[0], X.shape[1]
        
        X = self.base_mlp(X) #Shape: (batch_size, max_set_size, out_dim)
        phi_X_elementwise = self.inner_activation(torch.matmul(X, self.indep_layer_wt.T) + self.indep_layer_bias) #Shape: (batch size, max_set size, num channels)
        phi_X_elementwise = phi_X_elementwise * mask_X.unsqueeze(-1) #broadcasted multiplication with mask
        assert phi_X_elementwise.shape == torch.Size([batch_size, max_set_size, self.num_channels]), f"Supposed Size of list element: {torch.Size([batch_size, set_size, self.num_channels])}, Actual Size: {phi_X_elementwise.shape}\n"
        
        if self.agg == 'sum': 
            phi_X_per_channel = torch.sum(phi_X_elementwise, dim = 1).squeeze() #list of size num channels. Each element of list-> (batch_size, num_channels)
        elif self.agg == 'max':
            phi_X_per_channel, _ = torch.max(phi_X_elementwise, dim = 1)
            phi_X_per_channel = phi_X_per_channel.squeeze()
        assert phi_X_per_channel.shape == torch.Size([batch_size, self.num_channels]), f"Supposed Size of list element: {torch.Size([batch_size, self.num_channels])}, Actual Size: {phi_X_per_channel.shape}\n"
        
        phi_X = self.outer_activation(phi_X_per_channel)
        
        Y = self.base_mlp(Y)
        phi_Y_elementwise = self.inner_activation(torch.matmul(Y, self.indep_layer_wt.T) + self.indep_layer_bias) # (batch size, max set size, )
        phi_Y_elementwise = phi_Y_elementwise * mask_Y.unsqueeze(-1)
        if self.agg == 'sum':
            phi_Y_per_channel = torch.sum(phi_Y_elementwise, dim = 1).squeeze()
        elif self.agg == 'max':
            phi_Y_per_channel, _ = torch.max(phi_Y_elementwise, dim = 1)
            phi_Y_per_channel = phi_Y_per_channel.squeeze()
        phi_Y = self.outer_activation(phi_Y_per_channel)
        
        return phi_X, phi_Y #batch_size x num_channels
        

class MAB(nn.Module):
    """ Multihead Attention Block (MAB) """
    def __init__(self, dim_q, dim_kv, dim_out, num_heads=4):
        super().__init__()
        self.fc_q = nn.Linear(dim_q, dim_out)
        self.fc_kv = nn.Linear(dim_kv, dim_out * 2)
        self.fc_out = nn.Linear(dim_out, dim_out)
        self.attn = nn.MultiheadAttention(dim_out, num_heads, batch_first=True)
        self.ln_0 = nn.LayerNorm(dim_out)
        self.ln_1 = nn.LayerNorm(dim_out)

    def forward(self, Q, X):
        """ Q: (batch, set_size_1, dim_q), X: (batch, set_size_2, dim_kv) """
        kv = self.fc_kv(X).chunk(2, dim=-1)  # Split into K, V, each of shape (batch, size2, dim_out)
        Q = self.fc_q(Q) # (batch, size1, dim_out)
        attn_out, _ = self.attn(Q, *kv) #(batch, size1, dim_out)
        out_ = self.ln_0(attn_out + Q)
        out_ = out_ + F.relu(self.fc_out(out_))
        return self.ln_1(out_)
        

class SAB(nn.Module):
    """ Self-Attention Block (SAB) """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    """ Induced Self-Attention Block (ISAB) for SetTransformer """
    def __init__(self, dim, num_heads=4, num_inducing=4):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, dim))
        self.mab1 = MAB(dim, dim, dim, num_heads)
        self.mab2 = MAB(dim, dim, dim, num_heads)

    def forward(self, X):
        #X: (batch size, num elements, dim)
        I = self.inducing.expand(X.size(0), -1, -1)
        H = self.mab1(I, X) # shape: (batch size, num_inducing, dim)
        return self.mab2(X, H) # shape: (batch size, num elements, dim)
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)
        
    def forward(self, X):
        #X: (batch size, num elements, dim)
        S = self.S.expand(X.size(0), -1, -1) #(batch size, num seeds, dim)
        out_ = self.mab(S, X)
        return out_ #(batch size, num seeds, dim)

class SetTransformer(nn.Module):
    def __init__(self, args,num_heads=4, num_inducing=4):
        super().__init__()
        
        in_dim, hidden_dim, out_dim = args.d, args.hidden_dim, args.num_channels
        self.hidden_dim = hidden_dim
        self.encoder = nn.Identity()
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        if args.DATASET_NAME == 'POINTCLOUD' and args.pointcloud_encoder == 'pointnet':
            self.encoder = PointNet(args)
        elif args.DATASET_NAME == 'POINTCLOUD' and args.pointcloud_encoder == 'dgcnn':
            self.encoder = DGCNNPointFeatures(args)
        
        if args.encoder_type == 'isab':
            self.enc = nn.Sequential(
                ISAB(hidden_dim, num_heads, num_inducing),
                ISAB(hidden_dim, num_heads, num_inducing)
            )
        elif args.encoder_type == 'sab':
            self.enc = nn.Sequential(
                SAB(hidden_dim, num_heads),
                SAB(hidden_dim, num_heads)
            )
        else: raise Exception("only SAB and ISAB encoders supported")
        
        if args.pooling_type == 'mean':
            self.pooling = lambda X: X.mean(dim=1, keepdim=True)
        else:
            self.pooling = PMA(hidden_dim, num_heads, 1)
        
        self.dec = nn.Sequential(
            SAB(hidden_dim, num_heads),
            SAB(hidden_dim, num_heads),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X, mask_X, Y, mask_Y):
        bsize, _ = mask_X.shape

        X = self.input_projection(self.encoder(X))
        X = self.enc(X) * mask_X.unsqueeze(-1) #applying mask before pooling, broadcasted multiplication
        X = self.pooling(X) #(batch size, 1, h_dim)
        assert len(X.shape) == 3 and X.shape[1] == 1, f"Supposed shape: {torch.Size([bsize, 1, self.hidden_dim])}, Actual Shape: {X.shape}"
        X = self.dec(X).squeeze() #(batch size, out dim)
        
        Y = self.input_projection(self.encoder(Y))
        Y = self.enc(Y) * mask_Y.unsqueeze(-1)
        Y = self.pooling(Y) #(batch size, 1, hidden dim)
        Y = self.dec(Y).squeeze() #(batch size, out dim)
        
        return X, Y


def knn(x, k):
    """
    x: (B, F, N) tensor
    returns: (B, N, k) indices of k-nearest neighbors
    """
    B, F, N = x.shape
    x = x.transpose(2, 1)  # (B, N, F)
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)

    k = min(k, N)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Constructs edge features for each point by concatenating (x_j - x_i, x_i)
    
    Args:
        x: input point features, shape (B, F, N)
        k: number of neighbors
        idx: optional precomputed knn indices (B, N, k)
    
    Returns:
        edge features of shape (B, 2F, N, k)
    """
    B, F, N = x.size()
    device = x.device

    if idx is None:
        idx = knn(x, k)  # (B, N, k)

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N  # (B, 1, 1)
    idx = idx + idx_base  # global indexing
    idx = idx.view(-1)  # flatten

    x = x.transpose(2, 1).contiguous()  # (B, N, F)
    feature = x.reshape(B * N, F)[idx, :]  # (B*N*k, F)
    feature = feature.view(B, N, k, F)  # (B, N, k, F)

    x = x.view(B, N, 1, F).repeat(1, 1, k, 1)  # (B, N, k, F)
    edge_feature = torch.cat((feature - x, x), dim=3)  # (B, N, k, 2F)

    return edge_feature.permute(0, 3, 1, 2).contiguous()  # (B, 2F, N, k)


class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        self.args = args
        emb_dims = args.d
        num_layers = max(1, min(args.pointnet_num_layers, 5))  # 1–5 layers allowed

        layer_dims = [64, 64, 64, 128, emb_dims][:num_layers]
        in_channels = [3] + layer_dims[:-1]

        self.convs = nn.ModuleList()
        for in_c, out_c in zip(in_channels, layer_dims):
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU()
            ))

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 3, N)
        for conv in self.convs:
            x = conv(x)
        return x.transpose(1, 2)  # (B, N, emb_dims)


class DGCNNPointFeatures(nn.Module):
    def __init__(self, args):
        super(DGCNNPointFeatures, self).__init__()
        self.args = args
        self.k = args.dgcnn_k  
        self.emb_dims = args.d
        self.num_layers = 4

        # Define output dimensions for each layer
        self.out_dims = [64, 64, 128, 256]
        self.out_dims = self.out_dims[:self.num_layers]
        
        self.edge_convs = nn.ModuleList()
        last_dim = 6  # initial feature dimension (3+3)
        for out_dim in self.out_dims:
            self.edge_convs.append(nn.Sequential(
                nn.Conv2d(last_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            ))
            last_dim = out_dim * 2  # after get_graph_feature (concat of x_i and x_j - x_i)
        
        total_cat_dim = sum(self.out_dims)
        self.conv_final = nn.Sequential(
            nn.Conv1d(total_cat_dim, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 3, N)
        feature_list = []
        for conv in self.edge_convs:
            x_feat = get_graph_feature(x, k=self.k)     # (B, C, N, k)
            x_feat = conv(x_feat)                        # (B, out_dim, N, k)
            x_feat = x_feat.max(dim=-1)[0]              # (B, out_dim, N)
            feature_list.append(x_feat)
            x = x_feat                                   # update for next layer

        x = torch.cat(feature_list, dim=1)               # (B, total_cat_dim, N)
        x = self.conv_final(x)                           # (B, emb_dims, N)
        return x.transpose(2, 1)                         # (B, N, emb_dims)
    
class SetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):  # x: (B, n, d)
        return self.phi(x)  # (B, n, h)

class SoftSelector(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B, n, h)
        logits = self.scorer(x).squeeze(-1)  # (B, n)
        weights = torch.sigmoid(logits)  # soft inclusion probabilities
        return weights  # (B, n)

class NeuralSetFunction(nn.Module):
    """
    Simple dummy neural SFE extension: approximates f(S) = ||sum(x_i)|| over sampled soft subsets
    """
    def __init__(self, hidden_dim, n_samples=8):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x, weights):  # x: (B, n, h), weights: (B, n)
        B, n, h = x.shape
        f_vals = []
        for _ in range(self.n_samples):
            bernoulli_mask = torch.bernoulli(weights).unsqueeze(-1)  # (B, n, 1)
            selected = x * bernoulli_mask  # masked
            summed = selected.sum(dim=1)   # (B, h)
            f_val = torch.norm(summed, dim=-1)  # (B,)
            f_vals.append(f_val)
        f_vals = torch.stack(f_vals, dim=0)  # (n_samples, B)
        return f_vals.mean(dim=0)  # (B,)

class NeuralSFE(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.d
        hidden_dim = args.hidden_dim
        output_dim = args.num_channels
        self.encoder = SetEncoder(input_dim, hidden_dim)
        self.selector = SoftSelector(hidden_dim)
        self.extender = NeuralSetFunction(hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)  # Set-to-vector embedding

    def forward(self, x, mask_x, y, mask_y):  # x: (B, n, d)
        x_enc = self.encoder(x)  # (B, n, h)
        weights_x = self.selector(x_enc)  # (B, n)
        pooled_x = (x_enc * weights_x.unsqueeze(-1) * mask_x.unsqueeze(-1)).sum(dim=1)  # weighted sum: (B, h)
        z_set_x = self.readout(pooled_x)  # (B, output_dim)
        y_enc = self.encoder(y)  # (B, n, h)
        weights_y = self.selector(y_enc)  # (B, n)
        pooled_y = (y_enc * weights_y.unsqueeze(-1) * mask_y.unsqueeze(-1)).sum(dim=1)  # weighted sum: (B, h)
        z_set_y = self.readout(pooled_y)  # (B, output_dim)
        return z_set_x, z_set_y  # vector embedding + extension value