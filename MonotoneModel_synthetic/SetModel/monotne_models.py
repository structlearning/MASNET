import torch
import torch.nn as nn
import torch.nn.functional as F

class Hat(nn.Module):
    def __init__(self, args):
        super().__init__()
        print(f"\nUSING NORMAL HAT\n")
        self.hat_start = args.hat_start
        self.hat_width = args.hat_width
        
    def forward(self, x):
        return F.relu(x - self.hat_start) - 2 * F.relu(x - self.hat_start - 0.5* self.hat_width) + F.relu(x - self.hat_start - self.hat_width)

class TrainableHat(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_channels = args.num_channels
        self.hat_start = nn.Parameter(torch.randn(self.num_channels, requires_grad = True)) 
        self.hat_width_param = nn.Parameter(torch.rand(self.num_channels, requires_grad = True))
        self.transition_param = nn.Parameter(torch.randn(self.num_channels, requires_grad = True))
        self.temperature = args.temp
        self.elu_alpha = args.alpha
        self.show_change = args.show_change
        print(f"\nUSING PARAMETRIC-LEARNABLE HAT, ALPHA: {self.elu_alpha}, TEMP: {self.temperature}\n")
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch size, num channels)
        Output:
            Tensor of shape (batch size, num channels)
        """
        eps = 1e-3
        transition = 0.5 * (F.tanh(self.transition_param/self.temperature) + 1) #transition pts \in (0,1) for all channels. Shape: (num channels,)
        width = F.elu(self.hat_width_param, alpha = self.elu_alpha) + self.elu_alpha + eps #width > eps for all channels. Shape: (num channels,)
        
        if self.show_change:
            print(f"\nMAX WIDTH: {torch.max(width)}, MIN WIDTH: {torch.min(width)}")
            print(f"MAX TRANS: {torch.max(transition)}, MIN TRANS: {torch.min(transition)}")
            print(f"MAX START: {torch.max(self.hat_start)}, MIN START: {torch.min(self.hat_start)}\n")
        
        x = x - self.hat_start # Shape: (batch, num channels)
        term1 = -F.relu(torch.div(x - transition * width, width * transition * (1 - transition) + eps))# Shape: (batch, num channels), broadcasting
        term2 = F.relu(torch.div(x, width * transition + eps))# Shape: (batch, num channels), broadcasting
        term3 = F.relu(torch.div(x - width, width * (1 - transition) + eps))# Shape: (batch, num channels), broadcasting
        return term1 + term2 + term3 # Shape: (batch, num channels)

def get_act(activation_key:str, args = None):
    if activation_key not in ['relu', 'identity','trainable_hat','hat','l_relu']:
        raise AssertionError()
    if activation_key == 'trainable_hat':
        return TrainableHat(args)
    elif activation_key == 'hat':
        return Hat(args)
    elif activation_key == 'relu':
        return nn.ReLU()
    elif activation_key == 'l_relu':
        negative_slope = args.lrelu_slope
        return nn.LeakyReLU(negative_slope=negative_slope)
    elif activation_key == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Activation {activation_key} not supported")

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
        activation_name = args.activation
        self.activation = get_act(activation_name, args)
        self.normalize_in_sigma = args.sig_norm
        self.mlp = nn.Sequential(nn.Linear(self.num_channels, self.num_channels),nn.ReLU(),
                                 nn.Linear(self.num_channels, 1))
        self.task_type = args.task_type
        self.agg = args.agg

    def normalize_data(self, x):
        eps = 1e-6
        batch_min = x.min(dim=0, keepdim=True)[0]
        batch_max = x.max(dim=0, keepdim=True)[0]
        
        denom = batch_max - batch_min 
        denom = torch.clamp(denom, min=eps)
        
        #Normalize the data to have each co-ordinate in (0,1)
        x = (x - batch_min) / denom
        return x
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, input_dim).
            mask (torch.Tensor): Mask of shape(batch size, num points)

        Returns:
            torch.Tensor: Output of shape (batch_size, num_channels).
        """
        # Normalize `a` to be a unit vector
        batch_size, input_dim = x.shape[0], x.shape[-1]
        input_dim = torch.tensor(input_dim)
        
        # assert len(mask.shape) == 2, f"Supposed mask shape: {torch.Size([batch_size, x.shape[1]])}, Actual mask shape: {mask.shape}"
        
        if self.normalize_in_sigma:
            x = self.normalize_data(x)
        
        eps = 1e-6
        a_unit = self.a / (torch.norm(self.a, dim=1, keepdim=True) + eps)  # Shape: (num_channels, input_dim)

        # Compute projection of `x` onto `a_unit` for each channel
        proj = torch.matmul(x, a_unit.T)  # Shape: (batch_size, num_points, num_channels), broadcasted matrix multiplication 

        # Apply the scaling and bias
        scaled = (proj - self.b) / self.c  # Broadcasting (batch_size, num_points, num_channels)
        scaled_activation = self.activation(scaled) #(batch size, num points, num channels), broadcasted multiplication

        # Apply activation function and sum over `num_points`
        if self.agg == 'max': aggregated, _ = torch.max(scaled_activation, dim = 1) 
        else: aggregated = torch.sum(scaled_activation, dim = 1)
        assert aggregated.shape == torch.Size([batch_size, self.num_channels]), f"Supposed agg shape: {torch.Size([batch_size, self.num_channels])}, Actual agg shape: {agg.shape}"
        if self.task_type == 'FacilityLocation':
            aggregated = self.mlp(aggregated).squeeze()
        return aggregated

class MonotneodeSetModel(nn.Module):
    def __init__(self, args):
        """
        Custom model processing two tensor sets.

        Args:
            args: args from task_config.yaml
            activation_key: activation fn name(eg-leaky relu)
        """
        super().__init__()
        self.mlp = MLP(args)
        self.special_layer = SigmaLayer(args)

    def forward(self, x1, x2):
        """
        Forward pass for two input tensors.

        Args:
            x1 (torch.Tensor): First set of input tensors, shape (batch, num_points, input_dim).
            x2 (torch.Tensor): Second set of input tensors, shape (batch, num_points, input_dim).

        Returns:
            torch.Tensor: Output of shape (batch_size, num_channels).
        """
        out1 = self.mlp(x1)  # Shape: (batch, num_points, output_dim)
        out2 = self.mlp(x2)  # Shape: (batch, num_points, output_dim)

        return self.special_layer(out1), self.special_layer(out2)  # Returns tuple (out1, out2)


class InvariantDeepSets(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_dim, hidden_dim, out_dim = args.d, args.hidden_dim, args.out_dim
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X,  Y):
        # x: (batch_size, set_size, in_dim)
        # mask: (batch size, set size)
        phi_x = self.phi(X)  # Apply phi function elementwise
        phi_x_masked = phi_x 
        sum_x = phi_x_masked.sum(dim=1)  # Aggregate over set dimension
        out_x = self.rho(sum_x).squeeze()  # Apply rho function
        phi_y = self.phi(Y)  # Apply phi function elementwise
        phi_y_masked = phi_y
        sum_y = phi_y_masked.sum(dim=1)  # Aggregate over set dimension
        out_y = self.rho(sum_y).squeeze()  # Apply rho function
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
        
        in_dim, hidden_dim, out_dim = args.d, args.hidden_dim, args.out_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(in_dim, hidden_dim)  # Fix mismatch
        
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

    def forward(self, X, Y):
        
        X = self.input_projection(X)
        X = self.enc(X) #applying mask before pooling, broadcasted multiplication
        X = self.pooling(X) #(batch size, 1, h_dim)
        X = self.dec(X).squeeze() #(batch size, out dim)
        
        Y = self.input_projection(Y)
        Y = self.enc(Y) 
        Y = self.pooling(Y) #(batch size, 1, hidden dim)
        Y = self.dec(Y).squeeze() #(batch size, out dim)
        
        return X, Y