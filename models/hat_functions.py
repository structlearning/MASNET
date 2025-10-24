import torch
import torch.nn as nn
import torch.nn.functional as F
from models.parallel_integrals import *

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

def differentiable_safe_div(numerator, denominator, eps=1e-10):
    # Add a small offset to denominator based on its sign
    # This preserves sign while ensuring no values are close to zero
    sign = torch.sign(denominator)
    # Handle zero case (where sign would be 0)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    
    # Create a safe denominator that's never too close to zero
    safe_denom = denominator + sign * eps
    
    # Perform division with the safe denominator
    result = numerator / safe_denom
    
    # Use clamp for stability in outputs
    return torch.clamp(result, min=-1e5, max=1e5)

class Hat(nn.Module):
    def __init__(self, args):
        super().__init__()
        print(f"\nUSING NORMAL HAT\n")
        #self.hat_start = args.hat_start
        #self.hat_width = args.hat_width
        self.hat_start = 0
        self.hat_width = 1
        
    def forward(self, x):
        return F.relu(x - self.hat_start) - 2 * F.relu(x - self.hat_start - 0.5* self.hat_width) + F.relu(x - self.hat_start - self.hat_width)

class TrainableHat(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_channels = args.num_channels
        self.hat_start = nn.Parameter(torch.randn(self.num_channels, requires_grad = True)) 
        self.hat_width_param = nn.Parameter(torch.rand(self.num_channels, requires_grad = True))
        self.transition_param = nn.Parameter(torch.randn(self.num_channels, requires_grad = True))
        self.scale_param = nn.Parameter(torch.randn(self.num_channels, requires_grad = True))
        self.temp = args.temp
        self.elu_alpha = args.alpha
        self.show_change = False
        print(f"\nUSING PARAMETRIC-LEARNABLE HAT, ALPHA: {self.elu_alpha}, TEMP: {self.temp}\n")
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch size, num channels)
        Output:
            Tensor of shape (batch size, num channels)
        """
        eps = 1e-6
        adjust_shape = lambda tensor: tensor.unsqueeze(0) #(1, num channels)
        
        transition = adjust_shape(F.sigmoid(self.transition_param * self.temp)) #transition pts \in (0,1) for all channels. Shape: (1, num channels)
        width = adjust_shape(F.elu(self.hat_width_param, alpha = self.elu_alpha) + self.elu_alpha) #width > 0 for all channels. Shape: (1, num channels)
        #width = adjust_shape(torch.abs(self.hat_width_param))
        #width = adjust_shape(torch.abs(self.hat_width_param))
        #width = adjust_shape(self.hat_width_param * self.hat_width_param)
        scale = adjust_shape(torch.abs(self.scale_param))
        #scale=1.0
        #scale = adjust_shape(F.elu(self.scale_param, alpha = self.elu_alpha) + self.elu_alpha)
        #scale = adjust_shape(self.transition_param * self.transition_param)
        
        if self.show_change:
            print(f"\nMAX WIDTH: {torch.max(width)}, MIN WIDTH: {torch.min(width)}")
            print(f"MAX TRANS: {torch.max(transition)}, MIN TRANS: {torch.min(transition)}")
            print(f"MAX START: {torch.max(self.hat_start)}, MIN START: {torch.min(self.hat_start)}\n")
        
        x = x - self.hat_start # Shape: (batch, num channels)
        term1 = -F.relu(x - transition * width)/(transition * (1 - transition) + eps)# Shape: (batch, num channels), broadcasting
        term2 = F.relu(x)/(transition + eps)# Shape: (batch, num channels), broadcasting
        term3 = F.relu(x-width)/((1 - transition) + eps)# Shape: (batch, num channels), broadcasting
        result = (term1 + term2 + term3) * scale # Shape: (batch, num channels)
        #assert torch.all(result >= 0), f"Negative values found: min={result.min().item()}"
        return result
    
class MultiIntegrandNN(nn.Module):
    def __init__(self, num_channels: int = 10, input_dim: int = 1, hidden_dim: int = 5, num_hidden_layers: int = 1, positive: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        weight_in = nn.Parameter(torch.Tensor(num_channels, input_dim, hidden_dim))
        bias_in = nn.Parameter(torch.Tensor(num_channels, hidden_dim))
        nn.init.xavier_uniform_(weight_in)
        nn.init.zeros_(bias_in)
        self.weights.append(weight_in)
        self.biases.append(bias_in)
        
        for layer in range(num_hidden_layers-1):
            weight = nn.Parameter(torch.Tensor(num_channels, hidden_dim, hidden_dim))
            bias = nn.Parameter(torch.Tensor(num_channels, hidden_dim))
            nn.init.xavier_uniform_(weight)
            nn.init.zeros_(bias)
            self.weights.append(weight)
            self.biases.append(bias)
            
        weight_out = nn.Parameter(torch.Tensor(num_channels, hidden_dim, input_dim))
        bias_out = nn.Parameter(torch.Tensor(num_channels, input_dim))
        nn.init.xavier_uniform_(weight_out)
        nn.init.zeros_(bias_out)
        self.weights.append(weight_out)
        self.biases.append(bias_out)
        
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Sigmoid() if positive else nn.Identity()
        self.positive = positive
        
    def forward(self, x):
        #x shape: (batch size, num channels, input dim)
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum('bci,cio->bco', x, weight)
            x = x + bias.unsqueeze(0)
            
            if idx < len(self.weights) - 1: x = self.hidden_activation(x)
            else: x = self.output_activation(x)
        
        return x #shape: (batch size, num channels, input dim)
    
class MultiIntegrandNN_new(nn.Module):
    def __init__(self, num_channels: int = 10, hidden_dim: int = 5, num_hidden_layers: int = 1, positive: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        weight_in = nn.Parameter(torch.Tensor(num_channels, 1, hidden_dim))
        bias_in = nn.Parameter(torch.Tensor(num_channels, hidden_dim))
        nn.init.xavier_uniform_(weight_in)
        nn.init.zeros_(bias_in)
        self.weights.append(weight_in)
        self.biases.append(bias_in)
        
        for layer in range(num_hidden_layers-1):
            weight = nn.Parameter(torch.Tensor(num_channels, hidden_dim, hidden_dim))
            bias = nn.Parameter(torch.Tensor(num_channels, hidden_dim))
            nn.init.xavier_uniform_(weight)
            nn.init.zeros_(bias)
            self.weights.append(weight)
            self.biases.append(bias)
            
        weight_out = nn.Parameter(torch.Tensor(num_channels, hidden_dim, 1))
        bias_out = nn.Parameter(torch.Tensor(num_channels, 1))
        nn.init.xavier_uniform_(weight_out)
        nn.init.zeros_(bias_out)
        self.weights.append(weight_out)
        self.biases.append(bias_out)
        
        self.hidden_activation = nn.ReLU()
        if positive:
            self.output_activation = nn.Sigmoid()
            print("using sigmoid")
        else:
            self.output_activation = nn.Identity()
            print("uning identity")
        
    def forward(self, x):
        #x shape: (batch size, num channels, 1)
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum('bci,cio->bco', x, weight)
            x = x + bias.unsqueeze(0) # bias from (num channels, hidden dim) to (1, num channels, hidden dim)
            
            if idx < len(self.weights) - 1: x = self.hidden_activation(x)
            else: x = self.output_activation(x)
        
        return x #shape: (batch size, num channels, 1)
    
class IntegralHat(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_channels = args.num_channels
        self.a = nn.Parameter(torch.randn(num_channels, requires_grad = True)) # 0D tensor by using torch.randn(())
        self.b = nn.Parameter(torch.randn(num_channels, requires_grad = True))
        self.scaling = nn.Parameter(torch.randn(num_channels, requires_grad = True))
        
        self.integrand = MultiIntegrandNN(num_channels, 1, args.integrand_hidden_dim, args.integrand_num_hidden_layers)
        
        self.temp = args.temp
        self.elu_alpha = args.alpha
        
    def forward(self, x):
        #x shape: (batch size, num channels)
        if len(x.shape) == 2: x = x.unsqueeze(-1) #(batch size, num channels, 1)
        adjust_shape = lambda tensor: tensor.unsqueeze(0).unsqueeze(-1).expand(x.shape) #(batch size, num channels, 1)
        
        lower = adjust_shape(self.a)
        upper = F.relu(x-lower) + lower
        #width = adjust_shape(torch.abs(self.b))
        width = adjust_shape(F.elu(self.b, alpha = self.elu_alpha) + self.elu_alpha)
        scale = adjust_shape(self.scaling * self.scaling)
        scale = 1
        
        y = ParallelNeuralIntegral.apply(lower, upper, self.integrand, _flatten(self.integrand.parameters()))
        upper_supp_ind = 1 - F.sigmoid(self.temp * (x - (lower + width))) #(batch size, num channels, 1)
        
        assert upper_supp_ind.shape == x.shape, f"upper supp indicator shape: {upper_supp_ind.shape}, x shape: {x.shape}"
        y = y * upper_supp_ind * scale
        assert y.shape == x.shape
        return y.squeeze() #(batch size, num channels)

class DoubleIntegralHat(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_channels = args.num_channels
        self.a = nn.Parameter(torch.randn(num_channels, requires_grad = True)) # (num_channels,)
        self.b = nn.Parameter(torch.randn(num_channels, requires_grad = True)) # (num_channels,)
        self.m = nn.Parameter(torch.randn(num_channels, requires_grad = True))
        self.scaling = nn.Parameter(torch.randn(num_channels)) # (num_channels,)
        
        self.integrand1 = MultiIntegrandNN(num_channels, 1, args.integrand_hidden_dim, args.integrand_num_hidden_layers)
        self.integrand2 = MultiIntegrandNN(num_channels, 1, args.integrand_hidden_dim, args.integrand_num_hidden_layers)
        
        self.temp = args.int_temp
        self.elu_alpha = args.int_alpha
        
    def forward(self, x):
        """
        x: (batch size, num channels)
        """
        if len(x.shape) == 2: x = x.unsqueeze(-1) #(batch size, num channels, 1)
        bsize, num_channels, _ = x.shape
        adjust_shape = lambda tensor: tensor.unsqueeze(0).unsqueeze(-1).expand(x.shape)
        a = adjust_shape(self.a)
        b = adjust_shape(F.elu(self.b, alpha = self.elu_alpha) + self.elu_alpha)
        m = adjust_shape(F.sigmoid(self.m))
        
        integral1_lower_limit = a
        switch_point = a + m * b
        integral2_upper_limit = a + b
        
        ind_x_greater_than_switch = lambda x: F.sigmoid(self.temp * (x - switch_point)) #(batch size, num channels, 1)
        
        upper_1 = F.relu(x - a) + a
        integral1 = ParallelNeuralIntegral.apply(integral1_lower_limit, upper_1, self.integrand1, _flatten(self.integrand1.parameters())) #(batch size, num channels, 1)
        integral1 = integral1 * (1 - ind_x_greater_than_switch(x)) #(batch size, num channels, 1)
        
        lower_2 = a + b - F.relu(a + b - x) #(batch size, num channels, 1)
        integral2 = ParallelNeuralIntegral.apply(lower_2, integral2_upper_limit, self.integrand2, _flatten(self.integrand2.parameters()))
        
        full_integral_1 = ParallelNeuralIntegral.apply(integral1_lower_limit, switch_point, self.integrand1, _flatten(self.integrand1.parameters()))
        full_integral_2 = ParallelNeuralIntegral.apply(switch_point, integral2_upper_limit, self.integrand2, _flatten(self.integrand2.parameters()))
        scaling_factor_2 = torch.div(full_integral_1, full_integral_2 + 1e-5)
        integral2 = integral2 * scaling_factor_2 * ind_x_greater_than_switch(x) #(batch size, num channels, 1)
        
        y = integral1 + integral2
        assert y.shape == x.shape
        return y.squeeze() #(batch size, num channels)
    
class MaskModule(nn.Module):
    def __init__(self, num_channels, temp, elu_alpha = None):
        super().__init__()
        #self.device = device
        self.a = nn.Parameter(torch.randn(num_channels))
        self.b = nn.Parameter(torch.randn(num_channels))
        self.m = nn.Parameter(torch.randn(num_channels))
        self.temp = temp
        self.elu_alpha = None
        print(f"INTEGRAL HAT, alpha: {self.elu_alpha}, temp: {self.temp}")
        
    def forward(self, x):
        #x: (batch size, num channels, 1)
        adjust_shape = lambda tensor: tensor.unsqueeze(0).unsqueeze(-1).expand(x.shape)
        if self.elu_alpha is None:
            width = torch.abs(self.b)
        else:
            width = F.elu(self.b, alpha = self.elu_alpha) + self.elu_alpha
        left = adjust_shape(self.a)
        mid = adjust_shape(self.a + F.sigmoid(self.m) * width)
        right = adjust_shape(self.a + width)
        mask1 = F.sigmoid(self.temp * (x - left)) * F.sigmoid(self.temp * (mid - x))
        mask2 = F.sigmoid(self.temp * (x - mid)) * F.sigmoid(self.temp * (right - x))
        return mask1, mask2
    
    def get_left(self):
        return self.a
    
    def get_mid(self):
        return self.a + F.sigmoid(self.m) * torch.abs(self.b)
    
    def get_right(self):
        return self.a + torch.abs(self.b)
    
class MaskedIntegrandLeft(nn.Module):
    def __init__(self, maskmodule, integrand):
        super().__init__()
        self.maskmodule = maskmodule
        self.integrand = integrand
        
    def forward(self, x):
        #x: (batch, chnnels, 1)
        mask1, _ = self.maskmodule(x)
        return mask1 * self.integrand(x)
    
    def parameters(self):
        return list(self.maskmodule.parameters()) + list(self.integrand.parameters())
    
class MaskedIntegrandRight(nn.Module):
    def __init__(self, maskmodule, integrand):
        super().__init__()
        self.maskmodule = maskmodule
        self.integrand = integrand
        
    def forward(self, x):
        #x: (batch, chnnels, 1)
        _, mask2 = self.maskmodule(x)
        return mask2 * self.integrand(x)
    
    def parameters(self):
        return list(self.maskmodule.parameters()) + list(self.integrand.parameters())
    
class DoubleIntegralHatNew(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_channels = args.num_channels
        self.temp = args.int_temp
        self.elu_alpha = args.int_alpha if args.use_int_alpha else None
        self.maskmodule = MaskModule(num_channels, self.temp, self.elu_alpha)
        self.scaling = nn.Parameter(torch.randn(num_channels)) # (num_channels,)
        
        self.integrand1 = MultiIntegrandNN_new(num_channels, args.integrand_hidden_dim, args.integrand_num_hidden_layers)
        self.integrand2 = MultiIntegrandNN_new(num_channels, args.integrand_hidden_dim, args.integrand_num_hidden_layers)
        
    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(-1) #(batch, channels, 1)
        adjust_shape = lambda tensor: tensor.unsqueeze(0).unsqueeze(-1).expand(x.shape)
        left = adjust_shape(self.maskmodule.get_left())
        mid = adjust_shape(self.maskmodule.get_mid())
        right = adjust_shape(self.maskmodule.get_right())
        height = adjust_shape(torch.abs(self.scaling))
        full_integral1 = ParallelNeuralIntegral.apply(left, mid, self.integrand1, _flatten(self.integrand1.parameters()))
        full_integral2 = ParallelNeuralIntegral.apply(mid, right, self.integrand2, _flatten(self.integrand2.parameters()))
        #scaling_factor = differentiable_safe_div(full_integral1, full_integral2)
        scaling_factor = differentiable_safe_div(full_integral1, full_integral2)
        upper_limit_left = F.relu(x - left) + left
        lower_limit_right = right - F.relu(right - x)
        integral_left = ParallelNeuralIntegral.apply(left, upper_limit_left, self.integrand1, _flatten(self.integrand1.parameters()))
        integral_right = ParallelNeuralIntegral.apply(lower_limit_right, right, self.integrand2, _flatten(self.integrand2.parameters()))
        mask1, mask2 = self.maskmodule(x)
        result = (integral_left * mask1 + scaling_factor * integral_right * mask2) * height
        return result.squeeze()
        

def get_act(activation_key:str, args = None):
    if args is None and activation_key not in ['relu', 'identity']:
        raise AssertionError()
    if activation_key == 'trainable_hat':
        return TrainableHat(args)
    elif activation_key == 'hat':
        return Hat(args)
    elif activation_key == 'integral_hat':
        return IntegralHat(args)
    elif activation_key == 'double_integral_hat':
        return DoubleIntegralHatNew(args)
    elif activation_key == 'relu':
        return nn.ReLU()
    elif activation_key == 'l_relu':
        negative_slope = args.lrelu_slope
        return nn.LeakyReLU(negative_slope=negative_slope)
    elif activation_key == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Activation {activation_key} not supported")