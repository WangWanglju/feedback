import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
import numpy as np
# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = AutoConfig.from_pretrained(config_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model = self.re_init(self.model, 1)
        
        if self.cfg.pooling == 'weighted_pooling':
            self.pool = WeightedLayerPooling(num_hidden_layers=13, layer_start = 7)
        else:
            self.pool = MeanPooling()

        if self.cfg.multi_sample_dropout==True:
            self.dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1,0.5,5)])  
        
        self.concat_fc = nn.Linear(self.config.hidden_size*3, self.config.hidden_size)   

        self.fc = nn.Linear(self.config.hidden_size, 6)

        self._init_weights(self.concat_fc)

        self._init_weights(self.fc)

        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

        if cfg.mixout > 0.:
            self.model = self.apply_mixout(self.model, cfg.mixout)

    def apply_mixout(self, model, mixout=None):
        if mixout > 0:
            print('Initializing Mixout Regularization')
            for sup_module in model.modules():
                for name, module in sup_module.named_children():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    if isinstance(module, nn.Linear):
                        target_state_dict = module.state_dict()
                        bias = True if module.bias is not None else False
                        new_module = MixLinear(
                            module.in_features, module.out_features, bias, target_state_dict["weight"], mixout
                        )
                        new_module.load_state_dict(target_state_dict)
                        setattr(sup_module, name, new_module)
            print('Done.!')
        return model
    
    def re_init(self, model, layer_num):
        for module in model.encoder.layer[-layer_num:].modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        return model   

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def multi_concat(self, outputs, inputs):
        mean_feature = self.pool(outputs, inputs['attention_mask'])
        
        # attention based sentence representation
        weights = self.attention(outputs)

        attention_feature = torch.sum(weights * outputs, dim=1)
        
        # CLS Token representation
        cls_token_feature = outputs[:, 0, :] # only cls token
        
        # Concat them
        combine_feature = torch.cat([mean_feature, attention_feature, cls_token_feature], dim = -1)
        return combine_feature

    def feature(self, inputs):
        outputs = self.model(**inputs)
        outputs = outputs.last_hidden_state
        outputs = self.multi_concat(outputs, inputs)

        # MLP
        feature = self.concat_fc(outputs)

        return feature

    def forward(self, inputs, extract_feature = False):
        feature = self.feature(inputs)
        if extract_feature:
            return feature
        if self.cfg.multi_sample_dropout == True:
            output = torch.mean(torch.stack([
            self.fc(dropout(feature)) for i, dropout in enumerate(self.dropout)
            ], dim=0), dim=0)
        else:
            output = self.fc(feature)
        return output


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, features, attention_mask):
        if isinstance(features, dict):
            last_hidden_state = features['last_hidden_state']
        else:
            last_hidden_state = features
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings

#Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers - layer_start), dtype=torch.float)
            )
        self.pooling = MeanPooling()

    def forward(self, features, mask):
        ft_all_layers = features['hidden_states']

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features = self.pooling(weighted_average, mask)
        return features

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd.function import InplaceFunction
import math
class Mixout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("A mix probability of mixout has to be between 0 and 1," " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]
    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, self.p, self.training), self.bias)

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out", self.p, self.in_features, self.out_features, self.bias is not None
        )



if __name__ == '__main__':
    from collections import OrderedDict
    def delete(state_dict, prefix="module."):
        state_dict.pop('n_averaged')
        keys = sorted(state_dict.keys())
        if not any(key.startswith(prefix) for key in keys):
            return state_dict
        
        stripped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            stripped_state_dict[key.replace(prefix, "")] = value
        return stripped_state_dict

    from config import CFG
    OUTPUT_DIR = f'./exp/{CFG.EXP_NAME}/' 
    fold = 0
    model = CustomModel(CFG, config_path=None, pretrained=False)
    print('model loading...', OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_swa.pth")
    state = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_swa.pth",
                    map_location=torch.device('cpu'))
    state['model'] = delete(state['model'])
    for n,p in state['model'].items():
        # state['model'][n] = n.replace('module.', '')
        print(n)

    model.load_state_dict(state['model'], strict=True)