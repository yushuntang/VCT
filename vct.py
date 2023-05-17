"""
Copyright to VCT Authors,
built upon on SAR code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class VCT(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "VCT requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt_vct(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss
        outputs = self.model(x)
        if True:
            for name, param in self.model.named_parameters():
                if name in ["instance_token"]:
                    param.grad.data.zero_()
                    param.data.zero_()
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_vct(x, model, optimizer, margin, reset_constant, ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward
    outputs = model(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    entropys2 = softmax_entropy(model(x))
    entropys2 = entropys2[filter_ids_1]  # second time forward  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            print("ema < 0.2, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag


def collect_params(model, args):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        for np, p in m.named_parameters(): # VCT
            if 'cls_token' in np:
                print('finding token:',f"{np}: {p.shape}")
                p.requires_grad_(True)
                params += [{'params': p, 'lr': args.cls_token_lr}]
                names.append(f"{nm}.{np}")
            if 'instance_token' in np:
                print('finding token:',f"{np}: {p.shape}")
                p.requires_grad_(True)
                params += [{'params': p, 'lr': args.instance_token_lr}]
                names.append(f"{nm}.{np}")

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)): # SAR
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  
                    # params.append(p)
                    params += [{'params': p, 'lr': args.lr}]
                    names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with VCT."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    for np, p in model.named_parameters():
        if 'token' in np:
            p.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with VCT."""
    is_training = model.training
    assert is_training, "VCT needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "VCT needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "VCT should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "VCT needs normalization layer parameters for its optimization"
