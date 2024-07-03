# following similar structure as in:
#  https://github.com/google-research/slot-attention-video/blob/main/savi/modules/attention.py

import numpy as np
import torch
from torch import nn
from einops import einsum, rearrange, reduce

class GeneralizedMHDPAttentionModule(nn.Module):
    """Generalized Multi Head Dot Product (MHDP) Attention module

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 n_head,
                 n_feat_1,
                 n_feat_out,
                 n_qk,
                 n_v,
                 *args,
                 n_feat_2=None,
                 bias=False,
                 softmax_axis="k",
                 epsilon=1e-8,
                 **kwargs):
        """returns the updates resulting from the self-attention or 
        cross-attention, which is to be combined with the residual signal at a
        later stage. also returns the attention mask

        Args:
            softmax_axis (str, optional): the axis to perform softmax over. 
            Defaults to "k" or key dimension. Alternative performs softmax over 
            the "q" or query, as in Slot Attention
        """
        super().__init__()
        self.n_head = n_head
        self.n_feat_1 = n_feat_1
        self.n_feat_out = n_feat_out
        self.n_qk = n_qk
        self.n_v = n_v
        self.n_feat_2 = n_feat_2
        self.bias = bias
        self.sf = 1./np.sqrt(n_qk)
        self.softmax_axis = softmax_axis
        self.epsilon = epsilon

        self.w_qf = nn.Linear(self.n_feat_1,self.n_head*self.n_qk,bias=self.bias)

        self.cross_attn = self.n_feat_2 is not None

        # do cross attention if n_feat_2 is not none; else do self attention
        self.n_feat_2 = \
            self.n_feat_1 if self.n_feat_2 is None else self.n_feat_2
        self.w_kf = nn.Linear(self.n_feat_2,self.n_head*self.n_qk,
                              bias=self.bias)
        self.w_vf = nn.Linear(self.n_feat_2,self.n_head*self.n_v,
                              bias=self.bias)
        
        # the fully connected layer combining all heads
        self.fc_all_heads = nn.Linear(self.n_head*self.n_v,self.n_feat_out,
                                      bias=self.bias)
        
    def forward(self, feature1, feature2=None, mask=None):

        # feature1 = rearrange(feature1,"b n n_feat_1 -> b n n_feat_1")

        if self.cross_attn:
            assert feature2 is not None,\
                 "Feature 2 required for cross attention!"

        q = rearrange(self.w_qf(feature1),"b n (h n_qk) -> h b n n_qk",
                      h=self.n_head)
        if feature2 is None:
            feature2 = feature1
        k = rearrange(self.w_kf(feature2),"b m (h n_qk) -> h b m n_qk",
                      h=self.n_head)
        v = rearrange(self.w_vf(feature2),"b m (h n_v) -> h b m n_v",
                      h=self.n_head)
        
        # compute dot product
        dot_prod = einsum(q,k,"h b n Dq, h b m n_v -> h b n m")*self.sf

        # deal with masking potentially here
        if mask is not None:
            raise NotImplementedError

        # apply softmax
        if self.softmax_axis == "q":
            attn = torch.softmax(dot_prod, dim=-2) # shape: h b n m

            # if softmax over query, i.e., slot attention, do weighted sum over
            # the key axis
            attn = attn/(reduce(attn, "h b n m -> h b n 1","sum")+self.epsilon)

        elif self.softmax_axis == "k":
            attn = torch.softmax(dot_prod,axis=-1)

        # times value
        output = einsum(attn, v, "h b n m, h b m n_v -> h b n n_v")
        output = rearrange(output, "h b n n_v -> b n (h n_v)")

        updates = self.fc_all_heads(output)
        attn = rearrange(attn, "h b n m -> b h n m")

        return updates, attn

class SlotAttentionMLP(nn.Module):
    def __init__(self,
                 n_feat_in,
                 n_hidden,
                 n_feat_out,
                 n_layers,
                 pre_post_ln,
                 actv_fn="relu",
                 add_residual=True):
        super().__init__()
        self.add_residual = add_residual

        layers = []

        # pre layer norm
        if pre_post_ln == "pre":
            layers.append(nn.LayerNorm(normalized_shape=n_feat_in))

        # fc layers
        n_in = n_feat_in
        n_out = n_hidden
        for _ in range(n_layers):
            layers.append(nn.Linear(n_in,n_out))
            n_in = n_out
            n_out = n_hidden
            if actv_fn=="relu":
                layers.append(nn.ReLU())
        
        layers.append(nn.Linear(n_in, n_feat_out))

        # post layer norm
        if pre_post_ln == "post":
            layers.append(nn.LayerNorm(normalized_shape=n_feat_out))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        x = self.mlp(input)

        if self.add_residual:
            x = x + input
        
        return x

class SlotAttentionModule(nn.Module):
    """defines the slot module

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 n_slot,
                 n_slot_feat,
                 n_iter,
                 attn_module_config:dict) -> None:
        """_summary_

        Args:
            n_slot (_type_): the number of slots
            d_feature (_type_): the feature length of each slot
            attn_module_config (dict): _description_
            gru_config (_type_): _description_
        """
        super().__init__()

        self.n_iter = n_iter

        self.ln = nn.LayerNorm(normalized_shape=n_slot_feat)
        self.attn = GeneralizedMHDPAttentionModule(**attn_module_config)

        self.gru = nn.GRUCell(input_size=n_slot_feat,
                              hidden_size=n_slot_feat)
        
        self.mlp = SlotAttentionMLP(n_feat_in=n_slot_feat,n_hidden=512,
                                    n_feat_out=n_slot_feat,n_layers=2,
                                    pre_post_ln="pre",add_residual=True)
    
    def forward(self, slots, other_features=None):

        for _ in range(self.n_iter): # iterations
            prev_slots = slots
            slots = self.ln(slots)
            updates, attn_mask = self.attn(slots, other_features)
            slots = self.gru(updates, prev_slots)

        slots = self.mlp(slots)

        return slots, attn_mask