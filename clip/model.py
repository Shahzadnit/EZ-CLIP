from collections import OrderedDict
from typing import Tuple, Union
import math
from functools import reduce
from operator import mul
import numpy as np
from torch.nn import Conv2d, Dropout
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import cv2

def tensor_image_old(attn_weight):
    for i, weight in enumerate(attn_weight):
        heatmap = F.interpolate(weight.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # print(i)
        cv2.imwrite('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/test_img/img_%d.png'%(i),heatmap_colormap)
        
def tensor_image(attn_weight):
    b,_,_ = attn_weight.shape
    attn_weight =torch.mean(attn_weight,1)[:,1:].view(b,14,14)
    for i, weight in enumerate(attn_weight[:,]):
        heatmap = F.interpolate(weight.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # print(i)
        cv2.imwrite('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/test_img/img_%d.png'%(i),heatmap_colormap)



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


#################################### My Changes ########################################
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class Adapter_new(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.attn = nn.MultiheadAttention(D_features, 16,)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        x = x + self.attn(x, x, x, need_weights=False, attn_mask=None)[0]
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, config, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0., frames=8,model_for='image'):
        super().__init__()
        self.config = config
        self.T = frames
        self.model_for = model_for    
        self.attn = nn.MultiheadAttention(d_model, n_head,dropout=dropout)
        # self.prompt_attn = nn.MultiheadAttention(d_model, n_head,dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        if self.model_for == 'image' and not self.config.prompt.use and self.config.T_Adapter:
            self.T_Adapter = Adapter(d_model, skip_connect=False)
        self.Adapter = Adapter(d_model)
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    
    def attention_weight(self, x: torch.Tensor): 
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]
            
 
# for n, m in self.named_modules():
        #     if 'my_fc' in n:
        #         # print('My fc is initialized++++++++++++++++++++++++++++++')
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)
        # for n, m in self.named_modules():
        #     if 'my_fc2' in n:
        #         # print('My fc is initialized++++++++++++++++++++++++++++++')
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)
    def forward(self, x: torch.Tensor, T_prompt=None, Text_prompt=None, layer_num=None, return_attention=False):
        if self.model_for =='image':
            l, bt, d = x.size()    
            b = bt // self.T      
            if T_prompt is not None:
                b = bt // self.T
                x = x.view(l, b, self.T, d)
                ############### This for weighted mean  #########################
                T_prompt = T_prompt.expand(x.shape[1],-1,-1) + torch.mean(x, 0)                
                T_prompt = T_prompt.view(b, self.T, 1, d) 
                T_prompt = T_prompt.permute(1,2,0,3).view(self.T, b, d) 
                T_prompt = self.drop_path(self.attention(self.ln_1(T_prompt)))
                T_prompt = T_prompt.view(self.T, 1, b, d).permute(1,2,0,3)
                
                x = torch.cat([x, T_prompt], dim=0)
                x = x.view(l+1, -1, d)
            #################################################################    
            # if self.config.network.Return_Attention and layer_num==11:
            #     return self.attention_weight(self.ln_1(x)) 
            if return_attention: # ADDED
                return self.attention_weight(self.ln_1(x)) # ADDED
            ###################################
            if T_prompt is None and not self.config.prompt.use and self.config.T_Adapter:
                xt = x.view(l, b, self.T, d).permute(2,0,1,3).reshape(self.T,l*b,d)
                xt = self.T_Adapter(self.attention(self.ln_1(xt)))
                xt = xt.view(self.T,l,b,d).permute(1,0,2,3).reshape(l, self.T*b,d)
                x = x + self.drop_path(xt)
            ######################################
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x[:l,:,:]
            x = self.Adapter(x)
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            return x
        if self.model_for =='text':
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = self.Adapter(x)
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
            return x
    
    
class Transformer(nn.Module):
    def __init__(self, config, width: int, layers: int, heads: int, no_frame:int=1, patch_size:int=None, attn_mask: torch.Tensor = None, dropout=None,model_for='image'):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        self.no_frame= no_frame
        self.model_for = model_for
        self.config = config        
        val = math.sqrt(6. / float(3 * reduce(mul, (16,16), 1) + self.width))
        if self.model_for == 'image':
            if config.prompt.use:
                num_tokens = config.prompt.num_of_token
                self.prompt_proj = nn.Identity()
                self.prompt_dropout = Dropout(config.prompt.DROPOUT)
                scale = self.width ** -0.5
                # val = math.sqrt(6. / float(3 * reduce(mul, (patch_size,patch_size), 1) + self.width))  # noqa
                if config.prompt.INITIATION == "random":
                    if self.config.prompt.DEEP:
                        self.T_prompt_embeddings = nn.Parameter(torch.randn(self.layers, self.no_frame, self.width))
                    else:
                        self.T_prompt_embeddings = nn.Parameter(torch.randn(1, self.no_frame, self.width))
                    nn.init.uniform_(self.T_prompt_embeddings.data, -val, val)
                else:
                    raise ValueError("Other initiation scheme is not supported")
                        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(config, width, heads, attn_mask, dropout=dropout[i],frames=self.no_frame,model_for=self.model_for) for i in range(layers)])
        

    def forward(self, x: torch.Tensor):       
        if self.model_for=='image':
            if self.config.prompt.use:
                for i,block in enumerate(self.resblocks):
                    if i==0:
                        x = block(x,T_prompt=self.prompt_dropout(self.prompt_proj(self.T_prompt_embeddings[i:i+1,:,:])).to(x.dtype),layer_num=i)
                    elif self.config.prompt.DEEP:
                        x = block(x,T_prompt=self.prompt_dropout(self.prompt_proj(self.T_prompt_embeddings[i:i+1,:,:])).to(x.dtype),layer_num=i)
                    else:
                        x = block(x,T_prompt=None)
                return x
                
                
            else:
                return self.resblocks(x)
        if self.model_for=='text':
            return self.resblocks(x)
    
    def forward_attention(self, x: torch.Tensor):       
        if self.model_for=='image':
            if self.config.prompt.use:
                for i,block in enumerate(self.resblocks):
                    if i==0:
                        x = block(x,T_prompt=self.prompt_dropout(self.prompt_proj(self.T_prompt_embeddings[i:i+1,:,:])).to(x.dtype),layer_num=i)
                    elif self.config.prompt.DEEP:
                        if i < len(self.resblocks)-1:
                            x = block(x,T_prompt=self.prompt_dropout(self.prompt_proj(self.T_prompt_embeddings[i:i+1,:,:])).to(x.dtype),layer_num=i)
                        else:
                            x = block(x,T_prompt=self.prompt_dropout(self.prompt_proj(self.T_prompt_embeddings[i:i+1,:,:])).to(x.dtype),layer_num=i,return_attention=True)       
                    else:
                        x = block(x,T_prompt=self.prompt_dropout(self.prompt_proj(self.T_prompt_embeddings[i:i+1,:,:])).to(x.dtype))
                return x    
            else:
                for i,block in enumerate(self.resblocks):
                    if i < len(self.resblocks)-1:
                        x = block(x)
                    else:
                        x = block(x,return_attention=True)        
                return x
        if self.model_for=='text':
            return self.resblocks(x)
                


class VisualTransformer(nn.Module):
    def __init__(self, config, input_resolution: int, patch_size: int, width: int, no_frame:int, layers: int, heads: int, output_dim: int,dropout = None,joint=False, emb_dropout = 0.):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.num_frame = no_frame
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, no_frame, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        self.config = config
        if joint:
            print('=====using joint space-time====')
            self.time_embedding = nn.Parameter(scale * torch.randn(T, width))
        if emb_dropout > 0:
            print('emb_dropout:{}'.format(emb_dropout))

        ## Attention Blocks
        self.transformer = Transformer(config, width, layers, heads, no_frame, patch_size, dropout=dropout,model_for='image')

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        # self.my_fc =  nn.Linear(197, 1)
        # self.my_fc2 =  nn.Linear(self.config.data.num_segments, self.config.data.num_segments)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        
        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frame)
        x = x + self.temporal_embedding.to(x.dtype)
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        if self.joint:
            B = x.shape[0] // self.T
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=self.T)
            x = x + self.time_embedding.to(x.dtype)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=self.T)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
         # LND -> NLD
        x = x.permute(1, 0, 2) 
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x
    def forward_attention(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        
        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frame)
        x = x + self.temporal_embedding.to(x.dtype)
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.forward_attention(x)
        # x = x.permute(1, 0, 2) 
        # x = torch.mean(self.ln_post(x[:,:-1,:]),2).view(8,14,14)
        return x

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 config,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 no_frame: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,joint=False,
                 tsm=False, T=8,dropout = 0., emb_dropout = 0.
                 ):
        super().__init__()
        
        self.config = config
        self.context_length = context_length
        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(config,
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            no_frame = no_frame,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,joint=joint,dropout=dpr,
            emb_dropout=emb_dropout
        )
        if tsm:
            print('=========using TSM==========')
            from modules.temporal_shift import make_temporal_shift_vit
            make_temporal_shift_vit(self.visual, T)

        self.transformer = Transformer(config,
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=dpr,
            model_for='text'
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        self.init_adapter()

    def init_adapter(self):

        ## initialize S_Adapter
        for n, m in self.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            # print('--------------Adapter initialize-------------')
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        # if self.config.prompt.use:
        #     # num_tokens = self.config.prompt.num_of_text_token
        #     num_tokens = 0
        #     mask = torch.empty(self.context_length+num_tokens, self.context_length+num_tokens)
        # else:
        #     mask = torch.empty(self.context_length, self.context_length)    
        mask = torch.empty(self.context_length, self.context_length)     
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))
    
    
    def encode_image_attention(self, image):
        return self.visual.forward_attention(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, config, tsm=False,T=8,dropout=0., joint=False,emb_dropout=0.,pretrain=True):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = CLIP(
        embed_dim,config,
        image_resolution, vision_layers, vision_width, vision_patch_size,T,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  tsm=tsm,T=T,joint=joint,
        dropout=dropout, emb_dropout=emb_dropout
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if tsm:
        for k in list(state_dict.keys()):
            if k.find("conv1")>-1 and k.find("layer")>-1: 
                n_k = k.split('conv1.')[0]+'conv1.net.'+k.split('conv1.')[1]
                state_dict[n_k] = state_dict.pop(k)
            if k.find("resblocks")>-1 and k.find("visual")>-1: 
                tmp = ''
                for i, t_ in enumerate(k.split('resblocks.')[1].split('.')):
                    if i>=1:
                        tmp += '.' + t_ 
                
                n_k = k.split('resblocks.')[0]+'resblocks.' + k.split('resblocks.')[1].split('.')[0]+'.net'+ tmp
#                 print(n_k)
                state_dict[n_k] = state_dict.pop(k)

    convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        if joint:  #or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict,strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        print('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict,strict=False)

    return model.eval()
