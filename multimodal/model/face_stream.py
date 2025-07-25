from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn as nn 
import torch
from torch.amp import autocast
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import  to_2tuple, trunc_normal_
from timm.layers import DropPath
# from timm.models.vision_transformer import MLP, PatchEmbed
import pytorch_lightning as pl
# from timm.layers import mlp,patch_embed
from timm.layers.mlp import Mlp

#checked
class MambaVisionMixer(nn.Module):  #its mamba block 
    """
    hidden_states: (B, L, D)
    Returns: same shape as hidden_states
    """
    def __init__(
        self,
        d_model, #The input and output dimensionality of the hidden states.
        d_state=16,  #hinnden state dimensionality, used for the selective scan (h ki dimension)
        d_conv=3,
        expand=2,  # d_innner = expand * d_model (ssm kai formula mai x ki dimension d_inner hai)
        dt_rank="auto", # dt is learnable time step 
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    



#checked
class Attention(nn.Module):  #it performs multi head self attention
    """args:
        dim: input dimension
        num_heads: number of attention heads
        qkv_bias: if True, adds a learnable bias to query, key, value projections
        qk_norm: if True, normalizes the query and key before the attention
        attn_drop: dropout rate for attention weights
        proj_drop: dropout rate for the output projection
        norm_layer: normalization layer to apply to the query and key
        x: input tensor of shape (B, N, C) 
        output: output tensor of shape (B, N, C)
        """
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #Instead of using 3 separate layers, it's fused for speed
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x) # shape: (B, N, 3 * dim)
        qkv=qkv.reshape(B, N, 3, self.num_heads, self.head_dim) #(B, N, 3, num_heads, head_dim)
        qkv=qkv.permute(2, 0, 3, 1, 4) #  (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0) #splits the tensor along dimension 0 (the Q/K/V axis). each one has shape (B,num_heads,N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(    #instead of manually computing attention, we use PyTorch's built-in function
             q, k, v,
                dropout_p=self.attn_drop.p,
            )


            """dropout_p=self.attn_drop.p, ::: IN the softmax score for each token 
                it randomly drops out some of the wieghts so that model doesnt rely on a particyular patch 
                during training , it reduces overfitting"""
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)   #[B, N, C]  
        x = self.proj(x)  # since above step is only conctaneated info from all steps, this step mix features from diff attention heads
        x = self.proj_drop(x)
        return x



#checked
class PatchEmbed(nn.Module): 
    """input: B,T,C,H,W
       output: B*T, N, D"""
    def __init__(self,patch_dim,patch_size,num_frames,H_img,W_img,input_channels=3):
        super().__init__()
        self.conv=nn.Conv2d(
            in_channels=input_channels,
            out_channels=patch_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0)

        
        H_patches = int(H_img // patch_size)
        W_patches = int(W_img // patch_size)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, patch_dim, H_patches, W_patches))  # broadcasted over B, T
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames, patch_dim, 1, 1))   

        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(self,x):
        b,t,_,_,_= x.shape
        x=rearrange(x,"b t c h w -> (b t) c h w")  # (B*T, C, H, W)
        patches=self.conv(x)
        patches=rearrange(patches,'(b t) c h w -> b t c h w', b=b,t=t)  # (B*T, C, H, W) -> (B, T, C, H, W)

        patches=patches+ self.temporal_pos +self.spatial_pos # Adding temporal positional encoding

        patches=rearrange(patches,"b t c h w-> (b t) (h w) c")  # (B*T, N, D)
        return patches



#checked
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows  


#checked
class MambaAttnBlock(nn.Module):
    """Performs either Mamba or Attention based on the flags provided.
         take x: (B, L, D) 
         returns x: (B, L, D)

    """

    def __init__(self, dim, use_mamba=False, use_attn=False,layer_scale=None,drop_path=0.0,window_size=7):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_attn = use_attn
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if use_mamba:
            self.mamba = MambaVisionMixer(d_model=dim)
        elif use_attn:
            self.attn = Attention(dim=dim,num_heads=4)
        else:
            raise ValueError("Either use_mamba or use_attn must be True")

        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, drop=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        use_layer_scale = layer_scale is not None

        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        

    def forward(self, x):
        # if self.use_mamba:
        #     x = x + self.mamba(self.norm1(x))
        #     x = x + self.mlp(self.norm1(x))

        if self.use_attn:
            B, N, C = x.shape
            H= int(N**0.5)  # Assuming N is a perfect square for simplicity
            W=H
            x_img = x.transpose(1, 2).reshape(B, C, H, W) # Convert from token format (B, N, C) to image shape (B, C, H, W)
            x_win = window_partition(x_img, window_size=self.window_size) #ouputxs (num_windows*B, window_size*window_size, C)
            x_attent= self.attn(x_win)
            x_img = x_img + self.drop_path(self.gamma_1 * window_reverse(x_attent, window_size=self.window_size, H=H, W=W))
            x = x_img.flatten(2).transpose(1, 2)  #Convert back to (B, N, C) format

        else:
            x = x + self.drop_path(self.gamma_1 * self.mamba(self.norm1(x)))

        return x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))




#checked
class Downsample(nn.Module):
    """
    Down-sampling block"
    input: (B , L , D)
    output: (B , L/4 , 2D) , its equivalent to (B,H/2,W/2,2D)
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim

        
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        H_new=int(pow(x.shape[1],0.5))
        W_new=H_new
        x=rearrange(x,"b (h w) d -> b d h w ",h=H_new,w=W_new)  # (B, N, C) -> (B, C, H, W)
        x = self.reduction(x)
        x=rearrange(x ," b d h w -> b (h w) d ")
        return x


#checked
def window_reverse(windows, window_size, H, W): 
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x



#checked
class stageBlock(nn.Module):  #Hybrid mamba transformer block
    def __init__(self, num_layers, dim,window_size=7):
        """Expects input in the form (B,L,D)"""
        super().__init__()
        self.blocks = nn.Sequential(
            *[MambaAttnBlock(dim, use_mamba=True,window_size=window_size) for _ in range(num_layers)],
            *[MambaAttnBlock(dim, use_attn=True,window_size=window_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.blocks(x)


class model(pl.LightningDataModule):
    """
    input:(B,T,C,H,W)
    """
    def __init__(
        self,
        pretrain,
        patch_size,
        patch_dim,
        num_frames,
        H_img,
        W_img,
        num_layers,
        window_size):

        self.patch_dim = patch_dim
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.H_img = H_img
        self.W_img = W_img
        self.num_layers = num_layers
        self.window_size = window_size
        
      
        super().__init__()
        self.pretrain=pretrain
        # self.patchify=
        pass
    def forward(self,x):
        B,T,C,H,W = x.shape
        if self.pretrain:
            pass
        else:
            pass



if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(4, 25, 3, 224, 224).to(device)

    patching = PatchEmbed(patch_dim=96, patch_size=4, num_frames=25, H_img=224, W_img=224).to(device)
    stage = stageBlock(num_layers=2, dim=96, window_size=7).to(device)
    down = Downsample(dim=96, keep_dim=False).to(device)

    # Automatic Mixed Precision context
    with autocast(device_type="cuda",dtype=torch.float16):
        patches = patching(x)
        print(patches.shape)

        out = stage(patches)
        print(out.shape)

        out = down(out)
        print(out.shape)


