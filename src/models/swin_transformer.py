import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding layer that embeds [B, C, T, H, W] into [B, T', H', W', Embed_Dim].
    """

    def __init__(self, in_chans=8, embed_dim=96, patch_size=(1, 4, 4)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    Single Layer Block
    """

    def __init__(self, dim, num_heads=4, window_size=(2, 4, 4), shift_size=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        return shortcut + x


class SwinTransformer3D(nn.Module):
    """
    Input:  [B, C_in, T_in, H, W]
    Output: [B, C_out, T_out, H, W]
    """

    def __init__(
        self,
        in_chans=8,
        out_chans=8,
        in_len=7,
        out_len=1,
        embed_dim=96,
        num_heads=[3, 6, 12],
        depths=[2, 2, 2],
        window_size=(2, 4, 4),
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.in_len = in_len
        self.out_len = out_len
        self.embed_dim = embed_dim
        self.window_size = window_size

        # 3D Patch Embedding
        self.patch_embed = PatchEmbed3D(
            in_chans=in_chans, embed_dim=embed_dim, patch_size=(1, 4, 4)
        )

        # Transformer Encoder
        self.layers = nn.ModuleList()
        curr_dim = embed_dim
        for i, depth in enumerate(depths):
            stage = nn.ModuleList(
                [
                    SwinTransformerBlock3D(
                        dim=curr_dim, num_heads=num_heads[i], window_size=window_size
                    )
                    for _ in range(depth)
                ]
            )
            self.layers.append(stage)

        # Prediction Decoder
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=embed_dim,
                out_channels=embed_dim // 2,
                kernel_size=(1, 4, 4),
                stride=(1, 4, 4),
            ),
            nn.GELU(),
            nn.Conv3d(
                in_channels=embed_dim // 2,
                out_channels=out_chans * out_len,
                kernel_size=3,
                padding=1,
            ),
        )

    def _pad_input(self, x):
        """
        Pads H and W dimensions to multiples of size 4.
        """
        _, _, _, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def forward(self, x):
        b, c, t, orig_h, orig_w = x.shape

        # Match patch stride
        x, pad_h, pad_w = self._pad_input(x)

        # Patch Embedding
        x = self.patch_embed(x)

        # Transformer Encoder Layers
        for stage in self.layers:
            for block in stage:
                x = block(x)

        # Back to spatial dimensions
        x = x.permute(0, 4, 1, 2, 3)

        # Crop to original grid
        out = self.decoder_head(x)
        out = out[:, :, -1:, :orig_h, :orig_w]
        out = out.view(b, self.out_chans, self.out_len, orig_h, orig_w)

        return out
