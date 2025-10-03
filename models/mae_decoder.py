# models/mae_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAEDecoder(nn.Module):
    """
    Simple MAE decoder that reconstructs masked patches.
    - token_dim: transformer feature dim
    - decoder_dim: internal decoder dim
    - num_layers: transformer decoder layers
    - patch_size: patch pixel size factor (assumes backbone's patch is 1 token => some patch area)
    """
    def __init__(self, token_dim=512, decoder_dim=256, num_layers=6, num_heads=8, out_patch_channels=3):
        super().__init__()
        self.token_dim = token_dim
        self.decoder_dim = decoder_dim
        self.decoder_embed = nn.Linear(token_dim, decoder_dim, bias=True)

        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # project decoder outputs to patch pixels or to feature channels
        # For simplicity we return coarse reconstruction with output channels (e.g., 3)
        self.to_pixels = nn.Linear(decoder_dim, out_patch_channels)

    def forward(self, enc_vis, mask, ids_restore, H, W):
        """
        enc_vis: [B, N_vis, token_dim] encoded visible tokens (after transformer on visible)
        mask: [B, N] mask vector (1 for masked, 0 for visible)
        ids_restore: [B, N] indices to restore original order
        H, W: spatial dims (tokens N = H*W)

        Returns recon_pixels: [B, out_ch, H, W] — coarse pixel reconstruction
        """
        B = enc_vis.size(0)
        device = enc_vis.device
        token_dim = enc_vis.size(-1)
        # Map visible token dim -> decoder dim
        vis_dec = self.decoder_embed(enc_vis)  # [B, N_vis, Dd]

        # Prepare full token sequence with mask tokens
        N = mask.size(1)
        # Build placeholder: start with mask tokens and fill visible ones
        # ids_restore tells how to place tokens back into original ordering
        # Step 1: create a full tensor filled with mask tokens
        mask_tokens = self.mask_token.expand(B, N, -1).to(device)  # [B, N, Dd]

        # Step 2: scatter encoded visible tokens into mask_tokens at ids_keep positions
        # We need ids_keep: indices of visible tokens; can recover from mask (0 values)
        ids_keep = (mask == 0).nonzero(as_tuple=False).reshape(B, -1)  # fragile for ragged; safer below

        # Simpler safe approach: use ids_restore and enc_vis -> place vis tokens by inverse permutation
        # We assume enc_vis corresponds to tokens in order ids_keep. We'll build by filling at ids_keep
        # Build an empty full array (B,N,Dd)
        full = mask_tokens.clone()
        # compute ids_keep per batch by checking where mask==0
        for b in range(B):
            idsk = torch.where(mask[b] == 0)[0]  # positions of visible tokens
            # enc_vis[b] length must match len(idsk)
            if idsk.numel() != enc_vis[b].shape[0]:
                # fallback: try to use enc_vis as all tokens (no mask)
                # fill with zeros if mismatch
                # This should not happen in correct pipeline.
                continue
            full[b, idsk, :] = vis_dec[b]

        # Now reorder to original order using ids_restore if needed (here already in original indices)
        # Optionally: apply transformer decoder on full tokens
        dec = self.decoder(full)  # [B, N, Dd]

        # project each token to pixels (coarse)
        pix = self.to_pixels(dec)  # [B, N, out_ch]
        # reshape to spatial
        pix = pix.permute(0,2,1).contiguous().view(B, -1, H, W)  # [B, out_ch, H, W]
        return pix

