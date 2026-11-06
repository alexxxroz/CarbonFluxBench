'''
This module contains a custom torch function accounting for sample quality, IGBP and Koppen class balance, and
using a physics-guided constraint.
'''

from sklearn.preprocessing import StandardScaler
import torch

class CustomLoss(torch.nn.Module):
    """Custom loss function with carbon balance constraint and climate weighting."""
    def __init__(self, 
                IGBP_weights: dict,
                Koppen_weights: dict,
                device: str='cuda',
                y_scaler: StandardScaler=None
        ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.IGBP_weights = IGBP_weights
        self.Koppen_weights = Koppen_weights
        self.device = device
        
        IGBP_CLASSES = ["CRO","CSH","CVM","DBF","DNF","EBF","ENF","GRA","MF","OSH","SAV","SNO","URB","WAT","WET","WSA"]
        KOPPEN_CLASSES = ["A","B","C","D","E"]
        self.id2igbp = {i:c for i,c in enumerate(IGBP_CLASSES)}
        self.id2koppen = {i:c for i,c in enumerate(KOPPEN_CLASSES)}
        if y_scaler is not None:
            self.y_mean = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
            self.y_scale = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device)
        else:
            self.y_mean = self.y_scale = None

    def forward(self, preds, targets, nee_qc=None, igbp=None, koppen=None, alpha=0.1): 
        mse_loss = self.mse(preds, targets)

        if igbp is not None:
            igbp_flat = igbp.flatten()
            weights = torch.tensor([self.IGBP_weights[self.id2igbp[key.item()]] for key in igbp_flat], dtype=torch.float32).to(self.device)
            igbp = weights.reshape(igbp.shape)

        if koppen is not None:
            koppen_flat = koppen.flatten()
            weights = torch.tensor([self.Koppen_weights[self.id2koppen[key.item()]] for key in koppen_flat], dtype=torch.float32).to(self.device)
            koppen = weights.reshape(koppen.shape)
            
        # Carbon balance constraint: NEE = -GPP + RECO
        if targets.shape[-1]==3 and self.y_mean is not None:
            preds_inv = preds * self.y_scale + self.y_mean # (GPP, RECO, NEE) in physical units
            GPP, RECO, NEE = preds_inv.unbind(-1)
            balance_error = self.mse(NEE, RECO - GPP)
        else:
            balance_error = 0
            
        # Apply quality control and climate weighting
        if nee_qc is not None:
            total_loss = mse_loss.mean(axis=2) * nee_qc
        else:
            total_loss = mse_loss.mean(axis=2) 
            
        if igbp is not None:
            total_loss *= igbp 
        if koppen is not None:
            total_loss *= koppen 
            
        total_loss += balance_error*alpha

        return total_loss.mean()