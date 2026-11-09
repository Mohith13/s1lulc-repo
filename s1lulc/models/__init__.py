"""Model factory: U-Net family with interchangeable encoders and initialisations.

Variants compared in the report:
* ``weights=None``        random initialisation (baseline)
* ``weights="imagenet"``  ImageNet-pretrained RGB encoder; segmentation_models_pytorch adapts
                          the first convolution to 2 input channels
* ``weights="ssl4eo_s1"`` ResNet-50 pretrained self-supervised (MoCo v2) on Sentinel-1 GRD
                          VV/VH from SSL4EO-S12 (Wang et al., 2023). Same sensor and the same
                          two channels as our input, so no channel adaptation is needed.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# SSL4EO-S12 ResNet-50 Sentinel-1 MoCo weights (timm layout), as distributed by TorchGeo
# (ResNet50_Weights.SENTINEL1_ALL_MOCO). Pinned to a revision for reproducibility.
SSL4EO_S1_URL = ("https://hf.co/torchgeo/resnet50_sentinel1_all_moco/resolve/"
                 "e79862c667853c10a709bdd77ea8ffbad0e0f1cf/resnet50_sentinel1_all_moco-906e4356.pth")
# Input normalisation used in SSL4EO-S12 pretraining (dB). Our training-set z-score is close
# to this, so pretrained features remain meaningful; fine-tuning absorbs the difference.
SSL4EO_S1_MEAN_DB = (-12.59, -20.26)
SSL4EO_S1_STD_DB = (5.26, 5.91)


def build_model(arch: str = "unet", encoder: str = "resnet34", in_channels: int = 2,
                n_classes: int = 7, weights: str | None = None):
    import segmentation_models_pytorch as smp

    if weights not in (None, "imagenet", "ssl4eo_s1"):
        raise ValueError(f"unknown weights {weights!r}")
    if weights == "ssl4eo_s1" and (encoder != "resnet50" or in_channels % 2):
        raise ValueError("ssl4eo_s1 weights require encoder=resnet50 and VV/VH channel pairs")
    model = smp.create_model(arch, encoder_name=encoder,
                             encoder_weights="imagenet" if weights == "imagenet" else None,
                             in_channels=in_channels, classes=n_classes)
    if weights == "ssl4eo_s1":
        load_ssl4eo_encoder(model)
    return model


def load_ssl4eo_encoder(model, state_dict: dict | None = None) -> int:
    """Load SSL4EO-S12 S1 weights into ``model.encoder``. Returns the number of loaded tensors.

    The checkpoint uses timm/torchvision ResNet key names (conv1, bn1, layer1..4, fc), which
    match the segmentation_models_pytorch ResNet encoder; ``fc`` is dropped. For multi-month
    input (VV/VH pairs for several months) the 2-channel first convolution is replicated for
    every month and divided by the number of months, so a constant time series produces the
    same activations as the single-month pretrained network.
    """
    import torch

    if state_dict is None:
        state_dict = torch.hub.load_state_dict_from_url(SSL4EO_S1_URL, map_location="cpu",
                                                        progress=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    target = model.encoder.state_dict()
    w = state_dict.get("conv1.weight")
    if w is not None and "conv1.weight" in target:
        c_in = target["conv1.weight"].shape[1]
        if c_in != w.shape[1] and c_in % w.shape[1] == 0:
            reps = c_in // w.shape[1]
            state_dict = dict(state_dict)
            state_dict["conv1.weight"] = w.repeat(1, reps, 1, 1) / reps
    matched = {k: v for k, v in state_dict.items()
               if k in target and tuple(v.shape) == tuple(target[k].shape)}
    expected = [k for k in target if not k.endswith("num_batches_tracked")]
    if len(matched) < 0.95 * len(expected):
        raise RuntimeError(f"SSL4EO checkpoint matched only {len(matched)}/{len(expected)} "
                           "encoder tensors; refusing to train on a partly random encoder")
    model.encoder.load_state_dict(matched, strict=False)
    log.info("loaded %d/%d SSL4EO-S12 encoder tensors", len(matched), len(expected))
    return len(matched)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
