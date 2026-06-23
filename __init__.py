"""ERM baseline for the Waterbirds spurious-correlation project.

Modules:
    config     — hyperparameters and paths
    data       — Waterbirds loading + 4-subgroup bookkeeping
    model      — ResNet-18 backbone with 2-class head
    train_erm  — standard ERM training loop (every example weighted equally)
    evaluate   — subgroup-aware evaluation (worst-group accuracy)
    saliency   — Grad-CAM for visual diagnosis of the shortcut
    run_erm    — orchestrates train -> evaluate -> saliency
"""
