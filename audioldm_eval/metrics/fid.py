import torch
import numpy as np
import scipy.linalg
import logging

# Configure logging for FID metrics
fid_logger = logging.getLogger(__name__)

# FID评价保真度，越小越好
def calculate_fid(
    featuresdict_1, featuresdict_2, feat_layer_name
):  # using 2048 layer to calculate
    eps = 1e-6
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]

    fid_logger.debug(f"🔢 Calculating FID with features shapes: {features_1.shape}, {features_2.shape}")

    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2

    # Check for minimum sample requirements
    if features_1.shape[0] < 2:
        fid_logger.warning(f"⚠️ Insufficient samples for covariance calculation: {features_1.shape[0]} samples (need ≥2)")
        return {"frechet_distance": float('inf'), "error": "Insufficient samples for covariance calculation (need ≥2)"}
    
    if features_2.shape[0] < 2:
        fid_logger.warning(f"⚠️ Insufficient samples for covariance calculation: {features_2.shape[0]} samples (need ≥2)")
        return {"frechet_distance": float('inf'), "error": "Insufficient samples for covariance calculation (need ≥2)"}

    # Calculate statistics with validation
    try:
        features_1_np = features_1.numpy()
        features_2_np = features_2.numpy()
        
        fid_logger.debug(f"📊 Features 1 stats: shape={features_1_np.shape}, range=[{features_1_np.min():.4f}, {features_1_np.max():.4f}]")
        fid_logger.debug(f"📊 Features 2 stats: shape={features_2_np.shape}, range=[{features_2_np.min():.4f}, {features_2_np.max():.4f}]")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(features_1_np)) or np.any(np.isinf(features_1_np)):
            fid_logger.error(f"❌ Features 1 contain NaN or Inf values")
            return {"frechet_distance": float('inf'), "error": "Features contain NaN or Inf values"}
            
        if np.any(np.isnan(features_2_np)) or np.any(np.isinf(features_2_np)):
            fid_logger.error(f"❌ Features 2 contain NaN or Inf values")
            return {"frechet_distance": float('inf'), "error": "Features contain NaN or Inf values"}

        stat_1 = {
            "mu": np.mean(features_1_np, axis=0),
            "sigma": np.cov(features_1_np, rowvar=False),
        }
        stat_2 = {
            "mu": np.mean(features_2_np, axis=0),
            "sigma": np.cov(features_2_np, rowvar=False),
        }
        
        fid_logger.debug(f"📊 Statistics calculated - mu1 shape: {stat_1['mu'].shape}, sigma1 shape: {stat_1['sigma'].shape}")
        fid_logger.debug(f"📊 Statistics calculated - mu2 shape: {stat_2['mu'].shape}, sigma2 shape: {stat_2['sigma'].shape}")
        
        # Check for NaN or Inf in statistics
        if np.any(np.isnan(stat_1['sigma'])) or np.any(np.isinf(stat_1['sigma'])):
            fid_logger.error(f"❌ Sigma1 contains NaN or Inf values")
            return {"frechet_distance": float('inf'), "error": "Covariance matrix contains NaN or Inf values"}
            
        if np.any(np.isnan(stat_2['sigma'])) or np.any(np.isinf(stat_2['sigma'])):
            fid_logger.error(f"❌ Sigma2 contains NaN or Inf values")
            return {"frechet_distance": float('inf'), "error": "Covariance matrix contains NaN or Inf values"}
            
    except Exception as e:
        fid_logger.error(f"❌ Error calculating statistics: {e}")
        return {"frechet_distance": float('inf'), "error": f"Statistics calculation failed: {str(e)}"}

    fid_logger.info("🔢 Computing Frechet Distance")

    mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
    mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    fid_logger.debug(f"📊 Final shapes - mu1: {mu1.shape}, mu2: {mu2.shape}, sigma1: {sigma1.shape}, sigma2: {sigma2.shape}")

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    fid_logger.debug(f"📊 Mean difference norm: {np.linalg.norm(diff):.6f}")

    # Product might be almost singular
    try:
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            fid_logger.warning(f"⚠️ FID calculation produces singular product; adding {eps} to diagonal of cov")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    except Exception as e:
        fid_logger.error(f"❌ Error in sqrtm calculation: {e}")
        return {"frechet_distance": float('inf'), "error": f"Matrix square root calculation failed: {str(e)}"}

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            fid_logger.error(f"❌ Imaginary component {m}")
            return {"frechet_distance": float('inf'), "error": f"Imaginary component {m}"}
        covmean = covmean.real
        fid_logger.debug(f"🔄 Converted complex covmean to real")

    tr_covmean = np.trace(covmean)
    fid_logger.debug(f"📊 Trace of covmean: {tr_covmean:.6f}")

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    fid_logger.debug(f"📊 FID components - diff^2: {diff.dot(diff):.6f}, tr(sigma1): {np.trace(sigma1):.6f}, tr(sigma2): {np.trace(sigma2):.6f}")
    fid_logger.info(f"✅ FID calculated: {fid:.6f}")

    return {
        "frechet_distance": float(fid),
    }
