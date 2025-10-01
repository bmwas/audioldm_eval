"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid

VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import logging
import numpy as np
import torch

from torch import nn
from scipy import linalg
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from audioldm_eval.datasets.load_mel import WaveDataset
from torch.utils.data import DataLoader

# Configure logging for FAD metrics
fad_logger = logging.getLogger(__name__)

class FrechetAudioDistance:
    def __init__(
        self, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8
    ):
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()

    def load_audio_data(self, x):
        outputloader = DataLoader(
            WaveDataset(
                x,
                16000,
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=8,
        )
        data_list = []
        print("Loading data to RAM")
        for batch in tqdm(outputloader):
            data_list.append((batch[0][0,0], 16000))
        return data_list

    def get_embeddings(self, x, sr=16000, limit_num=None):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        x = self.load_audio_data(x)
        if isinstance(x, list): 
            try:
                for audio, sr in tqdm(x, disable=(not self.verbose)):
                    embd = self.model.forward(audio.numpy(), sr)
                    if self.model.device == torch.device("cuda"):
                        embd = embd.cpu()
                    embd = embd.detach().numpy()
                    embd_lst.append(embd)
            except Exception as e:
                print(
                    "[Frechet Audio Distance] get_embeddings throw an exception: {}".format(
                        str(e)
                    )
                )
        else:
            raise AttributeError

        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        fad_logger.debug(f"📊 Calculating embedding statistics for {len(embd_lst)} embeddings")
        
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
            fad_logger.debug(f"🔄 Converted list to numpy array, shape: {embd_lst.shape}")
        
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        
        fad_logger.debug(f"📊 Statistics calculated - mu shape: {mu.shape}, sigma shape: {sigma.shape}")
        fad_logger.debug(f"📊 Mean range: [{mu.min():.4f}, {mu.max():.4f}]")
        fad_logger.debug(f"📊 Sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        fad_logger.debug(f"🔢 Calculating Frechet distance with eps={eps}")

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        fad_logger.debug(f"📊 Input shapes - mu1: {mu1.shape}, mu2: {mu2.shape}, sigma1: {sigma1.shape}, sigma2: {sigma2.shape}")

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        fad_logger.debug(f"📊 Mean difference norm: {np.linalg.norm(diff):.6f}")

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            fad_logger.warning(f"⚠️ {msg}")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                fad_logger.error(f"❌ Imaginary component {m}")
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
            fad_logger.debug(f"🔄 Converted complex covmean to real")

        tr_covmean = np.trace(covmean)
        fad_logger.debug(f"📊 Trace of covmean: {tr_covmean:.6f}")

        frechet_dist = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        fad_logger.debug(f"📊 Frechet distance components - diff^2: {diff.dot(diff):.6f}, tr(sigma1): {np.trace(sigma1):.6f}, tr(sigma2): {np.trace(sigma2):.6f}")
        fad_logger.info(f"✅ Frechet distance calculated: {frechet_dist:.6f}")

        return frechet_dist

    def score(self, background_dir, eval_dir, store_embds=False, limit_num=None, recalculate = False): 
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        try:
            fad_target_folder_cache = eval_dir + "_fad_feature_cache.npy"
            fad_generated_folder_cache = background_dir + "_fad_feature_cache.npy"

            if(not os.path.exists(fad_generated_folder_cache) or recalculate):
                embds_background = self.get_embeddings(background_dir, limit_num=limit_num)
                np.save(fad_generated_folder_cache, embds_background)
            else:
                print("Reload fad_generated_folder_cache", fad_generated_folder_cache)
                embds_background = np.load(fad_generated_folder_cache)

            if(not os.path.exists(fad_target_folder_cache) or recalculate):
                embds_eval = self.get_embeddings(eval_dir, limit_num=limit_num)
                np.save(fad_target_folder_cache, embds_eval)
            else:
                print("Reload fad_target_folder_cache", fad_target_folder_cache)
                embds_eval = np.load(fad_target_folder_cache)

            if store_embds:
                np.save("embds_background.npy", embds_background)
                np.save("embds_eval.npy", embds_eval)

            if len(embds_background) == 0:
                print(
                    "[Frechet Audio Distance] background set dir is empty, exitting..."
                )
                return -1

            if len(embds_eval) == 0:
                print("[Frechet Audio Distance] eval set dir is empty, exitting...")
                return -1

            mu_background, sigma_background = self.calculate_embd_statistics(
                embds_background
            )
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background, sigma_background, mu_eval, sigma_eval
            )

            return {"frechet_audio_distance": fad_score}

        except Exception as e:
            print("[Frechet Audio Distance] exception thrown, {}".format(str(e)))
            return -1
