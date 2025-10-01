import os
import logging
from audioldm_eval.datasets.load_mel import load_npy_data, MelPairedDataset, WaveDataset
import numpy as np
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, AutoModel, Wav2Vec2FeatureExtractor
from audioldm_eval.metrics.fad import FrechetAudioDistance
from audioldm_eval import calculate_fid, calculate_isc, calculate_kid, calculate_kl
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from audioldm_eval.feature_extractors.panns import Cnn14
from audioldm_eval.audio.tools import save_pickle, load_pickle, write_json, load_json
from ssr_eval.metrics import AudioMetrics
import audioldm_eval.audio as Audio

# Configure logging for evaluation module
eval_logger = logging.getLogger(__name__)

class EvaluationHelper:
    def __init__(self, sampling_rate, device, backbone="mert") -> None:

        self.device = device
        self.backbone = backbone
        self.sampling_rate = sampling_rate
        self.frechet = FrechetAudioDistance(
            use_pca=False,
            use_activation=False,
            verbose=True,
        )
        
        # self.passt_model = get_basic_model(mode="logits")
        # self.passt_model.eval()
        # self.passt_model.to(self.device)

        # self.lsd_metric = AudioMetrics(self.sampling_rate)
        self.frechet.model = self.frechet.model.to(device)

        features_list = ["2048", "logits"]
        
        if self.backbone == "mert":
            self.mel_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
            self.target_sample_rate = self.processor.sampling_rate
            self.resampler = T.Resample(orig_freq=self.sampling_rate, new_freq=self.target_sample_rate).to(self.device)
        elif self.backbone == "cnn14":
            if self.sampling_rate == 16000:
                self.mel_model = Cnn14(
                    features_list=features_list,
                    sample_rate=16000,
                    window_size=512,
                    hop_size=160,
                    mel_bins=64,
                    fmin=50,
                    fmax=8000,
                    classes_num=527,
                )
            elif self.sampling_rate == 32000:
                self.mel_model = Cnn14(
                    features_list=features_list,
                    sample_rate=32000,
                    window_size=1024,
                    hop_size=320,
                    mel_bins=64,
                    fmin=50,
                    fmax=14000,
                    classes_num=527,
                )
            else:
                raise ValueError(
                    "We only support the evaluation on 16kHz and 32kHz sampling rate for CNN14."
                )
        else:
            raise ValueError("Backbone not supported")

        if self.sampling_rate == 16000:
            self._stft = Audio.TacotronSTFT(512, 160, 512, 64, 16000, 50, 8000)
        elif self.sampling_rate == 32000:
            self._stft = Audio.TacotronSTFT(1024, 320, 1024, 64, 32000, 50, 14000)
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )

        self.mel_model.eval()
        self.mel_model.to(self.device)
        self.fbin_mean, self.fbin_std = None, None

    def main(
        self,
        generate_files_path,
        groundtruth_path,
        limit_num=None,
    ):
        eval_logger.info(f"🚀 Starting evaluation with backbone: {self.backbone}, sampling_rate: {self.sampling_rate}")
        eval_logger.info(f"📁 Generated files path: {generate_files_path}")
        eval_logger.info(f"📁 Target files path: {groundtruth_path}")
        eval_logger.info(f"🔢 Limit num: {limit_num}")

        # File initialization checks with detailed logging
        eval_logger.debug(f"🔍 Performing file initialization check for generated files...")
        self.file_init_check(generate_files_path)
        eval_logger.debug(f"✅ Generated files directory check passed")
        
        eval_logger.debug(f"🔍 Performing file initialization check for target files...")
        self.file_init_check(groundtruth_path)
        eval_logger.debug(f"✅ Target files directory check passed")

        # Get filename intersection ratio with detailed logging
        eval_logger.debug(f"🔍 Calculating filename intersection ratio...")
        same_name = self.get_filename_intersection_ratio(
            generate_files_path, groundtruth_path, limit_num=limit_num
        )
        eval_logger.info(f"📊 Filename intersection analysis complete - same_name: {same_name}")

        # Calculate metrics with detailed logging
        eval_logger.debug(f"🔄 Starting metrics calculation...")
        # Enable PSNR, SSIM, and LSD calculations for comprehensive evaluation
        metrics = self.calculate_metrics(
            generate_files_path, 
            groundtruth_path, 
            same_name, 
            limit_num,
            calculate_psnr_ssim=True,  # Enable PSNR and SSIM
            calculate_lsd=True         # Enable LSD
        )
        eval_logger.info(f"✅ Metrics calculation completed successfully")
        eval_logger.debug(f"📊 Calculated metrics: {list(metrics.keys()) if metrics else 'None'}")

        return metrics

    def file_init_check(self, dir):
        eval_logger.debug(f"🔍 Checking directory: {dir}")
        
        # Check if directory exists
        if not os.path.exists(dir):
            eval_logger.error(f"❌ Directory does not exist: {dir}")
            raise AssertionError(f"The path does not exist {dir}")
        eval_logger.debug(f"✅ Directory exists: {dir}")
        
        # List files in directory
        files = os.listdir(dir)
        eval_logger.debug(f"📁 Files in directory {dir}: {files}")
        eval_logger.debug(f"📊 File count: {len(files)}")
        
        if len(files) == 0:
            eval_logger.error(f"❌ No files found in directory: {dir}")
            raise AssertionError(f"There is no files in {dir}")
        
        eval_logger.debug(f"✅ Directory check passed: {len(files)} files found")

    def get_filename_intersection_ratio(
        self, dir1, dir2, threshold=0.99, limit_num=None
    ):
        eval_logger.debug(f"🔍 Analyzing filename intersection between {dir1} and {dir2}")
        eval_logger.debug(f"📊 Threshold: {threshold}, Limit: {limit_num}")
        
        # Get file lists
        self.datalist1 = [os.path.join(dir1, x) for x in os.listdir(dir1)]
        self.datalist1 = sorted(self.datalist1)
        eval_logger.debug(f"📁 Directory 1 files ({len(self.datalist1)}): {[os.path.basename(x) for x in self.datalist1]}")

        self.datalist2 = [os.path.join(dir2, x) for x in os.listdir(dir2)]
        self.datalist2 = sorted(self.datalist2)
        eval_logger.debug(f"📁 Directory 2 files ({len(self.datalist2)}): {[os.path.basename(x) for x in self.datalist2]}")

        # Create filename dictionaries
        data_dict1 = {os.path.basename(x): x for x in self.datalist1}
        data_dict2 = {os.path.basename(x): x for x in self.datalist2}

        keyset1 = set(data_dict1.keys())
        keyset2 = set(data_dict2.keys())
        
        eval_logger.debug(f"🔑 Filenames in dir1: {sorted(keyset1)}")
        eval_logger.debug(f"🔑 Filenames in dir2: {sorted(keyset2)}")

        # Calculate intersection
        intersect_keys = keyset1.intersection(keyset2)
        eval_logger.debug(f"🔗 Intersecting filenames: {sorted(intersect_keys)}")
        
        # Calculate ratios
        ratio1 = len(intersect_keys) / len(keyset1) if len(keyset1) > 0 else 0
        ratio2 = len(intersect_keys) / len(keyset2) if len(keyset2) > 0 else 0
        
        eval_logger.debug(f"📊 Intersection ratios: {ratio1:.3f} (dir1), {ratio2:.3f} (dir2)")
        eval_logger.debug(f"📊 Threshold check: {ratio1 > threshold} and {ratio2 > threshold}")
        
        if ratio1 > threshold and ratio2 > threshold:
            eval_logger.info(f"✅ Paired mode detected: {len(intersect_keys)}/{len(keyset1)} and {len(intersect_keys)}/{len(keyset2)} files match")
            return True
        else:
            eval_logger.info(f"📊 Unpaired mode detected: {len(intersect_keys)}/{len(keyset1)} and {len(intersect_keys)}/{len(keyset2)} files match")
            return False

    def calculate_lsd(self, pairedloader, same_name=True, time_offset=160 * 7):
        if same_name == False:
            return {
                "lsd": -1,
                "ssim_stft": -1,
            }
        print("Calculating LSD using a time offset of %s ..." % time_offset)
        lsd_avg = []
        ssim_stft_avg = []
        for _, _, filename, (audio1, audio2) in tqdm(pairedloader):
            audio1 = audio1.cpu().numpy()[0, 0]
            audio2 = audio2.cpu().numpy()[0, 0]

            # If you use HIFIGAN (verified on 2023-01-12), you need seven frames' offset
            audio1 = audio1[time_offset:]

            audio1 = audio1 - np.mean(audio1)
            audio2 = audio2 - np.mean(audio2)

            audio1 = audio1 / np.max(np.abs(audio1))
            audio2 = audio2 / np.max(np.abs(audio2))

            min_len = min(audio1.shape[0], audio2.shape[0])

            audio1, audio2 = audio1[:min_len], audio2[:min_len]

            result = self.lsd(audio1, audio2)

            lsd_avg.append(result["lsd"])
            ssim_stft_avg.append(result["ssim"])

        return {"lsd": np.mean(lsd_avg), "ssim_stft": np.mean(ssim_stft_avg)}

    def lsd(self, audio1, audio2):
        result = self.lsd_metric.evaluation(audio1, audio2, None)
        return result

    def calculate_psnr_ssim(self, pairedloader, same_name=True):
        if same_name == False:
            return {"psnr": -1, "ssim": -1}
        psnr_avg = []
        ssim_avg = []
        for mel_gen, mel_target, filename, _ in tqdm(pairedloader):
            mel_gen = mel_gen.cpu().numpy()[0]
            mel_target = mel_target.cpu().numpy()[0]
            psnrval = psnr(mel_gen, mel_target)
            if np.isinf(psnrval):
                print("Infinite value encountered in psnr %s " % filename)
                continue
            psnr_avg.append(psnrval)
            data_range = max(np.max(mel_gen), np.max(mel_target)) - min(np.min(mel_gen), np.min(mel_target))
            ssim_avg.append(ssim(mel_gen, mel_target, data_range=data_range))
        return {"psnr": np.mean(psnr_avg), "ssim": np.mean(ssim_avg)}

    def calculate_metrics(self, generate_files_path, groundtruth_path, same_name, limit_num=None, calculate_psnr_ssim=False, calculate_lsd=False, recalculate=False):
        # Generation, target
        torch.manual_seed(0)

        num_workers = 6

        outputloader = DataLoader(
            WaveDataset(
                generate_files_path,
                self.sampling_rate, # TODO
                # 32000,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        resultloader = DataLoader(
            WaveDataset(
                groundtruth_path,
                self.sampling_rate, # TODO
                # 32000,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        out = {}

        # FAD
        ######################################################################################################################
        if(recalculate): 
            print("Calculate FAD score from scratch")
        fad_score = self.frechet.score(generate_files_path, groundtruth_path, limit_num=limit_num, recalculate=recalculate)
        out.update(fad_score)
        print("FAD: %s" % fad_score)
        ######################################################################################################################
        
        # PANNs or PassT
        ######################################################################################################################
        cache_path = groundtruth_path + "classifier_logits_feature_cache.pkl"
        if(os.path.exists(cache_path) and not recalculate):
            print("reload", cache_path)
            featuresdict_2 = load_pickle(cache_path)
        else:
            print("Extracting features from %s." % groundtruth_path)
            featuresdict_2 = self.get_featuresdict(resultloader)
            save_pickle(featuresdict_2, cache_path)
        
        cache_path = generate_files_path + "classifier_logits_feature_cache.pkl"
        if(os.path.exists(cache_path) and not recalculate):
            print("reload", cache_path)
            featuresdict_1 = load_pickle(cache_path)
        else:
            print("Extracting features from %s." % generate_files_path)
            featuresdict_1 = self.get_featuresdict(outputloader)
            save_pickle(featuresdict_1, cache_path)

        metric_kl, kl_ref, paths_1 = calculate_kl(
            featuresdict_1, featuresdict_2, "logits", same_name
        )
        
        out.update(metric_kl)

        # Check minimum sample requirements for ISC
        features_1_logits = featuresdict_1["logits"]
        eval_logger.debug(f"📊 Feature shape for ISC: {features_1_logits.shape}")
        
        if features_1_logits.shape[0] < 2:
            eval_logger.warning(f"⚠️ Insufficient samples for ISC calculation: {features_1_logits.shape[0]} samples (need ≥2)")
            out.update({"inception_score": float('inf'), "isc_error": "Insufficient samples for ISC calculation (need ≥2)"})
        else:
            eval_logger.debug(f"✅ Sufficient samples for ISC calculation, proceeding...")
            metric_isc = calculate_isc(
                featuresdict_1,
                feat_layer_name="logits",
                splits=10,
                samples_shuffle=True,
                rng_seed=2020,
            )
            out.update(metric_isc)

        if("2048" in featuresdict_1.keys() and "2048" in featuresdict_2.keys()):
            eval_logger.debug(f"🔍 Checking FID calculation requirements...")
            features_1_2048 = featuresdict_1["2048"]
            features_2_2048 = featuresdict_2["2048"]
            
            eval_logger.debug(f"📊 Feature shapes for FID: {features_1_2048.shape}, {features_2_2048.shape}")
            
            # Check minimum sample requirements for FID
            if features_1_2048.shape[0] < 2 or features_2_2048.shape[0] < 2:
                eval_logger.warning(f"⚠️ Insufficient samples for FID calculation: {features_1_2048.shape[0]} and {features_2_2048.shape[0]} samples (need ≥2 each)")
                out.update({"frechet_distance": float('inf'), "fid_error": "Insufficient samples for FID calculation (need ≥2 each)"})
            else:
                eval_logger.debug(f"✅ Sufficient samples for FID calculation, proceeding...")
                metric_fid = calculate_fid(
                    featuresdict_1, featuresdict_2, feat_layer_name="2048"
                )
                out.update(metric_fid)

        # Metrics for Autoencoder
        ######################################################################################################################
        if(calculate_psnr_ssim or calculate_lsd):
            pairedloader = DataLoader(
                MelPairedDataset(
                    generate_files_path,
                    groundtruth_path,
                    self._stft,
                    self.sampling_rate,
                    self.fbin_mean,
                    self.fbin_std,
                    limit_num=limit_num,
                ),
                batch_size=1,
                sampler=None,
                num_workers=16,
            )
            
        if(calculate_lsd):
            metric_lsd = self.calculate_lsd(pairedloader, same_name=same_name)
            out.update(metric_lsd)

        if(calculate_psnr_ssim):
            metric_psnr_ssim = self.calculate_psnr_ssim(pairedloader, same_name=same_name)
            out.update(metric_psnr_ssim)

        # KID calculation with validation
        if("2048" in featuresdict_1.keys() and "2048" in featuresdict_2.keys()):
            eval_logger.debug(f"🔍 Checking KID calculation requirements...")
            features_1_2048 = featuresdict_1["2048"]
            features_2_2048 = featuresdict_2["2048"]
            
            eval_logger.debug(f"📊 Feature shapes for KID: {features_1_2048.shape}, {features_2_2048.shape}")
            
            # Check minimum sample requirements for KID (needs more samples than FID)
            min_samples_kid = 10  # KID typically needs more samples
            if features_1_2048.shape[0] < min_samples_kid or features_2_2048.shape[0] < min_samples_kid:
                eval_logger.warning(f"⚠️ Insufficient samples for KID calculation: {features_1_2048.shape[0]} and {features_2_2048.shape[0]} samples (need ≥{min_samples_kid} each)")
                out.update({"kernel_inception_distance": float('inf'), "kid_error": f"Insufficient samples for KID calculation (need ≥{min_samples_kid} each)"})
            else:
                eval_logger.debug(f"✅ Sufficient samples for KID calculation, proceeding...")
                # metric_kid = calculate_kid(
                #     featuresdict_1,
                #     featuresdict_2,
                #     feat_layer_name="2048",
                #     subsets=100,
                #     subset_size=1000,
                #     degree=3,
                #     gamma=None,
                #     coef0=1,
                #     rng_seed=2020,
                # )
                # out.update(metric_kid)
                eval_logger.info(f"📊 KID calculation would proceed with {features_1_2048.shape[0]} and {features_2_2048.shape[0]} samples")

        # Print metrics with proper formatting for different data types
        for k, v in out.items():
            if isinstance(v, (int, float)):
                if np.isinf(v) or np.isnan(v):
                    print(f"{k}: {v}")
                else:
                    print(f"{k}: {v:.7f}")
            else:
                print(f"{k}: {v}")
        print("\n")
        print(limit_num)
        print(
            f'KL_Sigmoid: {out.get("kullback_leibler_divergence_sigmoid", float("nan")):8.5f};',
            f'KL: {out.get("kullback_leibler_divergence_softmax", float("nan")):8.5f};',
            f'PSNR: {out.get("psnr", float("nan")):.5f}',
            f'SSIM: {out.get("ssim", float("nan")):.5f}',
            f'ISc: {out.get("inception_score_mean", float("nan")):8.5f} ({out.get("inception_score_std", float("nan")):5f});',
            f'KID: {out.get("kernel_inception_distance_mean", float("nan")):.5f}',
            f'({out.get("kernel_inception_distance_std", float("nan")):.5f})',
            f'FD: {out.get("frechet_distance", float("nan")):8.5f};',
            f'FAD: {out.get("frechet_audio_distance", float("nan")):.5f}',
            f'LSD: {out.get("lsd", float("nan")):.5f}',
            # f'SSIM_STFT: {out.get("ssim_stft", float("nan")):.5f}',
        )
        result = {
            "frechet_distance": out.get("frechet_distance", float("nan")),
            "frechet_audio_distance": out.get("frechet_audio_distance", float("nan")),
            "kullback_leibler_divergence_sigmoid": out.get(
                "kullback_leibler_divergence_sigmoid", float("nan")
            ),
            "kullback_leibler_divergence_softmax": out.get(
                "kullback_leibler_divergence_softmax", float("nan")
            ),
            "lsd": out.get("lsd", float("nan")),
            "psnr": out.get("psnr", float("nan")),
            "ssim": out.get("ssim", float("nan")),
            # "ssim_stft": out.get("ssim_stft", float("nan")),
            "inception_score_mean": out.get("inception_score_mean", float("nan")),
            "inception_score_std": out.get("inception_score_std", float("nan")),
            "kernel_inception_distance_mean": out.get(
                "kernel_inception_distance_mean", float("nan")
            ),
            "kernel_inception_distance_std": out.get(
                "kernel_inception_distance_std", float("nan")
            ),
        }

        json_path = os.path.join(os.path.dirname(generate_files_path), self.get_current_time()+"_"+os.path.basename(generate_files_path) + ".json")
        write_json(result, json_path)
        return result

    def get_current_time(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d-%H:%M:%S")

    def get_featuresdict(self, dataloader):
        out = None
        out_meta = None

        # transforms=StandardNormalizeAudio()
        for waveform, filename in tqdm(dataloader):
            try:
                metadict = {
                    "file_path_": filename,
                }
                waveform = waveform.squeeze(1)

                # batch = transforms(batch)
                waveform = waveform.float().to(self.device)

                # featuresdict = {}
                # with torch.no_grad():
                #     if(waveform.size(-1) >= 320000):
                #         waveform = waveform[...,:320000]
                #     else:
                #         waveform = torch.nn.functional.pad(waveform, (0,320000-waveform.size(-1)))
                #     featuresdict["logits"] = self.passt_model(waveform)

                with torch.no_grad():
                    if self.backbone == "mert":
                        waveform = self.resampler(waveform[0])
                        mert_input = self.processor(waveform, sampling_rate=self.target_sample_rate, return_tensors="pt").to(self.device)
                        mert_output = self.mel_model(**mert_input, output_hidden_states=True)
                        time_reduced_hidden_states = torch.stack(mert_output.hidden_states).squeeze().mean(dim=1)
                        featuresdict = {"2048": time_reduced_hidden_states.cpu(), "logits": time_reduced_hidden_states.cpu()}
                    elif self.backbone == "cnn14":
                        featuresdict = self.mel_model(waveform)

                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

                if out is None:
                    out = featuresdict
                else:
                    out = {k: out[k] + featuresdict[k] for k in out.keys()}

                if out_meta is None:
                    out_meta = metadict
                else:
                    out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}
            except Exception as e:
                import ipdb

                ipdb.set_trace()
                print("Classifier Inference error: ", e)
                continue

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}

    def sample_from(self, samples, number_to_use):
        assert samples.shape[0] >= number_to_use
        rand_order = np.random.permutation(samples.shape[0])
        return samples[rand_order[: samples.shape[0]], :]


if __name__ == "__main__":
    import yaml
    import argparse
    from audioldm_eval import EvaluationHelper
    import torch

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--generation_result_path",
        type=str,
        required=False,
        help="Audio sampling rate during evaluation",
        default="/mnt/fast/datasets/audio/audioset/2million_audioset_wav/balanced_train_segments",
    )

    parser.add_argument(
        "-t",
        "--target_audio_path",
        type=str,
        required=False,
        help="Audio sampling rate during evaluation",
        default="/mnt/fast/datasets/audio/audioset/2million_audioset_wav/eval_segments",
    )

    parser.add_argument(
        "-sr",
        "--sampling_rate",
        type=int,
        required=False,
        help="Audio sampling rate during evaluation",
        default=16000,
    )

    parser.add_argument(
        "-l",
        "--limit_num",
        type=int,
        required=False,
        help="Audio clip numbers limit for evaluation",
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(f"cuda:{0}")

    evaluator = EvaluationHelper(args.sampling_rate, device)

    metrics = evaluator.main(
        args.generation_result_path,
        args.target_audio_path,
        limit_num=args.limit_num,
        same_name=args.same_name,
    )

    print(metrics)
