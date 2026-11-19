"""WildFusion feature extraction and calibrated similarity computation.

This module defines ``WildFusionFeatureExtractor``, a small wrapper around the
wildlife-tools WildFusion stack. It builds three similarity pipelines:

1. MiewID global deep features compared with cosine similarity.
2. ALIKED local features matched with LightGlue.
3. DISK local features matched with LightGlue.

The three pipelines are fused by ``WildFusion`` and calibrated with isotonic
calibration before being used to compute a similarity matrix for a dataset.
"""

from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor, DiskExtractor
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.calibration import IsotonicCalibration
from transformers import AutoModel
import torchvision.transforms as T
from src.data.data_transformation import SpeciesTransform
from src.utilities.finetuned_model import (
    load_arcface_finetuned_hf_model,
    load_finetuned_backbone_into_hf_model,
)


class WildFusionFeatureExtractor:
    """Build, calibrate, and run the WildFusion similarity pipeline.

    The class separates object construction from calibration. Instantiating the
    class only stores configuration and base transforms. The heavy work of
    loading models and fitting calibration happens when ``calibrate`` is called.

    Typical usage:
        extractor = WildFusionFeatureExtractor(dataset_calibration)
        extractor.calibrate()
        sim_matrix = extractor.run(ds_name, data)
    """

    def __init__(
        self,
        dataset_calibration=None,
        device='cuda',
        hf_tag="conservationxlabs/miewid-msv3",
        sphereface_version=None,
        arcface_version=None,
    ):
        """Initialize configuration without loading models.

        Args:
            dataset_calibration: Dataset used to fit the calibration layers for
                the individual pipelines and fused WildFusion scores. It can be
                provided here or later through ``calibrate``.
            device: Device used by the deep model and LightGlue matchers, usually
                ``"cuda"`` or ``"cpu"``.
            hf_tag: Hugging Face model id for the MiewID global feature model.
            sphereface_version: Optional path to a SphereFace fine-tuned MiewID
                checkpoint. When provided, its backbone weights replace the
                original Hugging Face MiewID weights.
            arcface_version: Optional path to an ArcFace fine-tuned MiewID
                checkpoint saved as a full Hugging Face model state dict.
        """
        if sphereface_version and arcface_version:
            raise ValueError("Use either sphereface_version or arcface_version, not both.")

        # MiewID expects ImageNet-style normalization after resizing. This
        # transform is used by the global deep-feature pipeline.
        self.base_transform_global = T.Compose([
            T.Resize([440, 440]),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # ALIKED and DISK are local feature extractors. They only need resizing
        # and tensor conversion because keypoint descriptors are not normalized
        # with ImageNet statistics like the global MiewID model.
        self.base_transform_local = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ])

        self.dataset_calibration = dataset_calibration
        self.device = device
        self.hf_tag = hf_tag
        self.sphereface_version = sphereface_version
        self.arcface_version = arcface_version

        # These pipeline attributes stay empty until calibrate() builds the full
        # WildFusion stack. Keeping them as attributes makes the pipeline objects
        # available for later inspection or transform updates.
        self.miewid = None
        self.aliked = None
        self.disk = None
        self.wildfusion = None

        # This is set after calibrate() succeeds. run() checks it so similarity
        # computation cannot accidentally happen with an uncalibrated model.
        self.calibrated_wildfusion = None


    def _load_miew(self):
        """Load the MiewID global feature pipeline.

        MiewID extracts one deep embedding per image. Similarity is computed with
        cosine similarity, then calibrated with isotonic calibration so the score
        scale is compatible with the other WildFusion pipelines.
        """
        print("Loading MiewIDv3")

        # trust_remote_code=True is required because the MiewID repository uses
        # custom Hugging Face model code.
        hf_model = AutoModel.from_pretrained(self.hf_tag, trust_remote_code=True).to(self.device)

        if self.sphereface_version:
            print(f"Loading SphereFace fine-tuned MiewID weights from {self.sphereface_version}")
            hf_model = load_finetuned_backbone_into_hf_model(
                hf_model=hf_model,
                ckpt_path=self.sphereface_version,
                device=self.device,
            )
        elif self.arcface_version:
            print(f"Loading ArcFace fine-tuned MiewID weights from {self.arcface_version}")
            hf_model = load_arcface_finetuned_hf_model(
                hf_model=hf_model,
                ckpt_path=self.arcface_version,
                device=self.device,
            )

        hf_model.eval()

        # SimilarityPipeline combines preprocessing, feature extraction,
        # pairwise similarity, and calibration into a single callable component.
        matcher_miew = SimilarityPipeline(
                            matcher = CosineSimilarity(),
                            extractor = DeepFeatures(model=hf_model, device=self.device, batch_size=16),
                            transform = self.base_transform_global,
                            calibration = IsotonicCalibration()
                        )
        print("MiewIDv3 is Loaded")

        return matcher_miew
    
    def _load_aliked(self):
        """Load the ALIKED local feature pipeline.

        ALIKED detects and describes local keypoints. LightGlue then matches
        those keypoints between image pairs and converts the matches into a
        similarity score that can be calibrated.
        """
        print("Loading Aliked")

        # MatchLightGlue wraps LightGlue matching for the chosen local feature
        # type. The extractor computes ALIKED keypoints/descriptors.
        matcher_aliked = SimilarityPipeline(
            matcher = MatchLightGlue(features='aliked', device=self.device, batch_size=16),
            extractor = AlikedExtractor(),
            transform = self.base_transform_local,
            calibration = IsotonicCalibration()
        )
        print("Aliked is Loaded")
        return matcher_aliked
    
    def _load_disk(self):
        """Load the DISK local feature pipeline.

        DISK is another local keypoint/descriptor method. Keeping both ALIKED and
        DISK gives WildFusion multiple local matching signals, which can improve
        robustness when animal pose, crop.
        """
        print("Loading Disk")

        # init_threshold controls LightGlue initialization/filtering behavior for
        # DISK matches. The resulting raw scores are calibrated later.
        matcher_disk = SimilarityPipeline(
            matcher = MatchLightGlue(features='disk', device=self.device, batch_size=16, init_threshold = 0.9),
            extractor = DiskExtractor(),
            transform = self.base_transform_local,
            calibration = IsotonicCalibration()
        )
        print("Disk is Loaded")
        return matcher_disk
    
    def _wildfusion_setup(self):
        """Build the uncalibrated WildFusion object from all component pipelines.

        Returns:
            A ``WildFusion`` instance containing ALIKED, MiewID, and DISK
            pipelines. Calibration is not fitted here; that is done by
            ``calibrate``.
        """
        # Load each component pipeline only when calibration starts. This keeps
        # __init__ lightweight and avoids expensive model loading during object
        # construction.
        self.miewid = self._load_miew()
        self.aliked = self._load_aliked()
        self.disk = self._load_disk()

        # MiewID is used as the priority pipeline because it provides the global
        # appearance score that WildFusion can fall back to or emphasize when
        # combining the calibrated component similarities.
        wildfusion = WildFusion(calibrated_pipelines=[self.aliked, self.miewid, self.disk], priority_pipeline=self.miewid)
        print("WildFusion is ready.")
        return wildfusion
    

    def calibrate(self, dataset_calibration=None):
        """Fit calibration for the component pipelines and fused WildFusion output.

        Args:
            dataset_calibration: Optional calibration dataset. When supplied, it
                replaces the dataset passed to ``__init__``.

        Returns:
            The calibrated ``WildFusion`` object, also stored in
            ``self.calibrated_wildfusion``.

        Raises:
            ValueError: If no calibration dataset was provided either during
                construction or in this method call.
        """
        # Allow the caller to provide calibration data either during object
        # construction or at the moment calibration is requested.
        if dataset_calibration is not None:
            self.dataset_calibration = dataset_calibration

        # Calibration cannot run without known data because fit_calibration needs
        # image pairs/labels from the dataset to learn score mappings.
        if self.dataset_calibration is None:
            raise ValueError("dataset_calibration must be provided before calibration.")

        print("Calibrating WildFusion.")

        # Build the three pipelines and wrap them in WildFusion.
        self.wildfusion = self._wildfusion_setup()

        # Fit isotonic calibration for the individual pipelines and the fused
        # WildFusion score distribution. The same calibration dataset is used as
        # both query and gallery here.
        self.wildfusion.fit_calibration(self.dataset_calibration, self.dataset_calibration)

        # Store the fitted object so run() can use the calibrated version instead
        # of a raw, uncalibrated WildFusion instance.
        self.calibrated_wildfusion = self.wildfusion
        print("Calibration Complete.")
        return self.calibrated_wildfusion

    
    def run(self, ds_name, data, B = 256):
        """Compute a calibrated WildFusion similarity matrix for a dataset.

        Args:
            ds_name: Dataset name used by ``SpeciesTransform`` to choose the
                species-specific preprocessing routine.
            data: Dataset/gallery tuple passed to WildFusion as both query and gallery.
            B: Batch size/chunk size forwarded to WildFusion similarity
                computation.

        Returns:
            Pairwise similarity matrix for ``data`` against itself.

        Raises:
            RuntimeError: If ``calibrate`` has not been called yet.
        """
        # Prevent accidental inference with uncalibrated scores. The class is
        # intentionally two-step: construct first, calibrate second, run third.
        if self.calibrated_wildfusion is None:
            raise RuntimeError("WildFusion must be calibrated before calling run(). Call calibrate(...) first.")

        # Swap in species-aware preprocessing before feature extraction. ALIKED
        # and DISK use local transforms, while MiewID uses the global normalized
        # transform expected by the deep model.
        self.calibrated_wildfusion.calibrated_pipelines[0].transform = SpeciesTransform(ds_name, self.base_transform_local) # Aliked
        self.calibrated_wildfusion.calibrated_pipelines[1].transform = SpeciesTransform(ds_name, self.base_transform_global) # MiewID

        print("Computing similarities...")

        # Passing data twice computes all pairwise similarities inside the same
        # dataset, which is the matrix used by later clustering/submission steps.
        sim_matrix = self.calibrated_wildfusion(data[0], data[1], B=B)

        return sim_matrix
