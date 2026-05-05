"""Dataset catalog and metadata validation."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

SourceType = Literal["requests"]


N2_CAPPED_DATASETS: set[str] = {"cover", "http", "smtp", "shuttle"}
"""Datasets capped for OCSVM/LOF/DBSCAN protocol."""

ADBENCH_BASE_URL = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical"
)
"""Base URL for ADBench classical tabular datasets."""

GAGOLEWSKI_ARCHIVE_URL = (
    "https://github.com/gagolews/clustering-data-v1/archive/refs/tags/v1.1.0.zip"
)
"""Official archive URL for Gagolewski clustering benchmark datasets."""

ODDS_DATASET_URLS: dict[str, str] = {
    "annthyroid": f"{ADBENCH_BASE_URL}/2_annthyroid.npz",
    "arrhythmia": "https://raw.githubusercontent.com/yzhao062/pyod/master/examples/data/arrhythmia.csv",
    "breastw": f"{ADBENCH_BASE_URL}/4_breastw.npz",
    "cardio": f"{ADBENCH_BASE_URL}/6_cardio.npz",
    "cover": f"{ADBENCH_BASE_URL}/10_cover.npz",
    "glass": f"{ADBENCH_BASE_URL}/14_glass.npz",
    "http": f"{ADBENCH_BASE_URL}/16_http.npz",
    "ionosphere": f"{ADBENCH_BASE_URL}/18_Ionosphere.npz",
    "letter": f"{ADBENCH_BASE_URL}/20_letter.npz",
    "lympho": f"{ADBENCH_BASE_URL}/21_Lymphography.npz",
    "mammography": f"{ADBENCH_BASE_URL}/23_mammography.npz",
    "mnist": f"{ADBENCH_BASE_URL}/24_mnist.npz",
    "musk": f"{ADBENCH_BASE_URL}/25_musk.npz",
    "optdigits": f"{ADBENCH_BASE_URL}/26_optdigits.npz",
    "pendigits": f"{ADBENCH_BASE_URL}/28_pendigits.npz",
    "pima": f"{ADBENCH_BASE_URL}/29_Pima.npz",
    "satellite": f"{ADBENCH_BASE_URL}/30_satellite.npz",
    "satimage-2": f"{ADBENCH_BASE_URL}/31_satimage-2.npz",
    "shuttle": f"{ADBENCH_BASE_URL}/32_shuttle.npz",
    "smtp": f"{ADBENCH_BASE_URL}/34_smtp.npz",
    "speech": f"{ADBENCH_BASE_URL}/36_speech.npz",
    "thyroid": f"{ADBENCH_BASE_URL}/38_thyroid.npz",
    "vertebral": f"{ADBENCH_BASE_URL}/39_vertebral.npz",
    "vowels": f"{ADBENCH_BASE_URL}/40_vowels.npz",
    "wbc": f"{ADBENCH_BASE_URL}/42_WBC.npz",
    "wine": f"{ADBENCH_BASE_URL}/45_wine.npz",
}
"""Dataset-specific source URLs for ODDS-style datasets."""

GAGOLEWSKI_DATASET_IDS: set[str] = {
    "wut/smile",
    "wut/circles",
    "wut/isolation",
    "wut/windows",
    "wut/x2",
    "sipu/compound",
    "sipu/jain",
    "sipu/flame",
    "sipu/spiral",
    "fcps/lsun",
    "graves/fuzzyx",
}
"""Datasets loaded from official clustering-data archive."""


class DatasetSpec(BaseModel):
    """Declarative metadata for one dataset.

    Attributes:
        dataset_id: Unique identifier used in API and file layout.
        source_type: Primary acquisition backend.
        source_ref: Source identifier or URL.
        expected_rows: Approximate dataset rows for sanity checks.
        expected_dim: Approximate feature count for sanity checks.
        contamination: Natural contamination ratio in [0, 1].
        stratified_subsample_for_n2: Whether n^2 models use capped sample.
    """

    dataset_id: str = Field(min_length=1)
    source_type: SourceType
    source_ref: str = Field(min_length=1)
    expected_rows: PositiveInt
    expected_dim: PositiveInt
    contamination: float = Field(ge=0.0, le=1.0)
    stratified_subsample_for_n2: bool = False


def _spec(
    dataset_id: str,
    expected_rows: int,
    expected_dim: int,
    contamination: float,
) -> DatasetSpec:
    """Construct ``DatasetSpec`` rows mirroring mirrored upstream assets.

    Args:
        dataset_id: Public slug used everywhere in loaders and manifests.
        expected_rows: Reference cardinality for QA ratio checks post ingest.
        expected_dim: Nominal numeric feature breadth prior to preprocessing.
        contamination: Annotated anomaly prevalence for benchmarking context.

    Returns:
        Hydrated pydantic specification including download routing metadata.

    Raises:
        ValueError: When neither ODDS nor Gagolewski mirrors know the slug.
    """
    if dataset_id in ODDS_DATASET_URLS:
        source_ref = ODDS_DATASET_URLS[dataset_id]
    elif dataset_id in GAGOLEWSKI_DATASET_IDS:
        source_ref = GAGOLEWSKI_ARCHIVE_URL
    else:
        raise ValueError(f"No source URL configured for dataset_id={dataset_id}")
    return DatasetSpec(
        dataset_id=dataset_id,
        source_type="requests",
        source_ref=source_ref,
        expected_rows=expected_rows,
        expected_dim=expected_dim,
        contamination=contamination,
        stratified_subsample_for_n2=dataset_id in N2_CAPPED_DATASETS,
    )


FULL_DATASET_SPECS: tuple[DatasetSpec, ...] = (
    _spec("annthyroid", 7200, 6, 0.074),
    _spec("arrhythmia", 452, 274, 0.146),
    _spec("breastw", 683, 9, 0.35),
    _spec("cardio", 1831, 21, 0.096),
    _spec("cover", 286048, 10, 0.0096),
    _spec("glass", 214, 7, 0.042),
    _spec("http", 567498, 3, 0.0039),
    _spec("ionosphere", 351, 33, 0.359),
    _spec("letter", 1600, 32, 0.0625),
    _spec("lympho", 148, 18, 0.041),
    _spec("mammography", 11183, 6, 0.023),
    _spec("mnist", 7603, 100, 0.092),
    _spec("musk", 3062, 166, 0.032),
    _spec("optdigits", 5216, 64, 0.029),
    _spec("pendigits", 6870, 16, 0.023),
    _spec("pima", 768, 8, 0.349),
    _spec("satellite", 6435, 36, 0.316),
    _spec("satimage-2", 5803, 36, 0.012),
    _spec("shuttle", 49097, 9, 0.072),
    _spec("smtp", 95156, 3, 0.0003),
    _spec("speech", 3686, 400, 0.0165),
    _spec("thyroid", 3772, 6, 0.025),
    _spec("vertebral", 240, 6, 0.125),
    _spec("vowels", 1456, 12, 0.034),
    _spec("wbc", 223, 9, 0.04484304932735426),
    _spec("wine", 129, 13, 0.078),
    _spec("wut/smile", 1000, 2, 0.0),
    _spec("wut/circles", 4000, 2, 0.0),
    _spec("wut/isolation", 9000, 2, 0.0),
    _spec("wut/windows", 2977, 2, 0.0),
    _spec("wut/x2", 120, 2, 10.0 / 120.0),
    _spec("sipu/compound", 399, 2, 0.0),
    _spec("sipu/jain", 373, 2, 0.0),
    _spec("sipu/flame", 240, 2, 12.0 / 240.0),
    _spec("sipu/spiral", 312, 2, 0.0),
    _spec("fcps/lsun", 400, 2, 0.0),
    _spec("graves/fuzzyx", 1000, 2, 0.0),
)
"""Complete dataset list from plan document."""


@dataclass(frozen=True, slots=True)
class DatasetCatalog:
    """Catalog wrapper exposing validated dataset specs.

    Attributes:
        specs: Immutable tuple of all supported ``DatasetSpec`` instances.
    """

    specs: tuple[DatasetSpec, ...]

    def ids(self) -> list[str]:
        """Return dataset identifiers in manifest declaration order."""
        return [spec.dataset_id for spec in self.specs]

    def get(self, dataset_id: str) -> DatasetSpec:
        """Find spec by dataset id.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            Matching dataset specification.

        Raises:
            KeyError: If dataset id not found.
        """
        for spec in self.specs:
            if spec.dataset_id == dataset_id:
                return spec
        raise KeyError(f"Unknown dataset_id={dataset_id}")


def build_default_catalog() -> DatasetCatalog:
    """Instantiate the frozen production catalog listing every planned dataset.

    Returns:
        ``DatasetCatalog`` wrapping ``FULL_DATASET_SPECS``.
    """
    return DatasetCatalog(specs=FULL_DATASET_SPECS)
