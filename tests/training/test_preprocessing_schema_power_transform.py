from app.services.training.config.schema import (
    NUMERIC_POWER_TRANSFORM_METHODS,
    NUMERIC_SCALING_METHODS,
    PreprocessingConfig,
    _legacy_preprocessing_capabilities,
)


def test_preprocessing_from_front_preserves_modern_numeric_power_transform_default():
    cfg = PreprocessingConfig.from_front(
        {
            "preprocessing": {
                "defaults": {
                    "numericPowerTransform": "yeo_johnson",
                }
            }
        }
    )

    assert cfg.numeric_power_transform == "yeo_johnson"
    assert cfg.numeric_scaling == "none"


def test_preprocessing_from_front_reads_legacy_normalization_default():
    cfg = PreprocessingConfig.from_front(
        {
            "preprocessing": {
                "normalization": {
                    "numeric": "box_cox",
                }
            }
        }
    )

    assert cfg.numeric_power_transform == "box_cox"
    assert cfg.numeric_scaling == "none"


def test_preprocessing_from_front_migrates_legacy_numeric_scaling_power_transform():
    cfg = PreprocessingConfig.from_front(
        {
            "preprocessing": {
                "numericScaling": "yeo_johnson",
            }
        }
    )

    assert cfg.numeric_power_transform == "yeo_johnson"
    assert cfg.numeric_scaling == "none"


def test_preprocessing_from_front_migrates_top_level_legacy_numeric_scaling_power_transform():
    cfg = PreprocessingConfig.from_front(
        {
            "numericScaling": "box_cox",
        }
    )

    assert cfg.numeric_power_transform == "box_cox"
    assert cfg.numeric_scaling == "none"


def test_preprocessing_from_front_keeps_true_numeric_scaling_separate():
    cfg = PreprocessingConfig.from_front(
        {
            "preprocessing": {
                "defaults": {
                    "numericScaling": "robust",
                }
            }
        }
    )

    assert cfg.numeric_scaling == "robust"
    assert cfg.numeric_power_transform == "none"


def test_preprocessing_from_front_invalid_numeric_choices_fall_back_safely():
    cfg = PreprocessingConfig.from_front(
        {
            "preprocessing": {
                "defaults": {
                    "numericScaling": "not_a_method",
                    "numericPowerTransform": "not_a_transform",
                }
            }
        }
    )

    assert cfg.numeric_scaling == "none"
    assert cfg.numeric_power_transform == "none"


def test_legacy_preprocessing_capabilities_publish_power_transform_under_normalization():
    caps = _legacy_preprocessing_capabilities()

    assert caps["scaling"]["numeric"] == NUMERIC_SCALING_METHODS
    assert caps["normalization"]["numeric"] == NUMERIC_POWER_TRANSFORM_METHODS
    assert caps["normalization"]["defaultNumeric"] == "none"
