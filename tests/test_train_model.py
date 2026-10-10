import subprocess
import pytest

@pytest.mark.parametrize(
    "C,l1_ratio,class_weight,random_state",
    [
        (0.5, 0.5, "none", 17),
        (1.0, 0.5, "none", 42),
        (2.0, 0.8, "none", 123),
        (0.5, 0.8, "balanced", 42),
        (2.0, 0.2, "balanced", 123),
    ]
)
def test_train_with_different_hyperparameters(
    C, l1_ratio, class_weight, random_state
):
    result = subprocess.run(
        [
            "make",
            "train",
            f"C={C}",
            f"L1_RATIO={l1_ratio}",
            f"CLASS_WEIGHT={class_weight}",
            f"RANDOM_STATE={random_state}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0