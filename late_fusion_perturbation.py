import os

# Custom imports
from utils import create_late_fusion, create_late_fusion_parameters
from paths import (
    RESULT_PATH, PERT_BASE,
    CLEAN_TEXT_CSV, CLEAN_IMAGE_CSV, PER_TEXT_CSV, PER_IMAGE_CSV,
    CLEAN_TEXT_PARAMS, CLEAN_IMAGE_PARAMS, PER_TEXT_PARAMS, PER_IMAGE_PARAMS,
)


if __name__ == "__main__":
    # Clean late-fusion: combine clean text + clean image scores
    for fusion in ["mean", "min", "max"]:
        clean_out_dir = os.path.join(RESULT_PATH, "clean", "late-fusion", fusion)
        create_late_fusion(CLEAN_TEXT_CSV, CLEAN_IMAGE_CSV, clean_out_dir, fusion, filename="results.csv")
        create_late_fusion_parameters(
            text_parameters=CLEAN_TEXT_PARAMS,
            image_parameters=CLEAN_IMAGE_PARAMS,
            output_dir=clean_out_dir,
            fusion_type=fusion,
            scenario="clean",
            text_state="clean",
            image_state="clean",
            text_csv=CLEAN_TEXT_CSV,
            image_csv=CLEAN_IMAGE_CSV,
        )

    scenarios = {
        # Testo perturbato + immagine perturbata
        "both-perturbed": {
            "text_csv": PER_TEXT_CSV,
            "image_csv": PER_IMAGE_CSV,
            "text_params": PER_TEXT_PARAMS,
            "image_params": PER_IMAGE_PARAMS,
            "text_state": "perturbed",
            "image_state": "perturbed",
            "subdir": None,
        },

        # Testo clean + immagine perturbata
        "image-perturbed": {
            "text_csv": CLEAN_TEXT_CSV,
            "image_csv": PER_IMAGE_CSV,
            "text_params": CLEAN_TEXT_PARAMS,
            "image_params": PER_IMAGE_PARAMS,
            "text_state": "clean",
            "image_state": "perturbed",
            "subdir": "image-perturbed",
        },

        # Testo perturbato + immagine clean
        "text-perturbed": {
            "text_csv": PER_TEXT_CSV,
            "image_csv": CLEAN_IMAGE_CSV,
            "text_params": PER_TEXT_PARAMS,
            "image_params": CLEAN_IMAGE_PARAMS,
            "text_state": "perturbed",
            "image_state": "clean",
            "subdir": "text-perturbed",
        },
    }

    for fusion in ["mean", "min", "max"]:
        fusion_base_dir = os.path.join(PERT_BASE, "late-fusion", fusion)

        for scenario_name, cfg in scenarios.items():
            if cfg["subdir"] is None:
                out_dir = fusion_base_dir
            else:
                out_dir = os.path.join(fusion_base_dir, cfg["subdir"])

            create_late_fusion(cfg["text_csv"], cfg["image_csv"], out_dir, fusion)
            create_late_fusion_parameters(
                text_parameters=cfg["text_params"],
                image_parameters=cfg["image_params"],
                output_dir=out_dir,
                fusion_type=fusion,
                scenario=scenario_name,
                text_state=cfg["text_state"],
                image_state=cfg["image_state"],
                text_csv=cfg["text_csv"],
                image_csv=cfg["image_csv"],
            )