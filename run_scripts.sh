# 1. Clean eval (text, image, feature-fusion, all late-fusion modes)
python3 run_clean.py

# 2. Missing-modality ablation (feature-fusion + late-fusion)
python3 run_ablation.py

# 3. All adversarial attacks (5 attack types × 6 fusion methods)
python3 run_multimodal_attacks.py
