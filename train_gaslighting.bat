@echo off
python scripts/train_gaslighting_detector.py --data_path data/processed/gaslighting/gaslighting_dataset.json --output_path models/gaslighting_detector --device cpu --epochs 5 --batch_size 8 --learning_rate 2e-5 --simplified
pause 