@echo off
REM BERT_Bike_order3.bat
REM STH-SepNet with BERT backbone, hypergraph order k=3, BIKE dataset (Windows)

echo ============================================
echo STH-SepNet: BERT + BIKE Inflow + order=3
echo ============================================

python train_sthsepnet.py ^
  --llm_model BERT ^
  --data inflow ^
  --root_path ./dataset/Bike/ ^
  --data_path inflow.csv ^
  --node_num 295 ^
  --scale_hyperedges 3 ^
  --gamma 0.0 ^
  --fusion_gate adaptive ^
  --batch_size 4 ^
  --train_epochs 1 ^
  --learning_rate 0.0001 ^
  --model_comment BERT-Bike-order3

echo.
echo ============================================
echo STH-SepNet: BERT + BIKE Outflow + order=3
echo ============================================

python train_sthsepnet.py ^
  --llm_model BERT ^
  --data outflow ^
  --root_path ./dataset/Bike/ ^
  --data_path outflow.csv ^
  --node_num 295 ^
  --scale_hyperedges 3 ^
  --gamma 0.0 ^
  --fusion_gate adaptive ^
  --batch_size 4 ^
  --train_epochs 1 ^
  --learning_rate 0.0001 ^
  --model_comment BERT-Bike-order3

echo Done.
