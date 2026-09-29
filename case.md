export CASE_DIR=/home/y00938753/ops-transformer/attention/quant_flash_attn/tests/assets

python3 $CASE_DIR/excel_to_csv.py --sheet 

python3 -m ttk list -i $CASE_DIR/qfa_mxfp8_excel.csv

python3 -m ttk e2e -i $CASE_DIR/qfa_mxfp8_excel.csv --plugin $CASE_DIR --warmup false --run 1 --pc 1 -t 