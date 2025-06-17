#!/bin/bash

e_types=("esm1b" "esm2_650M")
#e_types=("esm2_650M")
for e_type in "${e_types[@]}"; do
	python 03_encode_esm.py "$e_type"
	echo ""$e_type" done."
done
