#!/bin/bash
# Rebuild and install only the idxd and dcbm modules on the test box
# (kernel image unchanged), then reload them.
set -eu
cd ~/linux-dma-page-zero
git checkout -q mm-offload-v72
git log --oneline -1
make -j64 M=drivers/dma/idxd 2>&1 | grep -E "error|warning:" || true; make -j64 M=drivers/mm_offload/dcbm 2>&1 | grep -E 'error|warning:' || true
K=/lib/modules/$(uname -r)
sudo cp drivers/dma/idxd/idxd.ko $K/kernel/drivers/dma/idxd/idxd.ko
sudo cp drivers/mm_offload/dcbm/dcbm.ko $K/kernel/drivers/mm_offload/dcbm/dcbm.ko
sudo depmod -a
sudo rmmod dcbm 2>/dev/null || true
sudo rmmod idxd 2>/dev/null || true
sudo modprobe idxd
sleep 2
sudo ~/mmoffbench/setup_pgzero_dsa.sh "dsa0 dsa2 dsa8 dsa10" | tail -1
sudo modprobe dcbm
modinfo -F srcversion dcbm; modinfo -F srcversion idxd
