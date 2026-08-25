#!/bin/bash
# run_mmoff.sh - A/B validation of dcbm (mm_offload) on DSA.
#
#   run_mmoff.sh zero    THP fault zeroing, CPU vs DMA (thp_zero_bench)
#   run_mmoff.sh mig     move_pages migration, CPU vs DMA (migbench)
#   run_mmoff.sh all
#
# Expects DSA kernel WQs configured (setup_pgzero_dsa.sh) and the dcbm
# module loadable. Results are appended to $OUT (CSV).
set -u
P=/sys/module/dcbm/parameters
OUT=${OUT:-$HOME/mmoffbench/results.csv}
REPS=${REPS:-3}
NR_CHAN=${NR_CHAN:-16}
cd ~/mmoffbench

dcbm_on()  { sudo modprobe dcbm 2>/dev/null; echo $NR_CHAN | sudo tee $P/nr_dma_chan >/dev/null; echo 1 | sudo tee $P/offloading >/dev/null; }
dcbm_off() { [ -e $P/offloading ] && echo 0 | sudo tee $P/offloading >/dev/null; }
counters() { for f in folios_migrated folios_failures batches_refused folios_cleared clear_failures folios_gated; do [ -e $P/$f ] && printf '%s=%s ' $f $(cat $P/$f); done; echo; }
reset_counters() { for f in folios_migrated folios_failures batches_refused folios_cleared clear_failures folios_gated; do [ -e $P/$f ] && echo 0 | sudo tee $P/$f >/dev/null; done; }

log() { echo "$(date +%T) $*"; echo "$*" >> $OUT; }

zero_cell() { # mode threads mb
	local mode=$1 t=$2 mb=$3 r
	for r in $(seq $REPS); do
		reset_counters
		local out
		out=$(taskset -c 0-85 ./thp_zero_bench -t $t -s $mb -v 2>&1 | tr '\n' ' ')
		log "zero,$mode,t=$t,mb=$mb,rep=$r,$out,$(counters)"
	done
}

mig_cell() { # mode threads mb flags...
	local mode=$1 t=$2 mb=$3; shift 3
	local r out
	for r in $(seq $REPS); do
		reset_counters
		out=$(./migbench -s $mb -f 0 -t 1 -T $t -i 2 -v "$@" 2>&1 | grep '^iter' | tr '\n' ' ')
		log "mig,$mode,t=$t,mb=$mb,flags=$*,rep=$r,$out,$(counters)"
	done
}

run_zero() {
	dcbm_off; log "# zero CPU $(uname -r)"
	for t in 1 4 16; do zero_cell cpu $t 4096; done
	dcbm_on; log "# zero DMA nr_chan=$NR_CHAN"
	for t in 1 4 16; do zero_cell dma $t 4096; done
	dcbm_off
}

run_mig() {
	dcbm_off; log "# mig CPU $(uname -r)"
	for t in 1 4 16; do mig_cell cpu $t 4096 -H; done
	for t in 1 4 16; do mig_cell cpu $t 1024; done
	dcbm_on; log "# mig DMA nr_chan=$NR_CHAN"
	for t in 1 4 16; do mig_cell dma $t 4096 -H; done
	for t in 1 4 16; do mig_cell dma $t 1024; done
	dcbm_off
}

case ${1:-all} in
zero) run_zero ;;
mig) run_mig ;;
all) run_zero; run_mig ;;
esac
