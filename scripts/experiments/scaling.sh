#!/bin/zsh
# Square boxes centred on (40.85,-73.5); sides in km; modes all/nojet. Runs sequentially per size, 2 modes in parallel.
cd "$(dirname "$0")/../.."
box() { .venv/bin/python -c "
import math,sys
s=float(sys.argv[1])/2; la=s/111.32; lo=s/(111.32*math.cos(math.radians(40.85)))
print(40.85-la,40.85+la,-73.5-lo,-73.5+lo)" $1; }
for side in 6 12 24; do
  b=($(box $side))
  for mode in all nojet; do
    perl -e 'alarm 1800; exec @ARGV' .venv/bin/python scripts/experiments/run_search.py outputs/scale_${side}_${mode}.json $b $mode > outputs/scale_${side}_${mode}.log 2>&1 &
  done
  wait
done
