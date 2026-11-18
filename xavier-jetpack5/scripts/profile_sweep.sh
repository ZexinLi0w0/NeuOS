#!/bin/bash
# Per-layer profile of CaffeNet over a subset of all-AGX.config, for the Xavier hash table.
# Run with sudo. Subset: 6 CPU clocks x 4 core masks (1, 2, 4, 8 cores) x all 14 GPU clocks
# = 336 configurations; EMC is 2133 MHz in every line of all-AGX.config.
# Saves the board state first and restores it at the end (as dvfs_run.sh).
set -u
W=/mnt/ssd/neuos
R=$W/src
O=$W/out/profile
mkdir -p $O
GPU=/sys/devices/17000000.gv11b/devfreq/17000000.gv11b
CPUF=/sys/devices/system/cpu/cpufreq/policy0
FAN_PWM=$(ls /sys/class/hwmon/*/pwm1 | head -1)
FAN_BEFORE=$(cat $FAN_PWM)
CPU_GOV=$(cat $CPUF/scaling_governor); GPU_GOV=$(cat $GPU/governor)
EMC=/sys/kernel/debug/bpmp/debug/clk/emc
FILES="$CPUF/scaling_setspeed $CPUF/scaling_min_freq $CPUF/scaling_max_freq $GPU/max_freq $GPU/min_freq
/sys/devices/gpu.0/force_idle $EMC/rate $EMC/min_rate $EMC/max_rate
/sys/kernel/debug /sys/kernel/debug/bpmp /sys/kernel/debug/bpmp/debug /sys/kernel/debug/bpmp/debug/clk $EMC"
for c in 0 1 2 3 4 5 6 7; do FILES="$FILES /sys/devices/system/cpu/cpu$c/online"; done
stat -c '%a %n' $FILES 2>/dev/null > $O/modes_before.txt
jetson_clocks --show > $O/jetson_clocks_before.txt 2>&1

cd $R/power-utility/AGX && bash enable-dvfs.sh > $O/enable_dvfs.log 2>&1

CPUS="115200 499200 960000 1420800 1881600 2265600"
MASKS="1_0_0_0_0_0_0_0 1_1_0_0_0_0_0_0 1_1_1_1_0_0_0_0 1_1_1_1_1_1_1_1"
awk -v cpus="$CPUS" -v masks="$MASKS" 'BEGIN{n=split(cpus,c," "); for(i=1;i<=n;i++) C[c[i]]=1; n=split(masks,m," "); for(i=1;i<=n;i++) M[m[i]]=1}
  { k=$3"_"$4"_"$5"_"$6"_"$7"_"$8"_"$9"_"$10; if (($2 in C) && (k in M)) print }' \
  $R/examples/triangle/all-AGX.config > $O/configs.txt
echo "$(wc -l < $O/configs.txt) configurations" > $O/sweep.log

# one process: the model is loaded once, then each configuration is applied with NeuOS'
# setters and profiled (scripts/profile_layers.cpp); <id>.log records requested and measured clocks
$W/profile_layers $R/models/bvlc_reference_caffenet/deploy.prototxt \
  $R/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel 200 $O/configs.txt $O >> $O/sweep.log 2>&1
echo "sweep done" >> $O/sweep.log

# restore
for c in 1 2 3 4 5 6 7; do echo 1 > /sys/devices/system/cpu/cpu$c/online; done
echo "$CPU_GOV" > $CPUF/scaling_governor
echo "$GPU_GOV" > $GPU/governor
echo 0 > /sys/devices/gpu.0/force_idle 2>/dev/null
jetson_clocks          # the board was in the jetson_clocks (all max) state before
echo $FAN_BEFORE > $FAN_PWM
while read m f; do chmod "$m" "$f"; done < $O/modes_before.txt
norm() { grep -v "^SOC" "$1" | sed "s/ IdleStates.*//"; }
jetson_clocks --show > $O/jetson_clocks_after.txt 2>&1
diff <(norm $O/jetson_clocks_before.txt) <(norm $O/jetson_clocks_after.txt) && echo "clocks restored" >> $O/sweep.log
