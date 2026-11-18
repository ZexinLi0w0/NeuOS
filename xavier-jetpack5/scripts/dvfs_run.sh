#!/bin/bash
# NeuOS DVFS-on run on AGX Xavier (L4T R35.4.1). Run with sudo.
# 1) save board state, 2) enable-dvfs.sh, 3) sample clocks while classify.bin runs,
# 4) restore governors, cores, clocks (jetson_clocks) and file modes.
set -u
R=/mnt/ssd/neuos/src
O=/mnt/ssd/neuos/out
N=${1:-200}
TAG=${2:-on}
GPU=/sys/devices/17000000.gv11b/devfreq/17000000.gv11b
CPUF=/sys/devices/system/cpu/cpufreq/policy0
EMC=/sys/kernel/debug/bpmp/debug/clk/emc
HW0=$(ls -d /sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon*)

# file modes changed by enable-dvfs.sh, to restore afterwards
FILES="$CPUF/scaling_setspeed $CPUF/scaling_min_freq $CPUF/scaling_max_freq $GPU/max_freq $GPU/min_freq
/sys/devices/gpu.0/force_idle $EMC/rate $EMC/min_rate $EMC/max_rate
/sys/kernel/debug /sys/kernel/debug/bpmp /sys/kernel/debug/bpmp/debug /sys/kernel/debug/bpmp/debug/clk $EMC"
for c in 0 1 2 3 4 5 6 7; do FILES="$FILES /sys/devices/system/cpu/cpu$c/online"; done
stat -c '%a %n' $FILES 2>/dev/null > $O/modes_before_$TAG.txt
CPU_GOV=$(cat $CPUF/scaling_governor); GPU_GOV=$(cat $GPU/governor)
jetson_clocks --store $O/l4t_dfs_before_$TAG.conf
jetson_clocks --show > $O/jetson_clocks_before_$TAG.txt 2>&1

cd $R/power-utility/AGX && bash enable-dvfs.sh > $O/enable_dvfs_$TAG.log 2>&1
cd $R/examples/triangle

export NEUOS_SHM_DIR=/mnt/ssd/neuos/run
FAN_PWM=$(ls /sys/class/hwmon/*/pwm1 | head -1)
FAN_BEFORE=$(cat $FAN_PWM)

sample() {
  echo "t_ms cpu_khz gpu_hz emc_hz online sys5v_mw fds"
  while true; do
    v=$(cat $HW0/in3_input); i=$(cat $HW0/curr3_input)
    p=$(pgrep -x classify.bin); n=$(ls /proc/$p/fd 2>/dev/null | wc -l)
    echo "$(date +%s%3N) $(cat $CPUF/scaling_cur_freq) $(cat $GPU/cur_freq) $(cat $EMC/rate) $(cat /sys/devices/system/cpu/online) $((v*i/1000)) $n"
    sleep 0.02
  done
}
sample > $O/clocks_$TAG.tsv &
SP=$!
date +%s%3N > $O/start_$TAG.txt
./classify.bin $R/models/bvlc_reference_caffenet/deploy.prototxt \
  $R/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  $R/data/ilsvrc12/imagenet_mean.binaryproto $R/data/ilsvrc12/synset_words.txt \
  $R/examples/images/cat.jpg $N 25 5 all-AGX.config \
  $R/hashtables/TX2/caffenet-cudnn-HASHTABLE.txt \
  $R/models/lowrank/caffenet_lowrank_deploy.prototxt \
  $R/models/lowrank/caffenet_lowrank.caffemodel 1 > $O/dvfs_$TAG.log 2>&1
echo "classify rc=$?"
date +%s%3N > $O/end_$TAG.txt
sleep 1; kill $SP

# restore
for c in 1 2 3 4 5 6 7; do echo 1 > /sys/devices/system/cpu/cpu$c/online; done
echo "$CPU_GOV" > $CPUF/scaling_governor
echo "$GPU_GOV" > $GPU/governor
echo 0 > /sys/devices/gpu.0/force_idle 2>/dev/null
# NeuOS leaves min = max at its last setting; --restore alone does not undo that, so
# re-open the ranges to the hardware maximum first (max before min, since min <= max).
echo $(tr ' ' '\n' < $GPU/available_frequencies | sort -n | tail -1) > $GPU/max_freq
echo $(tr ' ' '\n' < $GPU/available_frequencies | sort -n | tail -1) > $GPU/min_freq
echo $(cat $CPUF/cpuinfo_max_freq) > $CPUF/scaling_max_freq
echo $(cat $CPUF/cpuinfo_max_freq) > $CPUF/scaling_min_freq
jetson_clocks --restore $O/l4t_dfs_before_$TAG.conf
echo $FAN_BEFORE > $FAN_PWM
norm() { grep -v "^SOC" "$1" | sed "s/ IdleStates.*//"; }   # clocks are pinned, so CurrentFreq must match too
jetson_clocks --show > $O/jetson_clocks_after_$TAG.txt 2>&1
if ! diff <(norm $O/jetson_clocks_before_$TAG.txt) <(norm $O/jetson_clocks_after_$TAG.txt); then
  # the board was in the jetson_clocks (all max) state before this run
  echo "restore incomplete; re-applying jetson_clocks"
  jetson_clocks
  echo $FAN_BEFORE > $FAN_PWM
  jetson_clocks --show > $O/jetson_clocks_after_$TAG.txt 2>&1
fi
diff <(norm $O/jetson_clocks_before_$TAG.txt) <(norm $O/jetson_clocks_after_$TAG.txt) && echo "clocks restored"
grep -E "cpu0|GPU|EMC|FAN" $O/jetson_clocks_after_$TAG.txt
while read m f; do chmod "$m" "$f"; done < $O/modes_before_$TAG.txt
stat -c '%a %n' $FILES 2>/dev/null > $O/modes_after_$TAG.txt
diff $O/modes_before_$TAG.txt $O/modes_after_$TAG.txt && echo "modes restored"
