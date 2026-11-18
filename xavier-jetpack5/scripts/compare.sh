#!/bin/bash
# Comparison runs for the Xavier hash table. Run with sudo from /mnt/ssd/neuos, with the
# board in its jetson_clocks state (MAXN, CPU/GPU/EMC pinned at max).
#   pinned:  DVFS off, clocks pinned at max (jetson_clocks)          x REPS, 200 jobs
#   default: DVFS off, Linux default DVFS (schedutil, nvhost_podgov, EMC unlocked)  x REPS
#   NeuOS DVFS on (dvfs_run.sh) with the TX2 table and the two Xavier tables  x REPS, 100 jobs
# Also records idle power (SYS5V, GPU, CPU, VDDRQ rails) for 10 s in the pinned and default
# states, for the per-period energy in the README.
set -u
W=/mnt/ssd/neuos
R=$W/src
O=$W/out/compare
REPS=${REPS:-3}
SKIP_NEUOS=${SKIP_NEUOS:-0}   # 1: baselines only
mkdir -p $O
export NEUOS_SHM_DIR=$W/run
GPU=/sys/devices/17000000.gv11b/devfreq/17000000.gv11b
CPUF=/sys/devices/system/cpu/cpufreq/policy0
EMC=/sys/kernel/debug/bpmp/debug/clk/emc
FAN_PWM=$(ls /sys/class/hwmon/*/pwm1 | head -1)
FAN_BEFORE=$(cat $FAN_PWM)
jetson_clocks --show > $O/jetson_clocks_before.txt 2>&1

classify() {  # $1 = n jobs, $2 = config_mode, $3 = hash table
  (cd $R/examples/triangle && ./classify.bin $R/models/bvlc_reference_caffenet/deploy.prototxt \
    $R/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    $R/data/ilsvrc12/imagenet_mean.binaryproto $R/data/ilsvrc12/synset_words.txt \
    $R/examples/images/cat.jpg $1 25 5 all-AGX.config $3 \
    $R/models/lowrank/caffenet_lowrank_deploy.prototxt $R/models/lowrank/caffenet_lowrank.caffemodel $2)
}

idle_power() {  # mean power (mW) of the four rails over 10 s, sampled every ~20 ms
  local h0 h1; h0=$(ls -d /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*); h1=$(ls -d /sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon*)
  local end=$(( $(date +%s) + 10 ))
  echo "sys5v_mw gpu_mw cpu_mw vddrq_mw"
  while [ "$(date +%s)" -lt "$end" ]; do
    echo "$(( $(cat $h1/in3_input) * $(cat $h1/curr3_input) / 1000 )) $(( $(cat $h0/in1_input) * $(cat $h0/curr1_input) / 1000 )) $(( $(cat $h0/in2_input) * $(cat $h0/curr2_input) / 1000 )) $(( $(cat $h1/in2_input) * $(cat $h1/curr2_input) / 1000 ))"
    sleep 0.02
  done
}

state() { echo "cpu $(cat $CPUF/scaling_min_freq)-$(cat $CPUF/scaling_max_freq) gov $(cat $CPUF/scaling_governor); gpu $(cat $GPU/min_freq)-$(cat $GPU/max_freq) gov $(cat $GPU/governor); emc $(cat $EMC/rate) locked $(cat $EMC/mrq_rate_locked); online $(cat /sys/devices/system/cpu/online)"; }

# 1. pinned
state > $O/state_pinned.txt
idle_power > $O/idle_pinned.tsv
for r in $(seq 1 $REPS); do classify 200 0 $R/hashtables/TX2/caffenet-cudnn-HASHTABLE.txt > $O/pinned_r$r.log 2>&1; done

# 2. Linux default DVFS: open the CPU/GPU ranges and unlock EMC (what jetson_clocks pinned)
echo $(cat $CPUF/cpuinfo_min_freq) > $CPUF/scaling_min_freq
echo $(tr ' ' '\n' < $GPU/available_frequencies | sort -n | head -1) > $GPU/min_freq
echo 0 > $EMC/mrq_rate_locked
sleep 5
state > $O/state_default.txt
idle_power > $O/idle_default.tsv
for r in $(seq 1 $REPS); do classify 200 0 $R/hashtables/TX2/caffenet-cudnn-HASHTABLE.txt > $O/default_r$r.log 2>&1; done
jetson_clocks
echo $FAN_BEFORE > $FAN_PWM
state > $O/state_after_default.txt

# 3. NeuOS with each table (dvfs_run.sh saves and restores the board itself)
[ "$SKIP_NEUOS" = 1 ] || for r in $(seq 1 $REPS); do
  bash $W/dvfs_run.sh 100 tx2_r$r $R/hashtables/TX2/caffenet-cudnn-HASHTABLE.txt > $O/dvfs_run_tx2_r$r.out 2>&1
  bash $W/dvfs_run.sh 100 xsys5v_r$r $W/caffenet-cudnn-HASHTABLE-xavier-sys5v.txt > $O/dvfs_run_xsys5v_r$r.out 2>&1
  bash $W/dvfs_run.sh 100 xsum4_r$r $W/caffenet-cudnn-HASHTABLE-xavier-sum4.txt > $O/dvfs_run_xsum4_r$r.out 2>&1
done

norm() { grep -v "^SOC" "$1" | sed "s/ IdleStates.*//"; }
jetson_clocks --show > $O/jetson_clocks_after.txt 2>&1
diff <(norm $O/jetson_clocks_before.txt) <(norm $O/jetson_clocks_after.txt) && echo "clocks restored"
echo "compare done"
