#include <iostream>
#include <string>
#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <sstream>
#include <vector>
#include <fstream>
#include <utility>
#include <dirent.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;


//#define BIG_CLUSTER_GOVERNOR "/sys/devices/system/cpu/cpufreq/policy1/scaling_governor"
//#define BIG_CLUSTER_FREQ "/sys/devices/system/cpu/cpufreq/policy1/scaling_setspeed"
//#define BIG_CLUSTER_MAX "/sys/devices/system/cpu/cpufreq/policy1/scaling_max_freq"
//#define BIG_CLUSTER_MIN "/sys/devices/system/cpu/cpufreq/policy1/scaling_min_freq"

//#define LITTLE_CLUSTER_GOVERNOR "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
//#define LITTLE_CLUSTER_FREQ "/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed"
//#define LITTLE_CLUSTER_MAX "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"
//#define LITTLE_CLUSTER_MIN "/sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"

#define CLUSTER_GOVERNOR "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
#define CLUSTER_FREQ "/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed"
#define CLUSTER_MAX "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"
#define CLUSTER_MIN "/sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"

#define GPU_GOVERNOR "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/governor"
#define GPU_FREQ "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/userspace/set_freq"
#define GPU_ONLINE "/sys/devices/gpu.0/force_idle"
#define GPU_MAX "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq"
#define GPU_MIN "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq"

#define CPU_ONLINE "/sys/devices/system/cpu/cpu%d/online"
#define EMC_FREQ "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
#define EMC_MIN "/sys/kernel/debug/bpmp/debug/clk/emc/min_rate"
#define EMC_MAX "/sys/kernel/debug/bpmp/debug/clk/emc/max_rate"


//int setBigClusterFreq(uint64_t freq){
//	FILE* f_freq = fopen(BIG_CLUSTER_FREQ,"w");
//	FILE* f_max = fopen(BIG_CLUSTER_MAX, "w");
//	FILE* f_min = fopen(BIG_CLUSTER_MIN, "w");
//	if(f_freq == NULL || f_max ==NULL || f_min ==NULL){
//		fprintf(stderr, "File open failure\n");
//		return -1;
//	}
//	char buff[128];
//	sprintf(buff, "%" PRIu64, freq);
//	fputs(buff, f_freq);
//	fputs(buff, f_min);
//	fputs(buff, f_max);
//	fclose(f_freq);
//	fclose(f_min);
//	fclose(f_max);
//	return 0;
//}
//int setLittleClusterFreq(uint64_t freq){
//	FILE* f_freq = fopen(LITTLE_CLUSTER_FREQ,"w");
//	FILE* f_max = fopen(LITTLE_CLUSTER_MAX, "w");
//	FILE* f_min = fopen(LITTLE_CLUSTER_MIN, "w");
//	if(f_freq == NULL || f_max ==NULL || f_min ==NULL){
//		fprintf(stderr, "File open failure\n");
//		return -1;
//	}
//	char buff[128];
//	sprintf(buff, "%" PRIu64, freq);
//	fputs(buff, f_freq);
//	fputs(buff, f_min);
//	fputs(buff, f_max);
//	fclose(f_freq);
//	fclose(f_min);
//	fclose(f_max);
//	return 0;
//}
// Each value is written and closed on its own, in an order that works whether the clock
// goes up or down: the kernel rejects min > max, so max, min, max again, then the target.
// (Writing through buffered FILE*s closed in freq/min/max order made every increase fail.)
static int write_sysfs(const char* path, const char* value){
	FILE* f = fopen(path, "w");
	if(f == NULL){
		fprintf(stderr, "File open failure: %s\n", path);
		return -1;
	}
	fputs(value, f);
	return fclose(f) == 0 ? 0 : -1;
}
static int set_clock(const char* max_path, const char* min_path, const char* cur_path, uint64_t value){
	char buff[128];
	sprintf(buff, "%" PRIu64, value);
	write_sysfs(max_path, buff);          // fails when lowering below the current min; retried below
	int rc = write_sysfs(min_path, buff);
	rc |= write_sysfs(max_path, buff);
	if(cur_path) rc |= write_sysfs(cur_path, buff);
	return rc;
}
int setClusterFreq(uint64_t freq){
	return set_clock(CLUSTER_MAX, CLUSTER_MIN, CLUSTER_FREQ, freq);
}
int setCoresOnline(int mask[]){
	int idx[]={0,1,2,3,4,5,6,7};
	int i;
	for(i=1;i<8;i++){
		char buffer[128];
		sprintf(buffer, CPU_ONLINE, idx[i]);
		FILE* f_online=fopen(buffer,"w");
		if(f_online==NULL){
			fprintf(stderr, "Failed to set cores online. File %s failed to open.\n",buffer);
			return -1;
		}
		if(mask[i]==0){
			fputs("0", f_online);	
		}else{
			fputs("1", f_online);
		}
		fclose(f_online);
	}
	return 0;
}

//int setSmallCoresOnline(int mask[]){
//	int smallIdx[]={0,3,4,5};
//	int i;
//	for(i=1;i<4;i++){
//		char buffer[128];
//		sprintf(buffer, CPU_ONLINE, smallIdx[i]);
//		FILE* f_online=fopen(buffer,"w");
//		if(f_online==NULL){
//			fprintf(stderr, "Failed to set small cores online\n");
//			return -1;
//		}
//		if(mask[i]==0){
//			fputs("0", f_online);	
//		}else{
//			fputs("1", f_online);
//		}
//		fclose(f_online);
//	}
//	return 0;
//}
//
//int setSmallCoresOnline(int cores){
//	int smallIdx[]={0,5,4,3};
//	int i;
//	for(i=1;i<4;i++){
//		char buffer[128];
//		sprintf(buffer, CPU_ONLINE, smallIdx[i]);
//		FILE* f_online=fopen(buffer,"w");
//		if(f_online==NULL){
//			fprintf(stderr, "Failed to set small cores online\n");
//			return -1;
//		}
//		if(i>=cores){
//			fputs("0", f_online);	
//		}else{
//			fputs("1", f_online);
//		}
//		fclose(f_online);
//	}
//	return 0;
//}
//
//int setBigCoresOnline(int mask[]){
//	int bigIdx[]={1,2};
//	int i;
//	for(i=0;i<2;i++){
//		char buffer[128];
//		sprintf(buffer, CPU_ONLINE, bigIdx[i]);
//		FILE* f_online=fopen(buffer,"w");
//		if(f_online==NULL){
//			fprintf(stderr, "Failed to set small cores online\n");
//			return -1;
//		}
//		if(mask[i]==0){
//			fputs("0", f_online);	
//		}else{
//			fputs("1", f_online);
//		}
//		fclose(f_online);
//	}
//	return 0;
//}
//
//int setBigCoresOnline(int cores){
//	int bigIdx[]={2,1};
//	int i;
//	for(i=0;i<2;i++){
//		char buffer[128];
//		sprintf(buffer, CPU_ONLINE, bigIdx[i]);
//		FILE* f_online=fopen(buffer,"w");
//		if(f_online==NULL){
//			fprintf(stderr, "Failed to set small cores online\n");
//			return -1;
//		}
//		if(i>=cores){
//			fputs("0", f_online);	
//		}else{
//			fputs("1", f_online);
//		}
//		fclose(f_online);
//	}
//	return 0;
//}
int setGpuFreq(uint64_t gFreq){
	if(gFreq == 0){
		int rc = write_sysfs(GPU_ONLINE, "1");
		return rc | set_clock(GPU_MAX, GPU_MIN, GPU_FREQ, 140250000);
	}
	int rc = write_sysfs(GPU_ONLINE, "0");
	return rc | set_clock(GPU_MAX, GPU_MIN, GPU_FREQ, gFreq);
}
int setEmcFreq(uint64_t emcFreq){
	// all-AGX.config stores the EMC clock in MHz (2133) but bpmp's emc/rate takes Hz;
	// writing 2133 pinned EMC to its 204 MHz floor.
	if(emcFreq > 0 && emcFreq < 1000000) emcFreq *= 1000000;
	return set_clock(EMC_MAX, EMC_MIN, EMC_FREQ, emcFreq);
}
