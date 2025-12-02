#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#ifdef __cplusplus
extern "C" {
#endif

void compute();
void initDeviceMemory();
void freeDeviceMemory();
void copyToDevice();
void copyFromDevice();

#ifdef __cplusplus
}
#endif

#endif