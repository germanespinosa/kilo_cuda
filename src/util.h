#ifndef UTIL_H__
#define UTIL_H__

#define __USE_BSD 1
#define INPUT_WIDTH  3984
#define INPUT_HEIGHT 4096


#include <stdio.h>
#include <time.h>

void** alloc_2d(size_t y_size, size_t x_size, size_t element_size);

#define TIME_IT(ROUTINE_NAME__, LOOPS__, ACTION__)\
{\
    struct timeval tv;\
    struct timezone tz;\
    gettimeofday(&tv, &tz); long GTODStartTime =  tv.tv_sec * 1000 + tv.tv_usec / 1000 ;\
    for (int loops = 0; loops < (LOOPS__); ++loops)\
    {\
        ACTION__;\
    }\
    gettimeofday(&tv, &tz); long GTODEndTime =  tv.tv_sec * 1000 + tv.tv_usec / 1000 ;\
    printf("%g,", LOOPS__, (double)(GTODEndTime - GTODStartTime));\
}

#endif
