#line 2

// Defines for static analyzer:
#ifndef WORKGROUP_SIZE
    #define WORKGROUP_SIZE 256
    #define WAVEFRONT_SIZE 1
    //#define N 1
    #define N       3
    #define EPSILON 0.01f
    #define LIMIT   100
    #define sigmaS  8.0f
    #define sigmaR  5.0f
#endif

#define lN (N + 2)

#define MAX_NEIGHBOURS 27
#define MAX_SAMPLES    64
#define IDXDS_MAX      (MAX_NEIGHBOURS * MAX_SAMPLES)
#define IDXDS_EMPTY    INT_MAX

#if 0
#define assert(expression) if (!(expression)) printf("Assertion failed at line %d!", __LINE__);
#else
#define assert(expression) 
#endif

inline int getBucNeigh(const int j, const int nBuck1, const int nBuck2)
{
    return ((j / 9) % 3 - 1) + nBuck1 * (((j / 3) % 3 - 1) + nBuck2 * (j % 3 - 1));
// Equal to:
//    idxd = 0;
//    for (cBuck1=-1; cBuck1<=1; cBuck1++)
//        for (cBuck2=-1; cBuck2<=1; cBuck2++)
//            for (cBuck3=-1; cBuck3<=1; cBuck3++)
//                bucNeigh[idxd++] = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
//    return bucNeigh[j];
}

__attribute__((reqd_work_group_size(1, WORKGROUP_SIZE, 1)))
__kernel void meanShiftFilter(__global const float* sdata,     // lN*L
                              __global const int*   buckets,   // nBuck1*nBuck2*nBuck3
                              __global const float* weightMap, // L
                              __global const int*   slist,     // L
                              __global       float* msRawData, // N*L
                              const int L,
                              const int width, const int height,
                              const float sMins,
                              const int nBuck1, const int nBuck2, const int nBuck3
)
{
    __local int    idxds[IDXDS_MAX];
    __local float* cache = (__local float*) idxds;
    assert (WORKGROUP_SIZE * (lN + 1) <= IDXDS_MAX);

    const int i = get_global_id(0);
    const int threadY = get_local_id(1);
    const int thread0 = 0;

    const float hiLTr = 80.0f / sigmaR;

    float yk[lN];
    float Mh[lN];
    float wsuml;

    // Assign window center (window centers are
    // initialized by createLattice to be the point
    // data[i])
    for (int j = 0; j < lN; j++)
        yk[j] = sdata[i * lN + j];

    // Calculate the mean shift vector using the lattice
    // LatticeMSVector(Mh, yk);
    /*****************************************************/
    // Initialize mean shift vector
    for (int k = 0; k < lN; ++k)
        Mh[k] = 0.0f;
    wsuml = 0.0f;

    if (threadY < MAX_NEIGHBOURS) {
        int j = threadY;
        int cBuck;
        {
            int cBuck1 = (int) yk[0] + 1;
            int cBuck2 = (int) yk[1] + 1;
            int cBuck3 = (int) (yk[2] - sMins) + 1;
            cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);
        }
        int bucketId = cBuck + getBucNeigh(j, nBuck1, nBuck2);
        int idxd = buckets[bucketId];
        // list parse, crt point is cHeadList
        int k = 0;
        for (; k < MAX_SAMPLES && idxd >= 0; ++k) {
            idxds[j * MAX_SAMPLES + k] = idxd;
            idxd = slist[idxd];
        }
        for (; k < MAX_SAMPLES; ++k) {
            idxds[j * MAX_SAMPLES + k] = IDXDS_EMPTY;
        }
        assert(idxd < 0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = threadY; j < IDXDS_MAX; j += WORKGROUP_SIZE) {
        int idxd = idxds[j];
        if (idxd != IDXDS_EMPTY) {
            int idxs = lN * idxd;
            // determine if inside search window

            float el, diff;
            el = sdata[idxs + 0] - yk[0];
            diff = el * el;
            el = sdata[idxs + 1] - yk[1];
            diff += el * el;

            if (diff < 1.0f) {
                el = sdata[idxs + 2] - yk[2];
                if (yk[2] > hiLTr)
                    diff = 4.0f * el * el;
                else
                    diff = el * el;

#if (N == 3)
                {
                    el = sdata[idxs + 3] - yk[3];
                    diff += el * el;
                    el = sdata[idxs + 4] - yk[4];
                    diff += el * el;
                }
#endif

                if (diff < 1.0f) {
                    float weight = 1.0f - weightMap[idxd];
                    for (int k = 0; k < lN; ++k)
                        Mh[k] += weight * sdata[idxs + k];
                    wsuml += weight;
                }
            }
        }
    }

    {
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < lN; ++k)
            cache[threadY * (lN + 1) + k] = Mh[k];
        cache[threadY * (lN + 1) + lN] = wsuml;

        int step = WORKGROUP_SIZE / 2;
        while (step > 0) {
            if (2 * step > WAVEFRONT_SIZE)
                barrier(CLK_LOCAL_MEM_FENCE);

            if (threadY < step) {
                for (int k = 0; k < lN; ++k)
                    cache[threadY * (lN + 1) + k] += cache[(threadY + step) * (lN + 1) + k];
                cache[threadY * (lN + 1) + lN] += cache[(threadY + step) * (lN + 1) + lN];
            }

            step /= 2;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < lN; ++k)
            Mh[k] = cache[thread0 * (lN + 1) + k];
        wsuml = cache[thread0 * (lN + 1) + lN];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (wsuml > 0) {
            for (int j = 0; j < lN; j++)
                Mh[j] = Mh[j] / wsuml - yk[j];
        } else {
            for (int j = 0; j < lN; j++)
                Mh[j] = 0.0f;
        }
    }
    /*****************************************************/
    // Calculate its magnitude squared
    float mvAbs = 0.0f;
    for (int j = 0; j < lN; j++)
        mvAbs += Mh[j] * Mh[j];

    // Keep shifting window center until the magnitude squared of the
    // mean shift vector calculated at the window center location is
    // under a specified threshold (Epsilon)

    // NOTE: iteration count is for speed up purposes only - it
    //       does not have any theoretical importance
    for (int iterationCount = 1; (mvAbs >= EPSILON) && (iterationCount < LIMIT); ++iterationCount) {

        // Shift window location
        for (int j = 0; j < lN; j++)
            yk[j] += Mh[j];

        // Calculate the mean shift vector at the new
        // window location using lattice
        // LatticeMSVector(Mh, yk);
        /*****************************************************/
        // Initialize mean shift vector
        for (int j = 0; j < lN; j++)
            Mh[j] = 0.0f;
        wsuml = 0.0f;

        if (threadY < MAX_NEIGHBOURS) {
            int j = threadY;
            int cBuck;
            {
                int cBuck1 = (int) yk[0] + 1;
                int cBuck2 = (int) yk[1] + 1;
                int cBuck3 = (int) (yk[2] - sMins) + 1;
                cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);
            }
            int bucketId = cBuck + getBucNeigh(j, nBuck1, nBuck2);
            int idxd = buckets[bucketId];
            // list parse, crt point is cHeadList
            int k = 0;
            for (; k < MAX_SAMPLES && idxd >= 0; ++k) {
                idxds[j * MAX_SAMPLES + k] = idxd;
                idxd = slist[idxd];
            }
            for (; k < MAX_SAMPLES; ++k) {
                idxds[j * MAX_SAMPLES + k] = IDXDS_EMPTY;
            }
            assert(idxd < 0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = threadY; j < IDXDS_MAX; j += WORKGROUP_SIZE) {
            //int idxd = idxds[j]; // BUT THIS EQUAL LINE LEADS TO -2 VGPRs ON R9 390X:
            int idxd = idxds[(j / 256) * 256 + j % 256];

            // list parse, crt point is cHeadList
            if (idxd != IDXDS_EMPTY) {
                int idxs = lN * idxd;
                // determine if inside search window
                float el, diff;
                el = sdata[idxs + 0] - yk[0];
                diff = el * el;
                el = sdata[idxs + 1] - yk[1];
                diff += el * el;

                if (diff < 1.0f) {
                    el = sdata[idxs + 2] - yk[2];
                    if (yk[2] > hiLTr)
                        diff = 4.0f * el * el;
                    else
                        diff = el * el;

#if (N == 3)
                    {
                        el = sdata[idxs + 3] - yk[3];
                        diff += el * el;
                        el = sdata[idxs + 4] - yk[4];
                        diff += el * el;
                    }
#endif

                    if (diff < 1.0f) {
                        float weight = 1.0f - weightMap[idxd];
                        for (int k = 0; k < lN; k++)
                            Mh[k] += weight * sdata[idxs + k];
                        wsuml += weight;
                    }
                }
            }
        }

        {
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < lN; ++k)
                cache[threadY * (lN + 1) + k] = Mh[k];
            cache[threadY * (lN + 1) + lN] = wsuml;

            int step = WORKGROUP_SIZE / 2;
            while (step > 0) {
                if (2 * step > WAVEFRONT_SIZE)
                    barrier(CLK_LOCAL_MEM_FENCE);

                if (threadY < step) {
                    for (int k = 0; k < lN; ++k)
                        cache[threadY * (lN + 1) + k] += cache[(threadY + step) * (lN + 1) + k];
                    cache[threadY * (lN + 1) + lN] += cache[(threadY + step) * (lN + 1) + lN];
                }

                step /= 2;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < lN; ++k)
                Mh[k] = cache[thread0 * (lN + 1) + k];
            wsuml = cache[thread0 * (lN + 1) + lN];

            barrier(CLK_LOCAL_MEM_FENCE);

            if (wsuml > 0) {
                for (int j = 0; j < lN; j++)
                    Mh[j] = Mh[j] / wsuml - yk[j];
            } else {
                for (int j = 0; j < lN; j++)
                    Mh[j] = 0.0f;
            }
        }
        /*****************************************************/

        // Calculate its magnitude squared
        mvAbs = (Mh[0] * Mh[0] + Mh[1] * Mh[1]) * sigmaS * sigmaS;
        if (N == 3)
            mvAbs += (Mh[2] * Mh[2] + Mh[3] * Mh[3] + Mh[4] * Mh[4]) * sigmaR * sigmaR;
        else
            mvAbs += Mh[2] * Mh[2] * sigmaR * sigmaR;
    }

    // Shift window location
    for (int j = 0; j < lN; j++)
        yk[j] += Mh[j];

    //store result into msRawData...
    if (threadY < N) {
        int j = threadY;
        msRawData[N * i + j] = (float) (yk[j + 2] * sigmaR);
    }
}
