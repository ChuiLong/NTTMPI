/*  ntt_simd.cpp  ——  MPI + OpenMP + AVX2 SIMD Barrett‑NTT  */

#include <bits/stdc++.h>
#include <immintrin.h>          // AVX / AVX2
#include <omp.h>
#include <mpi.h>
#include <iomanip>              // for std::setprecision
#include <numeric>              // for std::accumulate
#include <algorithm>            // for std::max_element, std::min_element

using std::int32_t;
using std::int64_t;
using std::uint64_t;

/* ---------- 性能统计全局变量 ---------- */
struct PerformanceStats {
    double compute_time = 0.0;      // 本地计算时间
    double comm_time = 0.0;         // 通信时间
    int work_units = 0;             // 工作单元数量
    double total_time = 0.0;        // 总时间
    std::vector<double> level_times;    // 每层计算时间
    std::vector<int> level_work_counts; // 每层工作量
};

static PerformanceStats perf_stats;

/* ---------- Barrett 取模 ---------- */
struct Barrett {
    int64_t p;
    uint64_t mul;               // floor(2^64 / p) + 1
};

static inline Barrett getBarrett(int64_t p)
{
    Barrett b;
    b.p   = p;
    b.mul = static_cast<uint64_t>(((__uint128_t)1 << 64) / p + 1);
    return b;
}

static inline int64_t barrett_mul_64(int64_t a,int64_t b,const Barrett& bar)
{
    __uint128_t z = (__uint128_t)a * (uint64_t)b;
    __uint128_t t = (z * bar.mul) >> 64;
    int64_t      r = (int64_t)(z - t * bar.p);
    if (r < 0)          r += bar.p;
    else if (r >= bar.p) r -= bar.p;
    return r;
}

static inline int64_t barrett_add_64(int64_t a,int64_t b,const Barrett& bar)
{
    int64_t r = a + b;
    if (r >= bar.p) r -= bar.p;
    return r;
}
static inline int64_t barrett_sub_64(int64_t a,int64_t b,const Barrett& bar)
{
    int64_t r = a - b;
    if (r < 0) r += bar.p;
    return r;
}

static int64_t barrett_pow_64(int64_t base,int64_t exp,const Barrett& bar)
{
    int64_t ans = 1 % bar.p;
    int64_t cur = (base % bar.p + bar.p) % bar.p;
    while (exp) {
        if (exp & 1) ans = barrett_mul_64(ans,cur,bar);
        cur = barrett_mul_64(cur,cur,bar);
        exp >>= 1;
    }
    return ans;
}

/* ---------- 全局常量 ---------- */
static const int G       = 3;        // 原根
static const int MAX_N   = 300000;   // 测试数据上限
static Barrett bar;                  // 当前模
static int32_t currentP;

/* ---------- 单位根表 ---------- */
static std::vector<std::vector<int32_t>> w_cache, w_inv_cache;

static inline void setBarrett(int p)
{
    currentP = p;
    bar      = getBarrett(p);
}

/* 快速幂接口保留用于 inv_m 计算 */
static inline int32_t mod_pow(int32_t x,int32_t y,int32_t p)
{
    int64_t ans = 1, base = (x % p + p) % p;
    while (y) {
        if (y & 1) ans = barrett_mul_64(ans,base,bar);
        base = barrett_mul_64(base,base,bar);
        y >>= 1;
    }
    return (int32_t)ans;
}

/* 预表 twiddle */
static void precompute_roots(int max_n,int p)
{
    int max_lvl = __builtin_ctz(max_n);
    w_cache.clear();
    w_inv_cache.clear();
    w_cache.resize(max_lvl+1);
    w_inv_cache.resize(max_lvl+1);
    
    for (int lvl=0,mid=1; mid<max_n; mid<<=1,++lvl) {
        int blk = mid<<1;
        int32_t w     = (int32_t)barrett_pow_64(G,(p-1)/blk,bar);
        int32_t w_inv = (int32_t)barrett_pow_64(w,p-2,bar);
        w_cache[lvl].resize(mid);
        w_inv_cache[lvl].resize(mid);
        int32_t omega     = 1, omega_inv = 1;
        for (int j=0;j<mid;++j) {
            w_cache[lvl][j]     = omega;
            w_inv_cache[lvl][j] = omega_inv;
            omega     = (int32_t)barrett_mul_64(omega,w,bar);
            omega_inv = (int32_t)barrett_mul_64(omega_inv,w_inv,bar);
        }
    }
}

/* ---------- bit‑reverse ---------- */
static void bit_reverse(int32_t* a,int n)
{
    int lg = __builtin_ctz(n);
    std::vector<int> rev(n);
    for (int i=0;i<n;++i)
        rev[i] = (rev[i>>1]>>1) | ((i&1)<<(lg-1));
    for (int i=0;i<n;++i)
        if (i < rev[i]) std::swap(a[i],a[rev[i]]);
}

/* ---------- SIMD 帮助 ---------- */
#if defined(__AVX2__)
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

#if HAS_AVX2
/* Original butterfly8 implementation - with fixed comparisons */
static inline void butterfly8(int32_t* x_ptr,int32_t* y_ptr,
                              const int32_t* omega_ptr)
{
    __m256i vx      = _mm256_loadu_si256((__m256i*)x_ptr);
    __m256i vy_raw  = _mm256_loadu_si256((__m256i*)y_ptr);
    __m256i vomega  = _mm256_loadu_si256((__m256i*)omega_ptr);

    // Use unaligned operations for safety
    int32_t ytmp[8];
    int32_t otmp[8];
    _mm256_storeu_si256((__m256i*)ytmp,  vy_raw);
    _mm256_storeu_si256((__m256i*)otmp,  vomega);
    for (int k=0;k<8;++k)
        ytmp[k] = (int32_t)barrett_mul_64(ytmp[k],otmp[k],bar);
    __m256i vy = _mm256_loadu_si256((__m256i*)ytmp);

    __m256i vp   = _mm256_set1_epi32((int)bar.p);
    __m256i vp_minus_1 = _mm256_set1_epi32((int)(bar.p - 1));

    __m256i vadd = _mm256_add_epi32(vx,vy);
    __m256i vsub = _mm256_sub_epi32(vx,vy);

    // For addition: if result > p-1, subtract p
    __m256i cmpadd = _mm256_cmpgt_epi32(vadd, vp_minus_1);
    vadd = _mm256_sub_epi32(vadd,_mm256_and_si256(cmpadd,vp));

    // For subtraction: if result < 0, add p
    __m256i cmpsub = _mm256_cmpgt_epi32(_mm256_setzero_si256(),vsub);
    vsub = _mm256_add_epi32(vsub,_mm256_and_si256(cmpsub,vp));

    _mm256_storeu_si256((__m256i*)x_ptr,  vadd);
    _mm256_storeu_si256((__m256i*)y_ptr,  vsub);
}

/* SIMD pointwise multiplication for 8 elements */
static inline void pointwise_mul8(int32_t* c, const int32_t* a, 
                                  const int32_t* b)
{
    __m256i va = _mm256_loadu_si256((__m256i*)a);
    __m256i vb = _mm256_loadu_si256((__m256i*)b);
    
    int32_t a_arr[8], b_arr[8], c_arr[8];
    _mm256_storeu_si256((__m256i*)a_arr, va);
    _mm256_storeu_si256((__m256i*)b_arr, vb);
    
    for (int i = 0; i < 8; i++) {
        c_arr[i] = (int32_t)barrett_mul_64(a_arr[i], b_arr[i], bar);
    }
    
    _mm256_storeu_si256((__m256i*)c, _mm256_loadu_si256((__m256i*)c_arr));
}

/* SIMD scalar multiplication for normalization */
static inline void scalar_mul8(int32_t* a, int32_t scalar)
{
    __m256i va = _mm256_loadu_si256((__m256i*)a);
    
    int32_t a_arr[8];
    _mm256_storeu_si256((__m256i*)a_arr, va);
    
    for (int i = 0; i < 8; i++) {
        a_arr[i] = (int32_t)barrett_mul_64(a_arr[i], scalar, bar);
    }
    
    _mm256_storeu_si256((__m256i*)a, _mm256_loadu_si256((__m256i*)a_arr));
}
#endif

/* ---------- ntt_local (含 SIMD) ---------- */
static void ntt_local(int32_t* a,int n,int inv)
{
    auto compute_start = std::chrono::high_resolution_clock::now();
    
    bit_reverse(a,n);

    for (int mid=1,lvl=0; mid<n; mid<<=1,++lvl)
    {
        const int32_t* omega = (inv==1? w_cache[lvl]:w_inv_cache[lvl]).data();

        #pragma omp parallel for schedule(static)
        for (int i=0;i<n;i+=mid<<1)
        {
            int j = 0;
#if HAS_AVX2
            if (mid >= 8) {
                for (; j+7<mid; j+=8)
                    butterfly8(a+i+j, a+i+j+mid, omega+j);
            }
#endif
            for (; j<mid; ++j) {
                int64_t x = a[i+j];
                int64_t y = barrett_mul_64(omega[j], a[i+j+mid], bar);
                a[i+j]       = (int32_t)barrett_add_64(x,y,bar);
                a[i+j+mid]   = (int32_t)barrett_sub_64(x,y,bar);
            }
        }
    }
    
    auto compute_end = std::chrono::high_resolution_clock::now();
    perf_stats.compute_time += std::chrono::duration<double, std::milli>(compute_end-compute_start).count();
}

/* ---------- MPI + OpenMP 混合 NTT (仅高层通信) ---------- */
static void ntt_hybrid(int32_t* a,int n,int inv,MPI_Comm comm)
{
    int rank,size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);

    const int MPI_THRESHOLD = n / (4*size);   // 块阈值

    auto comm_start = std::chrono::high_resolution_clock::now();
    if (rank==0) bit_reverse(a,n);
    MPI_Bcast(a,n,MPI_INT,0,comm);
    auto comm_end = std::chrono::high_resolution_clock::now();
    perf_stats.comm_time += std::chrono::duration<double, std::milli>(comm_end-comm_start).count();

    int local_lvl = 0;
    /* ---------- 低层：全部本地 ---------- */
    auto local_start = std::chrono::high_resolution_clock::now();
    for (int mid=1; mid<MPI_THRESHOLD && mid<n; mid<<=1,++local_lvl) {
        const int32_t* omega = (inv==1? w_cache[local_lvl]:w_inv_cache[local_lvl]).data();
        #pragma omp parallel for schedule(static)
        for (int i=0;i<n;i+=mid<<1) {
            int j=0;
#if HAS_AVX2
            if (mid>=8)
                for (; j+7<mid; j+=8)
                    butterfly8(a+i+j,a+i+j+mid,omega+j);
#endif
            for (; j<mid; ++j) {
                int64_t x=a[i+j];
                int64_t y=barrett_mul_64(omega[j],a[i+j+mid],bar);
                a[i+j]     =(int32_t)barrett_add_64(x,y,bar);
                a[i+j+mid] =(int32_t)barrett_sub_64(x,y,bar);
            }
        }
    }
    auto local_end = std::chrono::high_resolution_clock::now();
    perf_stats.compute_time += std::chrono::duration<double, std::milli>(local_end-local_start).count();

    /* ---------- 高层：分块 + Allgatherv ---------- */
    std::vector<int32_t> buf(n);
    int32_t* cur=a; int32_t* nxt=buf.data();

    for (int mid=MPI_THRESHOLD,lvl=local_lvl; mid<n; mid<<=1,++lvl)
    {
        auto level_start = std::chrono::high_resolution_clock::now();
        
        int blkSize=mid<<1, numBlk=n/blkSize;
        int blk_beg= rank   *numBlk/size;
        int blk_end=(rank+1)*numBlk/size;

        int work_this_level = blk_end - blk_beg;
        perf_stats.work_units += work_this_level;
        perf_stats.level_work_counts.push_back(work_this_level);

        const int32_t* omega = (inv==1? w_cache[lvl]:w_inv_cache[lvl]).data();

        auto compute_start2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (int blk=blk_beg; blk<blk_end; ++blk) {
            int base=blk*blkSize;
            int j=0;
#if HAS_AVX2
            if (mid>=8)
                for (; j+7<mid; j+=8)
                    butterfly8(cur+base+j,cur+base+j+mid,omega+j);
#endif
            for (; j<mid; ++j) {
                int64_t x=cur[base+j];
                int64_t y=barrett_mul_64(omega[j],cur[base+j+mid],bar);
                cur[base+j]       =(int32_t)barrett_add_64(x,y,bar);
                cur[base+j+mid]   =(int32_t)barrett_sub_64(x,y,bar);
            }
        }
        auto compute_end2 = std::chrono::high_resolution_clock::now();
        
        /* 汇总高层结果 */
        auto comm_start2 = std::chrono::high_resolution_clock::now();
        std::vector<int> cnt(size), disp(size);
        for (int r=0;r<size;++r) {
            int s=r*numBlk/size, e=(r+1)*numBlk/size;
            cnt [r]=(e-s)*blkSize;
            disp[r]= s   *blkSize;
        }
        MPI_Allgatherv(cur+blk_beg*blkSize, cnt[rank], MPI_INT,
                       nxt, cnt.data(), disp.data(),
                       MPI_INT, comm);
        auto comm_end2 = std::chrono::high_resolution_clock::now();
        
        auto level_end = std::chrono::high_resolution_clock::now();
        double level_time = std::chrono::duration<double, std::milli>(level_end-level_start).count();
        double compute_time = std::chrono::duration<double, std::milli>(compute_end2-compute_start2).count();
        double comm_time = std::chrono::duration<double, std::milli>(comm_end2-comm_start2).count();
        
        perf_stats.level_times.push_back(level_time);
        perf_stats.compute_time += compute_time;
        perf_stats.comm_time += comm_time;
        
        std::swap(cur,nxt);
    }
    
    if (cur!=a) std::memcpy(a,cur,n*sizeof(int32_t));
}

/* ---------- 性能分析函数 ---------- */
static void analyze_load_balance(int total_blocks, double total_time, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // 收集所有进程的性能数据
    std::vector<int> all_work(size);
    std::vector<double> all_compute_time(size);
    std::vector<double> all_comm_time(size);
    std::vector<double> all_total_time(size);
    
    double process_total_time = perf_stats.compute_time + perf_stats.comm_time;
    
    MPI_Gather(&perf_stats.work_units, 1, MPI_INT, all_work.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&perf_stats.compute_time, 1, MPI_DOUBLE, all_compute_time.data(), 1, MPI_DOUBLE, 0, comm);
    MPI_Gather(&perf_stats.comm_time, 1, MPI_DOUBLE, all_comm_time.data(), 1, MPI_DOUBLE, 0, comm);
    MPI_Gather(&process_total_time, 1, MPI_DOUBLE, all_total_time.data(), 1, MPI_DOUBLE, 0, comm);
    
    if (rank == 0) {
        // 计算负载不均衡率
        int max_work = *std::max_element(all_work.begin(), all_work.end());
        int min_work = *std::min_element(all_work.begin(), all_work.end());
        double avg_work = (double)total_blocks / size;
        
        double work_imbalance = (max_work > 0) ? (max_work - min_work) / std::max(avg_work, 1.0) * 100.0 : 0.0;
        
        // 计算时间不均衡率
        double max_compute = *std::max_element(all_compute_time.begin(), all_compute_time.end());
        double min_compute = *std::min_element(all_compute_time.begin(), all_compute_time.end());
        double avg_compute = std::accumulate(all_compute_time.begin(), all_compute_time.end(), 0.0) / size;
        
        double time_imbalance = (avg_compute > 0.001) ? (max_compute - min_compute) / avg_compute * 100.0 : 0.0;
        
        // 计算并行效率
        double total_compute_sum = std::accumulate(all_compute_time.begin(), all_compute_time.end(), 0.0);
        double max_total_time = *std::max_element(all_total_time.begin(), all_total_time.end());
        double parallel_efficiency = (max_total_time > 0.001) ? (total_compute_sum / size) / max_total_time * 100.0 : 0.0;
        
        // 计算通信开销比例
        double total_comm_sum = std::accumulate(all_comm_time.begin(), all_comm_time.end(), 0.0);
        double comm_overhead = (max_total_time > 0.001) ? (total_comm_sum / size) / max_total_time * 100.0 : 0.0;
        
        // 输出性能指标
        std::cout << "\n=== Performance Analysis ===\n";
        std::cout << "Total blocks: " << total_blocks << "\n";
        std::cout << "Load Imbalance Rate: " << std::fixed << std::setprecision(2) << work_imbalance << "%\n";
        std::cout << "Time Imbalance Rate: " << time_imbalance << "%\n";
        std::cout << "Parallel Efficiency: " << parallel_efficiency << "%\n";
        std::cout << "Communication Overhead: " << comm_overhead << "%\n";
        
        std::cout << "\nPer-Process Statistics:\n";
        std::cout << "Rank\tWork Units\tCompute(ms)\tComm(ms)\tTotal(ms)\n";
        for (int i = 0; i < size; i++) {
            std::cout << i << "\t" << all_work[i] << "\t\t" 
                     << std::fixed << std::setprecision(2) 
                     << all_compute_time[i] << "\t\t"
                     << all_comm_time[i] << "\t\t"
                     << all_total_time[i] << "\n";
        }
        
        // 计算负载平衡指标
        double work_variance = 0.0;
        for (int i = 0; i < size; i++) {
            double diff = all_work[i] - avg_work;
            work_variance += diff * diff;
        }
        work_variance /= size;
        double work_std_dev = sqrt(work_variance);
        double work_coefficient_variation = (avg_work > 0) ? work_std_dev / avg_work * 100.0 : 0.0;
        
        std::cout << "\nLoad Balance Metrics:\n";
        std::cout << "Work Standard Deviation: " << work_std_dev << "\n";
        std::cout << "Work Coefficient of Variation: " << work_coefficient_variation << "%\n";
        std::cout << "==============================\n";
    }
}

/* ---------- 策略选择 ---------- */
static void poly_mul_adaptive(int32_t* a,int32_t* b,int32_t* c,
                              int n,int p,MPI_Comm comm)
{
    int rank,size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);

    // 重置性能统计
    perf_stats = PerformanceStats();

    setBarrett(p);
    int m=1; while (m<2*n-1) m<<=1;

    // Ensure arrays are large enough
    if (m > MAX_N) {
        if (rank == 0) {
            std::cerr << "Error: m=" << m << " exceeds MAX_N=" << MAX_N << std::endl;
        }
        MPI_Abort(comm, 1);
        return;
    }

    precompute_roots(m,p);
    
    // Initialize padding with zeros
    for (int i=n;i<m;++i) a[i]=b[i]=0;

    bool use_mpi = (m >= 8192*size);

    if (!use_mpi || size==1) {         /* 纯本地 / 单进程 */
        if (rank==0) {
            ntt_local(a,m,1);  ntt_local(b,m,1);
            
            auto pointwise_start = std::chrono::high_resolution_clock::now();
            // SIMD pointwise multiplication
            #pragma omp parallel for
            for (int i=0;i<m;i+=8) {
#if HAS_AVX2
                if (i+7<m) {
                    pointwise_mul8(c+i, a+i, b+i);
                } else {
#endif
                    for (int j=i;j<m;++j)
                        c[j]= (int32_t)barrett_mul_64(a[j],b[j],bar);
#if HAS_AVX2
                }
#endif
            }
            auto pointwise_end = std::chrono::high_resolution_clock::now();
            perf_stats.compute_time += std::chrono::duration<double, std::milli>(pointwise_end-pointwise_start).count();
            
            ntt_local(c,m,-1);
            int32_t inv_m = mod_pow(m,p-2,p);
            
            auto norm_start = std::chrono::high_resolution_clock::now();
            // SIMD normalization
            #pragma omp parallel for
            for (int i=0;i<m;i+=8) {
#if HAS_AVX2
                if (i+7<m) {
                    scalar_mul8(c+i, inv_m);
                } else {
#endif
                    for (int j=i;j<m;++j)
                        c[j]= (int32_t)barrett_mul_64(c[j],inv_m,bar);
#if HAS_AVX2
                }
#endif
            }
            auto norm_end = std::chrono::high_resolution_clock::now();
            perf_stats.compute_time += std::chrono::duration<double, std::milli>(norm_end-norm_start).count();
        }
        auto bcast_start = std::chrono::high_resolution_clock::now();
        MPI_Bcast(c,m,MPI_INT,0,comm);
        auto bcast_end = std::chrono::high_resolution_clock::now();
        perf_stats.comm_time += std::chrono::duration<double, std::milli>(bcast_end-bcast_start).count();
    }
    else {                              /* 混合并行 */
        ntt_hybrid(a,m,1,comm);
        ntt_hybrid(b,m,1,comm);

        if (rank==0) {
            auto pointwise_start = std::chrono::high_resolution_clock::now();
            // SIMD pointwise multiplication
            #pragma omp parallel for
            for (int i=0;i<m;i+=8) {
#if HAS_AVX2
                if (i+7<m) {
                    pointwise_mul8(c+i, a+i, b+i);
                } else {
#endif
                    for (int j=i;j<m;++j)
                        c[j]=(int32_t)barrett_mul_64(a[j],b[j],bar);
#if HAS_AVX2
                }
#endif
            }
            auto pointwise_end = std::chrono::high_resolution_clock::now();
            perf_stats.compute_time += std::chrono::duration<double, std::milli>(pointwise_end-pointwise_start).count();
        }
        auto bcast_start = std::chrono::high_resolution_clock::now();
        MPI_Bcast(c,m,MPI_INT,0,comm);
        auto bcast_end = std::chrono::high_resolution_clock::now();
        perf_stats.comm_time += std::chrono::duration<double, std::milli>(bcast_end-bcast_start).count();

        ntt_hybrid(c,m,-1,comm);
        if (rank==0) {
            int32_t inv_m=mod_pow(m,p-2,p);
            auto norm_start = std::chrono::high_resolution_clock::now();
            // SIMD normalization
            #pragma omp parallel for
            for (int i=0;i<m;i+=8) {
#if HAS_AVX2
                if (i+7<m) {
                    scalar_mul8(c+i, inv_m);
                } else {
#endif
                    for (int j=i;j<m;++j)
                        c[j]=(int32_t)barrett_mul_64(c[j],inv_m,bar);
#if HAS_AVX2
                }
#endif
            }
            auto norm_end = std::chrono::high_resolution_clock::now();
            perf_stats.compute_time += std::chrono::duration<double, std::milli>(norm_end-norm_start).count();
        }
        auto bcast_start2 = std::chrono::high_resolution_clock::now();
        MPI_Bcast(c,m,MPI_INT,0,comm);
        auto bcast_end2 = std::chrono::high_resolution_clock::now();
        perf_stats.comm_time += std::chrono::duration<double, std::milli>(bcast_end2-bcast_start2).count();
    }
}

/* ---------- 文件I/O函数 ---------- */
void fRead(int32_t *a, int32_t *b, int *n, int *p, int input_id){
    std::string str1 = "C:/Users/86180/Downloads/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    
    std::ifstream fin(strin);
    if(!fin.is_open()) {
        std::cerr << "Failed to open " << strin << std::endl;
        return;
    }
    
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++){
        fin >> a[i];
    }
    for (int i = 0; i < *n; i++){
        fin >> b[i];
    }
    fin.close();
}

void fCheck(int32_t *ab, int n, int input_id){
    std::string str1 = "C:/Users/86180/Downloads/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    
    std::ifstream fin(strout);
    if(!fin.is_open()) {
        std::cerr << "Failed to open " << strout << std::endl;
        return;
    }
    
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin >> x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            fin.close();
            return;
        }
    }
    fin.close();
    std::cout<<"多项式乘法结果正确"<<std::endl;
}

void fWrite(int32_t *ab, int n, int input_id){
    std::string str1 = "C:/Users/86180/Downloads/nttdata/files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    
    std::ofstream fout(strout);
    if(!fout.is_open()) {
        std::cerr << "Failed to open " << strout << std::endl;
        return;
    }
    
    for (int i = 0; i < n * 2 - 1; i++){
        fout << ab[i] << '\n';
    }
    fout.close();
}

/* ---------- main ---------- */
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    int rank,size; 
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // 每个进程报告自己的信息
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    printf("Process %d/%d running on %s\n", rank, size, processor_name);

#ifdef _OPENMP
    omp_set_num_threads(8);  // 设置OpenMP线程数
    int num_threads = omp_get_max_threads();
    printf("Process %d: Using %d OpenMP threads\n", rank, num_threads);
    
    if (rank==0)
        std::cout<<"Total: MPI "<<size<<"  OpenMP "<<num_threads
                 <<"  AVX2 "<<HAS_AVX2<<"\n";
#endif

    // 确保所有进程都输出完信息后再继续
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int32_t> a(MAX_N,0),b(MAX_N,0),c(MAX_N,0);
    
    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n,p;
        
        if (rank == 0) {
            printf("=== Test case %d ===\n", i);
        }
        
        // 读取数据
        if (rank == 0) {
            fRead(a.data(), b.data(), &n, &p, i);
            printf("Process %d: Read n=%d, p=%d\n", rank, n, p);
        }
        
        // 广播数据大小
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 每个进程确认收到数据
        if (rank != 0) {
            printf("Process %d: Received n=%d, p=%d\n", rank, n, p);
        }
        
        // 计算所需大小
        int m = 1;
        while (m < 2*n-1) m <<= 1;
        
        // 检查是否需要广播输入数据
        bool need_broadcast = (m >= 8192 * size) && (size > 1);
        
        if(need_broadcast) {
            if (rank == 0) {
                printf("Broadcasting input data to all processes\n");
            }
            MPI_Bcast(a.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(b.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        poly_mul_adaptive(a.data(), b.data(), c.data(), n, p, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(t1-t0).count();
        
        if (rank==0) {
            fCheck(c.data(), n, i);
            std::cout << "Adaptive MPI+OpenMP+AVX2"
                      << " (strategy: " << (need_broadcast ? "MPI" : "Local") << ")"
                      << " : n=" << n
                      << " p=" << p << " latency=" << total_time
                      << " ms\n";
            fWrite(c.data(), n, i);
        }
        
        // 性能分析（仅对使用MPI的情况）
        if (need_broadcast && size > 1) {
            // 计算总工作量（估算）
            int total_blocks = 0;
            const int MPI_THRESHOLD = m / (4*size);
            for (int mid = MPI_THRESHOLD; mid < m; mid <<= 1) {
                total_blocks += m / (mid << 1);
            }
            // 3次NTT调用的总工作量
            total_blocks *= 3;
            
            analyze_load_balance(total_blocks, total_time, MPI_COMM_WORLD);
        }
        
        printf("Process %d: Completed test case %d\n", rank, i);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    printf("Process %d: Finalizing\n", rank);
    MPI_Finalize();
    return 0;
}