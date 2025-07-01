/*  ntt_simd.cpp  ——  MPI + OpenMP + AVX2 SIMD Barrett‑NTT  */

#include <bits/stdc++.h>
#include <immintrin.h>          // AVX / AVX2
#include <omp.h>
#include <mpi.h>

using std::int32_t;
using std::int64_t;
using std::uint64_t;

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

/* Simplified SIMD butterfly operation for AVX2 */
static inline void butterfly8_v2(int32_t* x_ptr, int32_t* y_ptr,
                                 const int32_t* omega_ptr)
{
    __m256i vx = _mm256_loadu_si256((__m256i*)x_ptr);
    __m256i vy = _mm256_loadu_si256((__m256i*)y_ptr);
    __m256i vomega = _mm256_loadu_si256((__m256i*)omega_ptr);
    __m256i vp = _mm256_set1_epi32((int)bar.p);
    
    // Process in two halves to avoid AVX512 requirements
    int32_t y_arr[8], omega_arr[8], result[8];
    _mm256_storeu_si256((__m256i*)y_arr, vy);
    _mm256_storeu_si256((__m256i*)omega_arr, vomega);
    
    // Perform Barrett multiplication on each element
    for (int i = 0; i < 8; i++) {
        result[i] = (int32_t)barrett_mul_64(y_arr[i], omega_arr[i], bar);
    }
    
    __m256i vy_mod = _mm256_loadu_si256((__m256i*)result);
    
    // Butterfly operations
    __m256i vadd = _mm256_add_epi32(vx, vy_mod);
    __m256i vsub = _mm256_sub_epi32(vx, vy_mod);
    
    // Conditional reductions using masks
    __m256i mask_add_ge = _mm256_cmpgt_epi32(vadd, _mm256_sub_epi32(vp, _mm256_set1_epi32(1)));
    __m256i mask_add_neg = _mm256_srai_epi32(vadd, 31); // Sign bit
    vadd = _mm256_sub_epi32(vadd, _mm256_and_si256(mask_add_ge, vp));
    vadd = _mm256_add_epi32(vadd, _mm256_and_si256(mask_add_neg, vp));
    
    __m256i mask_sub = _mm256_srai_epi32(vsub, 31); // Sign bit for negative check
    vsub = _mm256_add_epi32(vsub, _mm256_and_si256(mask_sub, vp));
    
    _mm256_storeu_si256((__m256i*)x_ptr, vadd);
    _mm256_storeu_si256((__m256i*)y_ptr, vsub);
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
}

/* ---------- MPI + OpenMP 混合 NTT (仅高层通信) ---------- */
static void ntt_hybrid(int32_t* a,int n,int inv,MPI_Comm comm)
{
    int rank,size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);

    const int MPI_THRESHOLD = n / (4*size);   // 块阈值

    if (rank==0) bit_reverse(a,n);
    MPI_Bcast(a,n,MPI_INT,0,comm);

    int local_lvl = 0;
    /* ---------- 低层：全部本地 ---------- */
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

    /* ---------- 高层：分块 + Allgatherv ---------- */
    std::vector<int32_t> buf(n);
    int32_t* cur=a; int32_t* nxt=buf.data();

    for (int mid=MPI_THRESHOLD,lvl=local_lvl; mid<n; mid<<=1,++lvl)
    {
        int blkSize=mid<<1, numBlk=n/blkSize;
        int blk_beg= rank   *numBlk/size;
        int blk_end=(rank+1)*numBlk/size;

        const int32_t* omega = (inv==1? w_cache[lvl]:w_inv_cache[lvl]).data();

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

        /* 汇总高层结果 */
        std::vector<int> cnt(size), disp(size);
        for (int r=0;r<size;++r) {
            int s=r*numBlk/size, e=(r+1)*numBlk/size;
            cnt [r]=(e-s)*blkSize;
            disp[r]= s   *blkSize;
        }
        MPI_Allgatherv(cur+blk_beg*blkSize, cnt[rank], MPI_INT,
                       nxt, cnt.data(), disp.data(),
                       MPI_INT, comm);
        std::swap(cur,nxt);
    }
    if (cur!=a) std::memcpy(a,cur,n*sizeof(int32_t));
}

/* ---------- 策略选择 ---------- */
static void poly_mul_adaptive(int32_t* a,int32_t* b,int32_t* c,
                              int n,int p,MPI_Comm comm)
{
    int rank,size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);

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
            
            ntt_local(c,m,-1);
            int32_t inv_m = mod_pow(m,p-2,p);
            
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
        }
        MPI_Bcast(c,m,MPI_INT,0,comm);
    }
    else {                              /* 混合并行 */
        ntt_hybrid(a,m,1,comm);
        ntt_hybrid(b,m,1,comm);

        if (rank==0) {
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
        }
        MPI_Bcast(c,m,MPI_INT,0,comm);

        ntt_hybrid(c,m,-1,comm);
        if (rank==0) {
            int32_t inv_m=mod_pow(m,p-2,p);
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
        }
        MPI_Bcast(c,m,MPI_INT,0,comm);
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
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

#ifdef _OPENMP
    omp_set_num_threads(2);  // 设置OpenMP线程数
    if (rank==0)
        std::cout<<"MPI "<<size<<"  OpenMP "<<2
                 <<"  AVX2 "<<HAS_AVX2<<"\n";
#endif

    std::vector<int32_t> a(MAX_N,0),b(MAX_N,0),c(MAX_N,0);
    
    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n,p;
        
        // 读取数据
        if (rank == 0) {
            fRead(a.data(), b.data(), &n, &p, i);
        }
        
        // 广播数据大小
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 计算所需大小
        int m = 1;
        while (m < 2*n-1) m <<= 1;
        
        // 检查是否需要广播输入数据
        bool need_broadcast = (m >= 8192 * size) && (size > 1);
        
        if(need_broadcast) {
            MPI_Bcast(a.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(b.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        poly_mul_adaptive(a.data(), b.data(), c.data(), n, p, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        if (rank==0) {
            fCheck(c.data(), n, i);
            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
            std::cout << "Adaptive MPI+OpenMP+AVX2"
                      << " (strategy: " << (need_broadcast ? "MPI" : "Local") << ")"
                      << " : n=" << n
                      << " p=" << p << " latency=" << ms
                      << " ms\n";
            fWrite(c.data(), n, i);
        }
    }
    
    MPI_Finalize();
    return 0;
}