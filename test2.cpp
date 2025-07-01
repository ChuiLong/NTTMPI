#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <bits/stdc++.h>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

typedef long long int ll;

// ---------------------- Barrett 取模相关 ----------------------
struct Barrett {
    ll p;
    unsigned long long mul;
};

Barrett getBarrett(ll p) {
    Barrett bar;
    bar.p = p;
    __uint128_t tmp = ((__uint128_t)1 << 64) / p;
    bar.mul = (unsigned long long)(tmp + 1);
    return bar;
}

inline ll barrett_mul_64(ll a, ll b, const Barrett &bar) {
    __uint128_t z = ( __uint128_t ) a * b;
    __uint128_t t = (z * bar.mul) >> 64; 
    ll ret = (ll)(z - t * bar.p);
    if(ret < 0)     ret += bar.p;
    if(ret >= bar.p)ret -= bar.p;
    return ret;
}

inline ll barrett_add_64(ll a, ll b, const Barrett &bar) {
    ll ret = a + b;
    if(ret >= bar.p) ret -= bar.p;
    return ret;
}

inline ll barrett_sub_64(ll a, ll b, const Barrett &bar) {
    ll ret = a - b;
    if(ret < 0) ret += bar.p;
    return ret;
}

ll barrett_pow_64(ll base, ll exp, const Barrett &bar){
    ll ans = 1 % bar.p;
    ll cur = (base % bar.p + bar.p) % bar.p;
    while(exp > 0){
        if(exp & 1) ans = barrett_mul_64(ans, cur, bar);
        cur = barrett_mul_64(cur, cur, bar);
        exp >>= 1;
    }
    return ans;
}

// ---------------------- 全局变量 ----------------------
static const int g = 3;
static const int MAX_N = 300000;
static Barrett bar;
static int currentP;

// 单位根缓存
static std::vector<std::vector<ll>> w_cache;
static std::vector<std::vector<ll>> w_inv_cache;

void setBarrett(int p) {
    currentP = p;
    bar = getBarrett((ll)p);
}

int pow(int x,int y,int p) {
    ll ans = 1;
    ll base = (ll)( (x % p + p) % p );
    while(y > 0){
        if(y & 1) ans = barrett_mul_64(ans, base, bar);
        base = barrett_mul_64(base, base, bar);
        y >>= 1;
    }
    return (int)ans;
}

// 预计算单位根
void precompute_roots(int max_n, int p) {
    int max_level = __builtin_ctz(max_n);
    w_cache.clear();
    w_inv_cache.clear();
    w_cache.resize(max_level + 1);
    w_inv_cache.resize(max_level + 1);
    
    for(int level = 0, mid = 1; mid < max_n; mid <<= 1, ++level) {
        int blkSize = mid << 1;
        ll w = barrett_pow_64(g, (p - 1) / blkSize, bar);
        ll w_inv = barrett_pow_64(w, p - 2, bar);
        
        w_cache[level].resize(mid);
        w_inv_cache[level].resize(mid);
        
        ll omega = 1, omega_inv = 1;
        for(int j = 0; j < mid; ++j) {
            w_cache[level][j] = omega;
            w_inv_cache[level][j] = omega_inv;
            omega = barrett_mul_64(omega, w, bar);
            omega_inv = barrett_mul_64(omega_inv, w_inv, bar);
        }
    }
}

// bit-reverse 操作
void bit_reverse_array(int *a, int n) {
    int lg = __builtin_ctz(n);
    std::vector<int> rev(n);
    
    for(int i = 0; i < n; ++i) {
        rev[i] = (rev[i>>1]>>1) | ((i&1)<<(lg-1));
    }
    
    for(int i = 0; i < n; ++i) {
        if(i < rev[i]) {
            std::swap(a[i], a[rev[i]]);
        }
    }
}

// 策略1：完全本地计算的NTT（适用于小规模）
void ntt_local(int *a, int n, int inv) {
    bit_reverse_array(a, n);
    
    for(int mid = 1, level = 0; mid < n; mid <<= 1, ++level) {
        const ll *omega_table = (inv == 1) ? 
            w_cache[level].data() : w_inv_cache[level].data();
        
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i += 2 * mid) {
            for(int j = 0; j < mid; ++j) {
                ll x = a[i + j];
                ll y = barrett_mul_64(omega_table[j], a[i + j + mid], bar);
                a[i + j] = (int)barrett_add_64(x, y, bar);
                a[i + j + mid] = (int)barrett_sub_64(x, y, bar);
            }
        }
    }
}

// 策略2：混合并行NTT（只在高层使用MPI）
void ntt_hybrid_optimized(int *a, int n, int inv, MPI_Comm comm) {
    int rank, size; 
    MPI_Comm_rank(comm, &rank); 
    MPI_Comm_size(comm, &size);
    
    // 定义通信阈值：只有当块大小超过阈值时才使用MPI
    const int MPI_THRESHOLD = n / (4 * size);  // 自适应阈值
    
    // Step 1: bit-reverse
    if(rank == 0) {
        bit_reverse_array(a, n);
    }
    MPI_Bcast(a, n, MPI_INT, 0, comm);
    
    // Step 2: 低层使用本地计算
    int local_level = 0;
    for(int mid = 1; mid < MPI_THRESHOLD && mid < n; mid <<= 1, ++local_level) {
        const ll *omega_table = (inv == 1) ? 
            w_cache[local_level].data() : w_inv_cache[local_level].data();
        
        // 所有进程都执行完整计算（数据冗余但避免通信）
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i += 2 * mid) {
            for(int j = 0; j < mid; ++j) {
                ll x = a[i + j];
                ll y = barrett_mul_64(omega_table[j], a[i + j + mid], bar);
                a[i + j] = (int)barrett_add_64(x, y, bar);
                a[i + j + mid] = (int)barrett_sub_64(x, y, bar);
            }
        }
    }
    
    // Step 3: 高层使用MPI并行
    std::vector<int> buf(n);
    int *cur = a;
    int *nxt = buf.data();
    
    for(int mid = MPI_THRESHOLD, level = local_level; mid < n; mid <<= 1, ++level) {
        if(mid < MPI_THRESHOLD) continue;
        
        int blkSize = mid << 1;
        int numBlk = n / blkSize;
        int blk_beg = rank * numBlk / size;
        int blk_end = (rank + 1) * numBlk / size;
        
        const ll *omega_table = (inv == 1) ? 
            w_cache[level].data() : w_inv_cache[level].data();
        
        // 本地计算
        #pragma omp parallel for schedule(dynamic)
        for(int blk = blk_beg; blk < blk_end; ++blk) {
            int base = blk * blkSize;
            for(int j = 0; j < mid; ++j) {
                ll x = cur[base + j];
                ll y = barrett_mul_64(omega_table[j], cur[base + j + mid], bar);
                cur[base + j] = (int)barrett_add_64(x, y, bar);
                cur[base + j + mid] = (int)barrett_sub_64(x, y, bar);
            }
        }
        
        // 只在必要时通信
        if(size > 1) {
            std::vector<int> cnt(size), disp(size);
            for(int r = 0; r < size; ++r) {
                int s = r * numBlk / size;
                int e = (r + 1) * numBlk / size;
                cnt[r] = (e - s) * blkSize;
                disp[r] = s * blkSize;
            }
            
            MPI_Allgatherv(cur + blk_beg * blkSize, cnt[rank], MPI_INT,
                           nxt, cnt.data(), disp.data(), MPI_INT, comm);
            std::swap(cur, nxt);
        }
    }
    
    // 拷贝结果
    if(cur != a) {
        std::memcpy(a, cur, n * sizeof(int));
    }
}

// 策略3：根据问题规模自动选择算法
void poly_multiply_adaptive(int *a, int *b, int *ab, int n, int p, MPI_Comm comm) {
    int rank, size; 
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    setBarrett(p);
    
    int m = 1; 
    while (m < 2*n - 1) m <<= 1;
    
    // 预计算单位根
    precompute_roots(m, p);
    
    // 清零扩展部分
    for (int i = n; i < m; ++i) { 
        a[i] = b[i] = 0; 
    }
    
    // 根据规模选择策略
    bool use_mpi = (m >= 8192 * size);  // 只有大规模问题才使用MPI
    
    if(!use_mpi || size == 1) {
        // 小规模或单进程：使用纯OpenMP
        if(rank == 0) {
            ntt_local(a, m, 1);
            ntt_local(b, m, 1);
            
            #pragma omp parallel for schedule(static)
            for(int i = 0; i < m; ++i) {
                ab[i] = (int)barrett_mul_64((ll)a[i], (ll)b[i], bar);
            }
            
            ntt_local(ab, m, -1);
            
            int inv_m = pow(m, p - 2, p);
            #pragma omp parallel for schedule(static)
            for(int i = 0; i < m; ++i) {
                ab[i] = (int)barrett_mul_64((ll)ab[i], (ll)inv_m, bar);
            }
        }
        // 广播结果
        MPI_Bcast(ab, m, MPI_INT, 0, comm);
    } else {
        // 大规模：使用混合并行
        ntt_hybrid_optimized(a, m, 1, comm);
        ntt_hybrid_optimized(b, m, 1, comm);
        
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for(int i = 0; i < m; ++i) {
                ab[i] = (int)barrett_mul_64((ll)a[i], (ll)b[i], bar);
            }
        }
        MPI_Bcast(ab, m, MPI_INT, 0, comm);
        
        ntt_hybrid_optimized(ab, m, -1, comm);
        
        if (rank == 0) {
            int inv_m = pow(m, p - 2, p);
            #pragma omp parallel for schedule(static)
            for(int i = 0; i < m; ++i) {
                ab[i] = (int)barrett_mul_64((ll)ab[i], (ll)inv_m, bar);
            }
        }
        MPI_Bcast(ab, m, MPI_INT, 0, comm);
    }
}

// 文件I/O函数
void fRead(int *a, int *b, int *n, int *p, int input_id){
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

void fCheck(int *ab, int n, int input_id){
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

void fWrite(int *ab, int n, int input_id){
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

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 设置OpenMP线程数
    #ifdef _OPENMP
    omp_set_num_threads(8);
    int num_threads = 8;
    if(rank == 0) {
        std::cout << "Using " << size << " MPI processes with " 
                  << num_threads << " OpenMP threads each" << std::endl;
    }
    #endif

    // 使用动态分配
    std::vector<int> a(MAX_N), b(MAX_N), ab(MAX_N);

    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n_, p_;
        
        // 读取数据
        if (rank == 0) {
            fRead(a.data(), b.data(), &n_, &p_, i);
        }
        
        // 广播数据大小
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 检查是否需要广播输入数据
        int m = 1;
        while (m < 2*n_ - 1) m <<= 1;
        bool need_broadcast = (m >= 8192 * size) && (size > 1);
        
        if(need_broadcast) {
            MPI_Bcast(a.data(), n_, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(b.data(), n_, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // 计时并执行
        MPI_Barrier(MPI_COMM_WORLD);
        auto Start = std::chrono::high_resolution_clock::now();
        
        poly_multiply_adaptive(a.data(), b.data(), ab.data(), n_, p_, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto End = std::chrono::high_resolution_clock::now();

        // 输出结果
        if (rank == 0) {
            fCheck(ab.data(), n_, i);
            std::chrono::duration<double, std::milli> elap = End - Start;
            std::cout << "Adaptive MPI+OpenMP"
                      << " (strategy: " << (need_broadcast ? "MPI" : "Local") << ")"
                      << " : n=" << n_
                      << " p=" << p_ << " latency=" << elap.count()
                      << " ms\n";
            fWrite(ab.data(), n_, i);
        }
    }

    MPI_Finalize();
    return 0;
}