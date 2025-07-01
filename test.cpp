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
    ll p;                   // 模数
    unsigned long long mul; // 用于快速约减的常数
};

Barrett getBarrett(ll p) {
    Barrett bar;
    bar.p = p;
    // 计算 (1 << 64) / p 在 128 位下
    __uint128_t tmp = ((__uint128_t)1 << 64) / p;
    bar.mul = (unsigned long long)(tmp + 1);
    return bar;
}

// Barrett 乘法: 计算 (a * b) mod p
ll barrett_mul_64(ll a, ll b, const Barrett &bar) {
    __uint128_t z = ( __uint128_t ) a * b;
    __uint128_t t = (z * bar.mul) >> 64; 
    ll ret = (ll)(z - t * bar.p);
    // 做一次或两次修正，确保结果在 [0, p) 内
    if(ret < 0)     ret += bar.p;
    if(ret >= bar.p)ret -= bar.p;
    return ret;
}

// Barrett 加法: (a + b) mod p
ll barrett_add_64(ll a, ll b, const Barrett &bar) {
    ll ret = a + b;
    if(ret >= bar.p) ret -= bar.p;
    return ret;
}

// Barrett 减法: (a - b) mod p
ll barrett_sub_64(ll a, ll b, const Barrett &bar) {
    ll ret = a - b;
    if(ret < 0) ret += bar.p;
    return ret;
}

// 幂取模 (使用 Barrett)
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

// 计算 x 在模 p 下的逆元 (p 为素数)
ll barrett_inv_64(ll x, const Barrett &bar) {
    // 费马小定理: x^(p-2) mod p
    return barrett_pow_64(x, bar.p - 2, bar);
}

// ---------------------- 全局 / 静态变量 ----------------------
static const int g = 3;           // 原根
static ll rev[300000] = {0};      // 用于 bit-reverse 排序
static Barrett bar;               // 全局 Barrett 结构
static int currentP;              // 记录当前 p，方便在 setBarrett 中更新

// ---------------------- 与 Barrett 整合的函数 ----------------------
// 设置全局的 Barrett 结构和 currentP
void setBarrett(int p) {
    currentP = p;
    bar = getBarrett((ll)p);
}

// 用 Barrett 版的 pow 来替换原先的快速幂
int pow(int x,int y,int p) // 注意：函数签名不变
{
    // 这里直接使用 barrett_pow_64
    // 需确保已经调用过 setBarrett(p)
    ll ans = 1;
    ll base = (ll)( (x % p + p) % p );
    while(y > 0){
        if(y & 1) ans = barrett_mul_64(ans, base, bar);
        base = barrett_mul_64(base, base, bar);
        y >>= 1;
    }
    return (int)ans;
}

// ntt 函数：用 Barrett 取代所有 mod 运算
void ntt_mpi(int *a, int n, int p, int inv, MPI_Comm comm)
{
    int rank, size; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);

    // ---- 0) bit‑reverse 排序（同原来） ----
    if(rank==0){
        int lg=__builtin_ctz(n);
        for(int i=0;i<n;++i){
            rev[i]=(rev[i>>1]>>1)|((i&1)<<(lg-1));
            if(i<rev[i]) std::swap(a[i],a[rev[i]]);
        }
    }
    MPI_Bcast(a,n,MPI_INT,0,comm);

    // ---- 1) 准备双缓冲 ----
    std::vector<int> buf(n);           // shadow buffer
    int *cur = a;                      // 当前层数据指针
    int *nxt = buf.data();             // 下一层数据指针

    // ---- 2) 蝴蝶层循环 ----
    for(int mid=1,level=0; mid<n; mid<<=1,++level)
    {
        int blkSize = mid<<1;
        int numBlk  = n/blkSize;
        int blk_beg = rank   *numBlk/size;
        int blk_end = (rank+1)*numBlk/size;

        // 单位根 (仍可预表，这里保持简单)
        int w = pow(g,(p-1)/blkSize,p);
        if(inv==-1) w = pow(w,p-2,p);

        // ---- 2.1 本地蝴蝶到 *cur* ----
        for(int blk=blk_beg; blk<blk_end; ++blk){
            int base = blk*blkSize;
            ll omega = 1;
            for(int j=0;j<mid;++j){
                ll x=cur[base+j];
                ll y=barrett_mul_64(omega,cur[base+j+mid],bar);
                cur[base+j]       = (int)barrett_add_64(x,y,bar);
                cur[base+j+mid]   = (int)barrett_sub_64(x,y,bar);
                omega = barrett_mul_64(omega,(ll)w,bar);
            }
        }

        // ---- 2.2 非阻塞收集 ----
        std::vector<int> cnt(size), disp(size);
        for(int r=0;r<size;++r){
            int s=r*numBlk/size, e=(r+1)*numBlk/size;
            cnt [r]=(e-s)*blkSize;
            disp[r]= s   *blkSize;
        }
        MPI_Request req;
        MPI_Iallgatherv(cur+blk_beg*blkSize, cnt[rank], MPI_INT,
                        nxt, cnt.data(), disp.data(), MPI_INT,
                        comm, &req);


        MPI_Wait(&req, MPI_STATUS_IGNORE);     // 等待通信完成

        std::swap(cur,nxt);                    // 下一层读写指针互换
    }

    // 如果最后结果不在 a[] 里就拷回
    if(cur!=a) std::memcpy(a,cur,n*sizeof(int));
}

// ------------- 并行多项式乘 -------------
void poly_multiply_mpi(int *a, int *b, int *ab, int n, int p, MPI_Comm comm)
{
    setBarrett(p);                     // 各进程各自调用

    int m = 1; while (m < 2*n - 1) m <<= 1;
    for (int i = n; i < m; ++i) { a[i] = b[i] = 0; }

    ntt_mpi(a,  m, p,  1, comm);
    ntt_mpi(b,  m, p,  1, comm);

    // 点乘：数据已完整同步，任选一个 rank 做即可，或全部做结果相同
    int rank; MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        for (int i = 0; i < m; ++i) {
            ab[i] = (int)barrett_mul_64((ll)a[i], (ll)b[i], bar);
        }
    }
    MPI_Bcast(ab, m, MPI_INT, 0, comm);

    ntt_mpi(ab, m, p, -1, comm);
    if (rank == 0) {
        int inv_m = pow(m, p - 2, p);
        for (int i = 0; i < m; ++i)
            ab[i] = (int)barrett_mul_64((ll)ab[i], (ll)inv_m, bar);
    }
}

void fRead(int *a, int *b, int *n, int *p, int input_id){
    std::string str1 = "C:/Users/86180/Downloads/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';

    std::ifstream fin;
    fin.open(data_path, std::ios::in);
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
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';

    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin >> x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误fffff"<<std::endl;
            fin.close();
            return;
        }
    }
    fin.close();
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    std::string str1 = "C:/Users/86180/Downloads/nttdata/files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';

    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout << ab[i] << '\n';
    }
    fout.close();
}

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n_, p_;
        if (rank == 0) fRead(a, b, &n_, &p_, i);
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_INT, 0, MPI_COMM_WORLD);

        auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply_mpi(a, b, ab, n_, p_, MPI_COMM_WORLD);
        auto End   = std::chrono::high_resolution_clock::now();

        if (rank == 0) {
            fCheck(ab, n_, i);
            std::chrono::duration<double, std::milli> elap = End - Start;
            std::cout << "MPI "  << " : n=" << n_
                      << " p=" << p_ << " latency=" << elap.count()
                      << " ms\n";
            fWrite(ab, n_, i);
        }
    }

    MPI_Finalize();
    return 0;
}
