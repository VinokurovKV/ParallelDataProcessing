#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define Max(a,b) ((a)>(b)?(a):(b))

#define N (2*2*2*2*2*2*2*2*2*2+2)
double maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
double A[N][N], B[N][N];

void relax();
void resid();
void init();
void verify();

int main(int an, char **as)
{
    int it;
    double start_time = omp_get_wtime();
    
    init();
    for(it=1; it<=itmax; it++)
    {
        eps = 0.;
        relax();
        resid();
        printf("it=%4i   eps=%f\n", it,eps);
        if (eps < maxeps) break;
    }
    verify();
    
    double end_time = omp_get_wtime();
    printf("Total time: %f seconds\n", end_time - start_time);
    return 0;
}

void init()
{ 
    #pragma omp parallel for private(j)
    for(i=0; i<N; i++)
    for(j=0; j<N; j++)
    {
        if(i==0 || i==N-1 || j==0 || j==N-1) 
            A[i][j] = 0.;
        else 
            A[i][j] = (1. + i + j);
    }
} 

void relax()
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            int chunk_size = 8; 
            if (chunk_size > N-4) chunk_size = N-4;
            
            for(i=2; i<=N-3; i += chunk_size)
            {
                int i_end = i + chunk_size;
                if (i_end > N-2) i_end = N-2;
                
                #pragma omp task firstprivate(i, i_end)
                {
                    for(int ii = i; ii < i_end; ii++)
                    {
                        double* A_ii_minus_2 = A[ii-2];
                        double* A_ii_minus_1 = A[ii-1];
                        double* A_ii_plus_1 = A[ii+1];
                        double* A_ii_plus_2 = A[ii+2];
                        double* A_ii = A[ii];
                        double* B_ii = B[ii];
                        
                        for(int jj = 2; jj <= N-3; jj++)
                        {
                            B_ii[jj] = (A_ii_minus_2[jj] + A_ii_minus_1[jj] + 
                                       A_ii_plus_2[jj] + A_ii_plus_1[jj] +
                                       A_ii[jj-2] + A_ii[jj-1] + 
                                       A_ii[jj+2] + A_ii[jj+1]) * 0.125;
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
    }
}

void resid()
{ 
    double local_eps = 0.;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            int chunk_size = 8;
            if (chunk_size > N-2) chunk_size = N-2;
            
            for(i=1; i<=N-2; i += chunk_size)
            {
                int i_end = i + chunk_size;
                if (i_end > N-1) i_end = N-1;
                
                #pragma omp task firstprivate(i, i_end) shared(local_eps)
                {
                    double thread_eps = 0.;
                    
                    for(int ii = i; ii < i_end; ii++)
                    {
                        double* A_ii = A[ii];
                        double* B_ii = B[ii];
                        
                        for(int jj = 1; jj <= N-2; jj++)
                        {
                            double e = fabs(A_ii[jj] - B_ii[jj]);         
                            A_ii[jj] = B_ii[jj]; 
                            if (e > thread_eps) thread_eps = e;
                        }
                    }
                    
                    #pragma omp critical
                    {
                        if (thread_eps > local_eps) local_eps = thread_eps;
                    }
                }
            }
            #pragma omp taskwait
        }
    }
    
    eps = local_eps;
}

void verify()
{
    double s = 0.;
    double inv_n2 = 1.0 / (N * N);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            int chunk_size = 8;
            if (chunk_size > N) chunk_size = N;
            
            for(i=0; i<N; i += chunk_size)
            {
                int i_end = i + chunk_size;
                if (i_end > N) i_end = N;
                
                #pragma omp task firstprivate(i, i_end) shared(s)
                {
                    double local_s = 0.;
                    
                    for(int ii = i; ii < i_end; ii++)
                    {
                        double* A_ii = A[ii];
                        double i_plus_1 = ii + 1;
                        
                        for(int jj = 0; jj < N; jj++)
                        {
                            local_s += A_ii[jj] * i_plus_1 * (jj + 1) * inv_n2;
                        }
                    }
                    
                    #pragma omp atomic
                    s += local_s;
                }
            }
            #pragma omp taskwait
        }
    }
    printf("S = %f\n", s);
}