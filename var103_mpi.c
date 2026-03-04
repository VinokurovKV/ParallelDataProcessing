#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define  N   (2*2*2*2*2*2+2)

double maxeps = 0.1e-7;
int itmax = 100;

void relax(double A[N][N], double B[N][N], int rank, int size, int start_row, int end_row);
double resid(double A[N][N], double B[N][N], int rank, int size, int start_row, int end_row);
void init(double A[N][N], int rank, int size, int start_row, int end_row);
void verify(double A[N][N], int rank, int size, int start_row, int end_row);

int main(int argc, char **argv) {
    int rank, size;
    int it;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_rows = N - 2;
    int rows_per_process = total_rows / size;
    int remainder = total_rows % size;

    int start_row, end_row;
    if (rank < remainder) {
        start_row = 1 + rank * (rows_per_process + 1);
        end_row = start_row + rows_per_process + 1;
    } else {
        start_row = 1 + remainder * (rows_per_process + 1) + (rank - remainder) * rows_per_process;
        end_row = start_row + rows_per_process;
    }

    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    double A[N][N];
    double B[N][N];

    init(A, rank, size, start_row, end_row);

    for (it = 1; it <= itmax; it++) {
        relax(A, B, rank, size, start_row, end_row);

        double local_eps = resid(A, B, rank, size, start_row, end_row);
        double global_eps;

        MPI_Allreduce(&local_eps, &global_eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("it=%4i   eps=%f\n", it, global_eps);
        }

        if (global_eps < maxeps) break;
    }

    verify(A, rank, size, start_row, end_row);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Total time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}

void init(double A[N][N], int rank, int size, int start_row, int end_row) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                A[i][j] = 0.0;
            } else if (i >= start_row && i < end_row) {
                A[i][j] = (1.0 + i + j);
            } else {
                A[i][j] = 0.0;
            }
        }
    }
}

void relax(double A[N][N], double B[N][N], int rank, int size, int start_row, int end_row) {
    MPI_Request send_req[2], recv_req[2];
    MPI_Status status[2];

    if (size > 1) {
        if (rank > 0) {
            MPI_Isend(&A[start_row][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_req[0]);
            MPI_Isend(&A[start_row + 1][0], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &send_req[1]);
            MPI_Irecv(&A[start_row - 2][0], N, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &recv_req[0]);
            MPI_Irecv(&A[start_row - 1][0], N, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &recv_req[1]);

            MPI_Waitall(2, send_req, status);
            MPI_Waitall(2, recv_req, status);
        }
        if (rank < size - 1) {
            MPI_Isend(&A[end_row - 2][0], N, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &send_req[0]);
            MPI_Isend(&A[end_row - 1][0], N, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &send_req[1]);
            MPI_Irecv(&A[end_row][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_req[0]);
            MPI_Irecv(&A[end_row + 1][0], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &recv_req[1]);

            MPI_Waitall(2, send_req, status);
            MPI_Waitall(2, recv_req, status);
        }
    }

    int actual_start = (start_row < 2) ? 2 : start_row;
    int actual_end = (end_row > N - 2) ? N - 2 : end_row;

    for (int i = actual_start; i < actual_end; i++) {
        for (int j = 2; j <= N - 3; j++) {
            B[i][j] = (A[i - 2][j] + A[i - 1][j] + A[i + 2][j] + A[i + 1][j] +
                       A[i][j - 2] + A[i][j - 1] + A[i][j + 2] + A[i][j + 1]) / 8.0;
        }
    }
}

double resid(double A[N][N], double B[N][N], int rank, int size, int start_row, int end_row) {
    double local_eps = 0.0;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j <= N - 2; j++) {
            double e = fabs(A[i][j] - B[i][j]);
            A[i][j] = B[i][j];
            local_eps = Max(local_eps, e);
        }
    }

    return local_eps;
}

void verify(double A[N][N], int rank, int size, int start_row, int end_row) {
    double local_s = 0.0;

    for (int i = 0; i < N; i++) {
        if (i >= start_row && i < end_row) {
            for (int j = 0; j < N; j++) {
                local_s += A[i][j] * (i + 1) * (j + 1) / (N * N);
            }
        }
    }

    double s;
    MPI_Reduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("  S = %f\n", s);
    }
}