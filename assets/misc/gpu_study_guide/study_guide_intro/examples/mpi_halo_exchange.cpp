#include <mpi.h>
#include <cstdio>

// Minimal 1-D halo exchange: every rank sends its right-edge value rightward
// and receives the left neighbor's edge. Requires an MPI implementation.
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int sendValue = 100 + rank, leftHalo = -1;
    int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(&sendValue, 1, MPI_INT, right, 0,
                 &leftHalo, 1, MPI_INT, left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::printf("rank=%d received-left-halo=%d\n", rank, leftHalo);
    MPI_Finalize();
}

