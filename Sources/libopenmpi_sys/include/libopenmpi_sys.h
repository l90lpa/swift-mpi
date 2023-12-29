#pragma once
#include <mpi.h>

extern const MPI_Datatype SWIFT_MPI_FLOAT;
extern const MPI_Datatype SWIFT_MPI_DOUBLE;

extern const MPI_Comm SWIFT_MPI_COMM_WORLD;

extern const MPI_Op SWIFT_MPI_SUM;
extern const MPI_Op SWIFT_MPI_MIN;
extern const MPI_Op SWIFT_MPI_MAX;
extern const MPI_Op SWIFT_MPI_MIN_LOC;
extern const MPI_Op SWIFT_MPI_MAX_LOC;