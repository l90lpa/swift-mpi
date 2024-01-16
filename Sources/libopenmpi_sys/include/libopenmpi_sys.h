#pragma once
#include <mpi.h>

extern const MPI_Datatype SWIFT_MPI_FLOAT;
extern const MPI_Datatype SWIFT_MPI_DOUBLE;
extern const MPI_Datatype SWIFT_MPI_FLOAT_INT;
extern const MPI_Datatype SWIFT_MPI_DOUBLE_INT;

extern const MPI_Comm SWIFT_MPI_COMM_WORLD;

extern const MPI_Op SWIFT_MPI_SUM;
extern const MPI_Op SWIFT_MPI_MIN;
extern const MPI_Op SWIFT_MPI_MAX;
extern const MPI_Op SWIFT_MPI_MINLOC;
extern const MPI_Op SWIFT_MPI_MAXLOC;

struct FloatInt {
    float value;
    int   loc;
};

struct DoubleInt {
    double value;
    int    loc;
};