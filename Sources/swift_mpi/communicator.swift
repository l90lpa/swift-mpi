
import libopenmpi
import libopenmpi_sys


public func swift_mpi_comm_size(_ comm: MPI_Comm) -> Int32 {
    print("mpi_comm_size")
    var size: Int32 = 0
    MPI_Comm_size(comm, &size)
    return size
}

public func swift_mpi_comm_rank(_ comm: MPI_Comm) -> Int32 {
    print("mpi_comm_rank")
    var rank: Int32 = 0
    MPI_Comm_rank(comm, &rank)
    return rank
}