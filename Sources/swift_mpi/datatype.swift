
import libopenmpi
import libopenmpi_sys

public protocol MPIEquivalent {
    static func mpi_equivalent() -> MPI_Datatype
}

extension Float : MPIEquivalent {
    public static func mpi_equivalent() -> MPI_Datatype {
        return SWIFT_MPI_FLOAT
    }
}

extension Double : MPIEquivalent {
    public static func mpi_equivalent() -> MPI_Datatype {
        return SWIFT_MPI_DOUBLE
    }
}
