
import libopenmpi
import libopenmpi_sys
import _Differentiation

public protocol MPIEquivalent: Numeric {
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
