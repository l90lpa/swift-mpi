
public protocol C_MPI_Equivalent {
    associatedtype C_MPI_Type
    static func mpi_equivalent() -> C_MPI_Type
}