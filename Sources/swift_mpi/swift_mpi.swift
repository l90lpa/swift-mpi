

public func swift_mpi_finalize() -> Void {
    print("mpi_finalize")
}

public func swift_mpi_init(_ argc: Int32, _ argv: UnsafeMutablePointer<UnsafeMutablePointer<Int8>?>) -> Void {
    print("mpi_init")
}

public func swift_mpi_comm_size(_ comm: MPI_Comm, _ size: UnsafeMutablePointer<Int32>) -> Void {
    print("mpi_comm_size")
}

public func swift_mpi_comm_rank(_ comm: MPI_Comm, _ rank: UnsafeMutablePointer<Int32>) -> Void {
    print("mpi_comm_rank")
}

public func swift_mpi_send(_ sendbuf: Array<Double>, _ dst: Int32, _ tag: Int32, _ comm: MPI_Comm) -> Void {
    print("mpi_send")
}

public func swift_mpi_recv(_ recvbuf: inout Array<Double>, _ src: Int32, _ tag: Int32, _ comm: MPI_Comm) -> Void {
    print("mpi_recv")
}