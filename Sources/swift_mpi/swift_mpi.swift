
import libopenmpi
import libopenmpi_sys


public func swift_mpi_finalize() -> Void {
    print("mpi_finalize")
    MPI_Finalize();
}

public func swift_mpi_init(_ argc: Int32, _ argv: UnsafeMutablePointer<UnsafeMutablePointer<Int8>?>) -> Void {
    print("mpi_init")
    var argc_copy = argc
    var argv_copy = argv
    swift_mpi_init_helper(&argc_copy, &argv_copy)
}

internal func swift_mpi_init_helper(_ argc: UnsafeMutablePointer<Int32>, _ argv: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<Int8>?>>) -> Void {
    argv.withMemoryRebound(
            to: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?.self, 
            capacity: Int(argc.pointee)) { ptr in
        MPI_Init(argc, ptr)
    }
}

public func swift_mpi_comm_size(_ comm: MPI_Comm, _ size: UnsafeMutablePointer<Int32>) -> Void {
    print("mpi_comm_size")
    MPI_Comm_size(comm, size)
}

public func swift_mpi_comm_rank(_ comm: MPI_Comm, _ rank: UnsafeMutablePointer<Int32>) -> Void {
    print("mpi_comm_rank")
    MPI_Comm_rank(comm, rank)
}

public func swift_mpi_send(_ sendbuf: Array<Double>, _ dst: Int32, _ tag: Int32, _ comm: MPI_Comm) -> Void {
    print("mpi_send")
    let count = Int32(sendbuf.count)
    sendbuf.withUnsafeBufferPointer {buffer in 
        MPI_Send(UnsafeRawPointer(buffer.baseAddress), count, SWIFT_MPI_DOUBLE, dst, tag, comm)
    }
}

public func swift_mpi_recv(_ recvbuf: inout Array<Double>, _ src: Int32, _ tag: Int32, _ comm: MPI_Comm) -> Void {
    print("mpi_recv")
    let count = Int32(recvbuf.count)
    recvbuf.withUnsafeMutableBufferPointer { buffer in
        MPI_Recv(UnsafeMutableRawPointer(buffer.baseAddress), Int32(count), SWIFT_MPI_DOUBLE, src, tag, comm, nil)
    }
}