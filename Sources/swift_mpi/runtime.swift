
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