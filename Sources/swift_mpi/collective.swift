
import libopenmpi
import libopenmpi_sys

public func swift_mpi_allreduce<Element: MPIEquivalent>(_ sendbuf: Array<Element>, _ recvbuf: inout Array<Element>, _ op: MPI_Op, _ comm: MPI_Comm) -> Void {
    print("mpi_allreduce")
    let type = Element.mpi_equivalent()
    let count = Int32(sendbuf.count)
    sendbuf.withUnsafeBufferPointer { sb in 
        recvbuf.withUnsafeMutableBufferPointer { rb in
            MPI_Allreduce(UnsafeRawPointer(sb.baseAddress), UnsafeMutableRawPointer(rb.baseAddress), count, type, op, comm)
        }
    }
}
