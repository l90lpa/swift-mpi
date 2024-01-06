
import libopenmpi
import libopenmpi_sys

public func smpi_allreduce<Element: MPIEquivalent>(sendbuf: Array<Element>, recvbuf_desc: Array<Element>, op: MPI_Op, comm: MPI_Comm) -> Array<Element> {
    print("mpi_allreduce")
    let type = Element.mpi_equivalent()
    let count = Int32(sendbuf.count)
    var recvbuf = recvbuf_desc
    sendbuf.withUnsafeBufferPointer { sb in 
        recvbuf.withUnsafeMutableBufferPointer { rb in
            MPI_Allreduce(UnsafeRawPointer(sb.baseAddress), UnsafeMutableRawPointer(rb.baseAddress), count, type, op, comm)
        }
    }
    return recvbuf
}
