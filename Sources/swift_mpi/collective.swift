
import libopenmpi
import libopenmpi_sys

public func swift_mpi_send<Element: MPIEquivalent>(_ sendbuf: Array<Element>, _ dst: Int32, _ tag: Int32, _ comm: MPI_Comm) -> Void {
    print("mpi_send")
    let type = Element.mpi_equivalent()
    let count = Int32(sendbuf.count)
    sendbuf.withUnsafeBufferPointer {b in 
        MPI_Send(UnsafeRawPointer(b.baseAddress), count, type, dst, tag, comm)
    }
}

public func swift_mpi_recv<Element: MPIEquivalent>(_ recvbuf: inout Array<Element>, _ src: Int32, _ tag: Int32, _ comm: MPI_Comm) -> Void {
    print("mpi_recv")
    let type = Element.mpi_equivalent()
    let count = Int32(recvbuf.count)
    recvbuf.withUnsafeMutableBufferPointer { b in
        MPI_Recv(UnsafeMutableRawPointer(b.baseAddress), count, type, src, tag, comm, nil)
    }
}

public func swift_mpi_sendrecv<SendElement: MPIEquivalent, RecvElement: MPIEquivalent>(_ sendbuf: Array<SendElement>, _ dst: Int32, _ sendtag: Int32, _ recvbuf: inout Array<RecvElement>, _ src: Int32, _ recvtag: Int32, _ comm: MPI_Comm) -> Void {
    print("mpi_sendrecv")
    let sendtype = SendElement.mpi_equivalent()
    let sendcount = Int32(sendbuf.count)
    let recvtype = RecvElement.mpi_equivalent()
    let recvcount = Int32(recvbuf.count)
    sendbuf.withUnsafeBufferPointer { sb in 
        recvbuf.withUnsafeMutableBufferPointer { rb in
            MPI_Sendrecv(UnsafeRawPointer(sb.baseAddress), sendcount, sendtype, dst, sendtag, UnsafeMutableRawPointer(rb.baseAddress), recvcount, recvtype, src, recvtag, comm, nil)
        }
    }
}

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
