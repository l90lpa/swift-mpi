
import libopenmpi
import libopenmpi_sys
import _Differentiation

public func smpi_allreduce<Element: Datatype, Op: Operation>(sendbuf: Array<Element>, recvbuf_desc: Array<Element>, op: Op, comm: MPI_Comm) -> Array<Element> {
    print("mpi_allreduce")
    let type = Element.mpi_equivalent()
    let mpi_op = Op.mpi_equivalent()
    let count = Int32(sendbuf.count)
    var recvbuf = recvbuf_desc
    sendbuf.withUnsafeBufferPointer { sb in 
        recvbuf.withUnsafeMutableBufferPointer { rb in
            MPI_Allreduce(UnsafeRawPointer(sb.baseAddress), UnsafeMutableRawPointer(rb.baseAddress), count, type, mpi_op, comm)
        }
    }
    return recvbuf
}

public func smpi_scan<Element: Datatype, Op: Operation>(sendbuf: Array<Element>, recvbuf_desc: Array<Element>, op: Op, comm: MPI_Comm) -> Array<Element> {
    print("mpi_scan")
    let type = Element.mpi_equivalent()
    let mpi_op = Op.mpi_equivalent()
    let count = Int32(sendbuf.count)
    var recvbuf = recvbuf_desc
    sendbuf.withUnsafeBufferPointer { sb in 
        recvbuf.withUnsafeMutableBufferPointer { rb in
            MPI_Scan(UnsafeRawPointer(sb.baseAddress), UnsafeMutableRawPointer(rb.baseAddress), count, type, mpi_op, comm)
        }
    }
    return recvbuf
}

@derivative(of: smpi_allreduce, wrt: (sendbuf, recvbuf_desc))
public func smpi_allreduce_value_and_vjp<Element: ValueDatatype & Differentiable, Op: DifferentiableValueOperation>(_ sendbuf: Array<Element>, _ recvbuf_desc: Array<Element>, _ op: Op, _ comm: MPI_Comm) -> (value: Array<Element>, pullback: (Array<Element>.TangentVector) -> (Array<Element>.TangentVector, Array<Element>.TangentVector)) {
    func pullback(_ Dy: Array<Element>.TangentVector) -> (Array<Element>.TangentVector, Array<Element>.TangentVector) {
        let vjp : (Array<Element>, Array<Element>.TangentVector, MPI_Comm) -> (Array<Element>.TangentVector, Array<Element>.TangentVector) = Op.allreduce_vjp
        return vjp(sendbuf, Dy, comm)
    }
    let value = smpi_allreduce(sendbuf: sendbuf, recvbuf_desc: recvbuf_desc, op: op, comm: comm)
    return (value, pullback)
}