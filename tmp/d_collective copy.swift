
import libopenmpi
import libopenmpi_sys
import _Differentiation

public func smpi_send<Element: MPIEquivalent>(sendbuf: Array<Element>, dst: Int32, tag: Int32, comm: MPI_Comm) -> Array<Element> {
    print("mpi_send")
    let type = Element.mpi_equivalent()
    let count = Int32(sendbuf.count)
    sendbuf.withUnsafeBufferPointer {b in 
        MPI_Send(UnsafeRawPointer(b.baseAddress), count, type, dst, tag, comm)
    }
    return zerosLike(sendbuf)
}

public func smpi_recv<Element: MPIEquivalent>(recvbuf_desc: Array<Element>, src: Int32, tag: Int32, comm: MPI_Comm) -> Array<Element> {
    print("mpi_recv")
    var recvbuf = zerosLike(recvbuf_desc)
    let type = Element.mpi_equivalent()
    let count = Int32(recvbuf.count)
    recvbuf.withUnsafeMutableBufferPointer { b in
        MPI_Recv(UnsafeMutableRawPointer(b.baseAddress), count, type, src, tag, comm, nil)
    }
    return recvbuf
}

// public func smpi_sendrecv<SendElement: MPIEquivalent, RecvElement: MPIEquivalent>(_ sendbuf: Array<SendElement>, _ dst: Int32, _ sendtag: Int32, _ recvbuf_description: Array<RecvElement>, _ src: Int32, _ recvtag: Int32, _ comm: MPI_Comm) -> Array<RecvElement> {
//     print("mpi_sendrecv")
//     let sendtype = SendElement.mpi_equivalent()
//     let sendcount = Int32(sendbuf.count)
//     var recvbuf = recvbuf_description
//     let recvtype = RecvElement.mpi_equivalent()
//     let recvcount = Int32(recvbuf.count)
//     sendbuf.withUnsafeBufferPointer { sb in 
//         recvbuf.withUnsafeMutableBufferPointer { rb in
//             MPI_Sendrecv(UnsafeRawPointer(sb.baseAddress), sendcount, sendtype, dst, sendtag, UnsafeMutableRawPointer(rb.baseAddress), recvcount, recvtype, src, recvtag, comm, nil)
//         }
//     }
//     return recvbuf
// }

// public func smpi_allreduce<Element: MPIEquivalent>(_ sendbuf: Array<Element>, _ op: MPI_Op, _ comm: MPI_Comm) -> Array<Element> {
//     print("mpi_allreduce")
//     let type = Element.mpi_equivalent()
//     let count = Int32(sendbuf.count)
//     var recvbuf = sendbuf
//     sendbuf.withUnsafeBufferPointer { sb in 
//         recvbuf.withUnsafeMutableBufferPointer { rb in
//             MPI_Allreduce(UnsafeRawPointer(sb.baseAddress), UnsafeMutableRawPointer(rb.baseAddress), count, type, op, comm)
//         }
//     }
//     return recvbuf
// }

// ========= Derivatives =========

@derivative(of: smpi_send, wrt: sendbuf)
public func smpi_send_value_and_jvp<Element: MPIEquivalent & Differentiable>(_ sendbuf: Array<Element>, _ dst: Int32, _ tag: Int32, _ comm: MPI_Comm) -> (value: Array<Element>, differential: (Array<Element>.TangentVector) -> Array<Element>.TangentVector) {
    func differential(_ dx: Array<Element>.TangentVector) -> Array<Element>.TangentVector {
        let buf = dx.base as! [Element]
        let dy = smpi_send(sendbuf: buf, dst: dst, tag: tag, comm: comm)
        return Array<Element>.TangentVector(dy as! [Element.TangentVector])
    }
    let value = smpi_send(sendbuf: sendbuf, dst: dst, tag: tag, comm: comm)
    return (value, differential)
}

@derivative(of: smpi_send, wrt: sendbuf)
public func smpi_send_value_and_vjp<Element: MPIEquivalent & Differentiable>(_ sendbuf: Array<Element>, _ dst: Int32, _ tag: Int32, _ comm: MPI_Comm) -> (value: Array<Element>, pullback: (Array<Element>.TangentVector) -> Array<Element>.TangentVector) {
    func pullback(_ Dy: Array<Element>.TangentVector) -> Array<Element>.TangentVector {
        let Dx = smpi_recv(recvbuf_desc: sendbuf, src: dst, tag: tag, comm: comm)
        return Array<Element>.TangentVector(Dx as! [Element.TangentVector])
    }
    let value = smpi_send(sendbuf: sendbuf, dst: dst, tag: tag, comm: comm)
    return (value, pullback)
}

@derivative(of: smpi_recv, wrt: recvbuf)
public func smpi_recv_value_and_jvp<Element: MPIEquivalent & Differentiable>(_ recvbuf: Array<Element>, _ src: Int32, _ tag: Int32, _ comm: MPI_Comm) -> (value: Array<Element>, differential: (Array<Element>.TangentVector) -> Array<Element>.TangentVector) {
    func differential(_ Dy: Array<Element>.TangentVector) -> Array<Element>.TangentVector {
        let dy = smpi_recv(recvbuf_desc: recvbuf, src: src, tag: tag, comm: comm)
        return Array<Element>.TangentVector(dy as! [Element.TangentVector])
    }
    let value = smpi_recv(recvbuf_desc: recvbuf, src: src, tag: tag, comm: comm)
    return (value, differential)
}

@derivative(of: smpi_recv, wrt: recvbuf)
public func smpi_recv_value_and_vjp<Element: MPIEquivalent & Differentiable>(_ recvbuf: Array<Element>, _ src: Int32, _ tag: Int32, _ comm: MPI_Comm) -> (value: Array<Element>, pullback: (Array<Element>.TangentVector) -> Array<Element>.TangentVector) {
    func pullback(_ Dy: Array<Element>.TangentVector) -> Array<Element>.TangentVector {
        let buf = Dy.base as! [Element]
        let Dx = smpi_send(sendbuf: buf, dst: src, tag: tag, comm: comm)
        let zero = zerosLike(buf) 
        return Array<Element>.TangentVector(zero as! [Element.TangentVector])
    }
    let value = smpi_recv(recvbuf_desc: recvbuf, src: src, tag: tag, comm: comm)
    return (value, pullback)
}