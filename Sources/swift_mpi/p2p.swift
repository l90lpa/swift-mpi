
import libopenmpi
import libopenmpi_sys
import _Differentiation

public func smpi_send<Element: Datatype>(sendbuf: Array<Element>, dst: Int32, tag: Int32, comm: MPI_Comm) -> Array<Element> {
    print("mpi_send")
    let type = Element.mpi_equivalent()
    let count = Int32(sendbuf.count)
    sendbuf.withUnsafeBufferPointer {b in 
        MPI_Send(UnsafeRawPointer(b.baseAddress), count, type, dst, tag, comm)
    }
    return zerosLike(sendbuf)
}

public func smpi_recv<Element: Datatype>(recvbuf_desc: Array<Element>, src: Int32, tag: Int32, comm: MPI_Comm) -> Array<Element> {
    print("mpi_recv")
    var recvbuf = zerosLike(recvbuf_desc)
    let type = Element.mpi_equivalent()
    let count = Int32(recvbuf.count)
    recvbuf.withUnsafeMutableBufferPointer { b in
        MPI_Recv(UnsafeMutableRawPointer(b.baseAddress), count, type, src, tag, comm, nil)
    }
    return recvbuf
}

public func smpi_sendrecv<SendElement: Datatype, RecvElement: Datatype>(sendbuf: Array<SendElement>, dst: Int32, sendtag: Int32, recvbuf_desc: Array<RecvElement>, src: Int32, recvtag: Int32, comm: MPI_Comm) -> Array<RecvElement> {
    print("mpi_sendrecv")
    let sendtype = SendElement.mpi_equivalent()
    let sendcount = Int32(sendbuf.count)
    var recvbuf = recvbuf_desc
    let recvtype = RecvElement.mpi_equivalent()
    let recvcount = Int32(recvbuf.count)
    sendbuf.withUnsafeBufferPointer { sb in 
        recvbuf.withUnsafeMutableBufferPointer { rb in
            MPI_Sendrecv(UnsafeRawPointer(sb.baseAddress), sendcount, sendtype, dst, sendtag, UnsafeMutableRawPointer(rb.baseAddress), recvcount, recvtype, src, recvtag, comm, nil)
        }
    }
    return recvbuf
}

// ========= Derivatives =========

@derivative(of: smpi_send, wrt: 0)
public func smpi_send_value_and_vjp<Element: Datatype & Differentiable>(_ sendbuf: Array<Element>, _ dst: Int32, _ tag: Int32, _ comm: MPI_Comm) -> (value: Array<Element>, pullback: (Array<Element>.TangentVector) -> Array<Element>.TangentVector) {
    func pullback(_ Dy: Array<Element>.TangentVector) -> Array<Element>.TangentVector {
        let Dx = smpi_recv(recvbuf_desc: sendbuf, src: dst, tag: tag, comm: comm)
        return Array<Element>.TangentVector(Dx as! [Element.TangentVector])
    }
    let value = smpi_send(sendbuf: sendbuf, dst: dst, tag: tag, comm: comm)
    return (value, pullback)
}

@derivative(of: smpi_recv, wrt: 0)
public func smpi_recv_value_and_vjp<Element: Datatype & Differentiable>(_ recvbuf: Array<Element>, _ src: Int32, _ tag: Int32, _ comm: MPI_Comm) -> (value: Array<Element>, pullback: (Array<Element>.TangentVector) -> Array<Element>.TangentVector) {
    func pullback(_ Dy: Array<Element>.TangentVector) -> Array<Element>.TangentVector {
        let buf = Dy.base as! [Element]
        let Dx = smpi_send(sendbuf: buf, dst: src, tag: tag, comm: comm)
        let zero = zerosLike(buf) 
        return Array<Element>.TangentVector(zero as! [Element.TangentVector])
    }
    let value = smpi_recv(recvbuf_desc: recvbuf, src: src, tag: tag, comm: comm)
    return (value, pullback)
}

@derivative(of: smpi_sendrecv, wrt: (0, 3))
public func smpi_sendrecv_value_and_vjp<SendElement: Datatype & Differentiable, RecvElement: Datatype & Differentiable>(_ sendbuf: Array<SendElement>, _ dst: Int32, _ sendtag: Int32, _ recvbuf_desc: Array<RecvElement>, _ src: Int32, _ recvtag: Int32, _ comm: MPI_Comm) -> (value: Array<RecvElement>, pullback: (Array<RecvElement>.TangentVector) -> (Array<SendElement>.TangentVector, Array<RecvElement>.TangentVector)) {
    func pullback(_ Dy: Array<RecvElement>.TangentVector) -> (Array<SendElement>.TangentVector, Array<RecvElement>.TangentVector) {
        let Dy_ = Dy.base as! [RecvElement]
        let Dx = smpi_sendrecv(sendbuf: Dy_, dst: src, sendtag: 0, recvbuf_desc: sendbuf, src: dst, recvtag: 0, comm: comm)
        let zero = zerosLike(Dy_)
        return (Array<SendElement>.TangentVector(Dx as! [SendElement.TangentVector]), Array<RecvElement>.TangentVector(zero as! [RecvElement.TangentVector]))
    }
    let value = smpi_sendrecv(sendbuf: sendbuf, dst: dst, sendtag: sendtag, recvbuf_desc: recvbuf_desc, src: src, recvtag: recvtag, comm: comm)
    return (value, pullback)
}