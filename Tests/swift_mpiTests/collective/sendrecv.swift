
import XCTest

@testable import libopenmpi_sys
@testable import swift_mpi
import _Differentiation

final class sendrecv: XCTestCase {

    func test_mpi_sendrecv() throws {

        GlobalTestObservationCenter.shared.registerAllObservers()

        let world =  SWIFT_MPI_COMM_WORLD!
        let size = swift_mpi_comm_size(world)
        let rank = swift_mpi_comm_rank(world)

        let count = 4
        let sendbuf = [Double](repeating: Double(rank + 1), count: count)
        var recvbuf = [Double]()

        if size > 2 {
            if rank == 0 {
                smpi_send(sendbuf: sendbuf, dst: 1, tag: 0, comm: world)
                recvbuf = smpi_recv(recvbuf_desc: sendbuf, src: 1, tag: 0, comm: world)
                XCTAssertEqual(recvbuf, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 1.")
            }
            if rank == 1 {
                recvbuf = smpi_sendrecv(sendbuf: sendbuf, dst: 2, sendtag: 0, recvbuf_desc: sendbuf, src: 0, recvtag: 0, comm: world)
                XCTAssertEqual(recvbuf, [Double](repeating: 1, count: count), "recvbuf equal to sendbuf on rank 0.")
                recvbuf = smpi_sendrecv(sendbuf: sendbuf, dst: 0, sendtag: 0, recvbuf_desc: recvbuf, src: 2, recvtag: 0, comm: world)
                XCTAssertEqual(recvbuf, [Double](repeating: 3, count: count), "recvbuf equal to sendbuf on rank 2.")
            }
            if rank == 2 {
                recvbuf = smpi_recv(recvbuf_desc: sendbuf, src: 1, tag: 0, comm: world)
                smpi_send(sendbuf: sendbuf, dst: 1, tag: 0, comm: world)
                XCTAssertEqual(recvbuf, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 0.")
            }
        }
    }

    func test_mpi_sendrecv_vjp() throws {

        GlobalTestObservationCenter.shared.registerAllObservers()

        let world =  SWIFT_MPI_COMM_WORLD!
        let size = swift_mpi_comm_size(world)
        let rank = swift_mpi_comm_rank(world)

        @differentiable(reverse)
        func exchange(_ sendbuf: [Double]) -> [Double] {
            var recvbuf = [Double]()
            if size > 2 {
                if rank == 0 {
                    var sendbuf = smpi_send(sendbuf: sendbuf, dst: 1, tag: 0, comm: world)
                    recvbuf = smpi_recv(recvbuf_desc: sendbuf, src: 1, tag: 0, comm: world)
                }
                if rank == 1 {
                    recvbuf = smpi_sendrecv(sendbuf: sendbuf, dst: 2, sendtag: 0, recvbuf_desc: sendbuf, src: 0, recvtag: 0, comm: world)
                    recvbuf = smpi_sendrecv(sendbuf: sendbuf, dst: 0, sendtag: 0, recvbuf_desc: recvbuf, src: 2, recvtag: 0, comm: world)
                }
                if rank == 2 {
                    recvbuf = smpi_recv(recvbuf_desc: sendbuf, src: 1, tag: 0, comm: world)
                    var sendbuf = extract_left(sendbuf, recvbuf)
                    sendbuf = smpi_send(sendbuf: sendbuf, dst: 1, tag: 0, comm: world)
                    recvbuf = extract_left(recvbuf, sendbuf)
                }
            }
            return recvbuf
        }

        let count = 4
        let x = [Double](repeating: Double(rank + 1), count: count)

        let (y, exchange_vjp) = valueWithPullback(at: x, of: exchange)

        let Dy = rank == 1 ? [Double].TangentVector([1,0,0,0]) : [Double].TangentVector([0,0,0,0])
        let Dx = exchange_vjp(Dy)


        if size > 2 {
            if rank == 0 {
                XCTAssertEqual(Dx, [Double].TangentVector([0,0,0,0]))
            }
            if rank == 1 {
                XCTAssertEqual(Dx, [Double].TangentVector([0,0,0,0]))
            }
            if rank == 2 {
                XCTAssertEqual(Dx, [Double].TangentVector([1,0,0,0]))
            }
        }
    }
}
