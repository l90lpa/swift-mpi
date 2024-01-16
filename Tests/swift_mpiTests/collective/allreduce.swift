
import XCTest

@testable import libopenmpi_sys
@testable import swift_mpi
import _Differentiation

final class allreduce: XCTestCase {

    func test_mpi_allreduce_sum() throws {

        GlobalTestObservationCenter.shared.registerAllObservers()

        let world =  SWIFT_MPI_COMM_WORLD!
        // let op = BuiltInOperation.sum()
        let op = SMPI_SUM()
        let size = swift_mpi_comm_size(world)

        let count = 4
        let sendbuf = [Double](arrayLiteral: 1,2,3,4) 
        var recvbuf = [Double](repeating: 0, count: count)

        if size > 1 {
            recvbuf = smpi_allreduce(sendbuf: sendbuf, recvbuf_desc: recvbuf, op: op, comm: world)
            XCTAssertEqual(sendbuf, [Double](arrayLiteral: 1,2,3,4), "sendbuf has not changed.")
            XCTAssertEqual(recvbuf, [Double](arrayLiteral: Double(1*size),
                                                           Double(2*size),
                                                           Double(3*size),
                                                           Double(4*size)), "recvbuf equal to sendbuf on rank 0.")
        }
    }

    func test_mpi_allreduce_maxloc() throws {

        GlobalTestObservationCenter.shared.registerAllObservers()

        let world =  SWIFT_MPI_COMM_WORLD!
        let op = SMPI_MAXLOC()
        let size = swift_mpi_comm_size(world)
        let rank = swift_mpi_comm_rank(world)

        let sendbuf = [DoubleInt(value: Double(rank), loc: rank), DoubleInt(value: Double(-rank), loc: rank)]
        var recvbuf = [DoubleInt(value: 0, loc: rank), DoubleInt(value: 0, loc: rank)]

        if size > 1 {
            recvbuf = smpi_allreduce(sendbuf: sendbuf, recvbuf_desc: recvbuf, op: op, comm: world)
            XCTAssertEqual(recvbuf,
                           [DoubleInt(value: Double(size-1), loc: size-1), DoubleInt(value: 0, loc: 0)],
                           "recvbuf equal to sendbuf on rank 0.")
        }
    }

    func test_mpi_allreduce_max_vjp() throws {

        GlobalTestObservationCenter.shared.registerAllObservers()

        let world =  SWIFT_MPI_COMM_WORLD!
        let size = swift_mpi_comm_size(world)
        let rank = swift_mpi_comm_rank(world)

        @differentiable(reverse)
        func allreduce_max(_ sendbuf: [Double]) -> [Double] {
            return smpi_allreduce(sendbuf: sendbuf, recvbuf_desc: sendbuf, op: SMPI_MAX(), comm: world)
        }

        let x = [Double(-rank), Double(rank)]
        
        let (y, allreduce_max_vjp) = valueWithPullback(at: x, of: allreduce_max)

        let Dy = [Double].TangentVector([1,1])
        let Dx = allreduce_max_vjp(Dy)
        print(Dx)

        if size > 1 {
            if rank == 0 {
                XCTAssertEqual(Dx,
                [Double(size), Double(0)],
                "Only the first element on the first rank contributes to the first element of Dx.")
            }
            else if rank == size-1 {
                XCTAssertEqual(Dx,
                [Double(0), Double(size)],
                "Only the second element on the last rank contributes to the second element of Dx.")
            } 
            else {
                 XCTAssertEqual(Dx,
                [Double(0), Double(0)],
                "no elements contribute to Dx.")
            }
        }
    }
}
