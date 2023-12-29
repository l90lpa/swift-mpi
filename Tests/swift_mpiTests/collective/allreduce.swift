
import XCTest

@testable import libopenmpi_sys
@testable import swift_mpi

final class allreduce: XCTestCase {

    func test_mpi_allreduce() throws {

        swift_mpi_init(CommandLine.argc, CommandLine.unsafeArgv)

        let comm =  SWIFT_MPI_COMM_WORLD!
        let op = SWIFT_MPI_SUM!
        let size = swift_mpi_comm_size(comm)

        let count = 4
        let sendbuf = [Double](arrayLiteral: 1,2,3,4) 
        var recvbuf = [Double](repeating: 0, count: count)

        if size > 1 {
            swift_mpi_allreduce(sendbuf, &recvbuf, op, comm)
            XCTAssertEqual(sendbuf, [Double](arrayLiteral: 1,2,3,4), "sendbuf has not changed.")
            XCTAssertEqual(recvbuf, [Double](arrayLiteral: Double(1*size),
                                                           Double(2*size),
                                                           Double(3*size),
                                                           Double(4*size)), "recvbuf equal to sendbuf on rank 0.")
        }

        swift_mpi_finalize()
    }
}
