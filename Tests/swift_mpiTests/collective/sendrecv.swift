
import XCTest

@testable import libopenmpi_sys
@testable import swift_mpi

final class sendrecv: XCTestCase {

    func test_mpi_sendrecv() throws {

        swift_mpi_init(CommandLine.argc, CommandLine.unsafeArgv)

        let comm =  SWIFT_MPI_COMM_WORLD!
        let size = swift_mpi_comm_size(comm)
        let rank = swift_mpi_comm_rank(comm)

        let count = 4
        let sendbuf = [Double](repeating: Double(rank + 1), count: count)
        var recvbuf = [Double](repeating: Double(rank + 1), count: count)

        if size > 2 {
            if rank == 0 {
                swift_mpi_send( sendbuf, 1, 0, comm)
                swift_mpi_recv(&recvbuf, 1, 0, comm)
                XCTAssertEqual(sendbuf, [Double](repeating: 1, count: count), "sendbuf has not changed.")
                XCTAssertEqual(recvbuf, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 1.")
            }
            if rank == 1 {
                swift_mpi_sendrecv(sendbuf, 2, 0, &recvbuf, 0, 0, comm)
                swift_mpi_sendrecv(sendbuf, 0, 0, &recvbuf, 2, 0, comm)
                XCTAssertEqual(sendbuf, [Double](repeating: 2, count: count), "sendbuf has not changed.")
                XCTAssertEqual(recvbuf, [Double](repeating: 3, count: count), "recvbuf equal to sendbuf on rank 0.")
            }
            if rank == 2 {
                swift_mpi_recv(&recvbuf, 1, 0, comm)
                swift_mpi_send( sendbuf, 1, 0, comm)
                XCTAssertEqual(sendbuf, [Double](repeating: 3, count: count), "sendbuf has not changed.")
                XCTAssertEqual(recvbuf, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 0.")
            }
        }

        swift_mpi_finalize()
    }
}
