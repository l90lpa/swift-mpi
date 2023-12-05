
import XCTest

@testable import libopenmpi_sys
@testable import swift_mpi

final class swift_mpiTests: XCTestCase {
    func test_mpi_send_recv() throws {
        // XCTest Documentation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods


        swift_mpi_init(CommandLine.argc, CommandLine.unsafeArgv)

        var comm =  SWIFT_MPI_COMM_WORLD!
        var size: Int32 = 0
        var rank: Int32 = 0
        swift_mpi_comm_size(comm, &size)
        swift_mpi_comm_rank(comm, &rank)

        let count = 4
        var sendbuf = [Double](repeating: Double(rank + 1), count: count)
        var recvbuf = [Double](repeating: Double(rank + 1), count: count)

        if size > 1 {
            if rank == 0 {
                swift_mpi_send( sendbuf, 1, 0, comm)
                swift_mpi_recv(&recvbuf, 1, 0, comm)
                XCTAssertEqual(sendbuf, [Double](repeating: 1, count: count), "sendbuf has not changed.")
                XCTAssertEqual(recvbuf, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 1.")
            }
            if rank == 1 {
                swift_mpi_recv(&recvbuf, 0, 0, comm)
                swift_mpi_send( sendbuf, 0, 0, comm)
                XCTAssertEqual(sendbuf, [Double](repeating: 2, count: count), "sendbuf has not changed.")
                XCTAssertEqual(recvbuf, [Double](repeating: 1, count: count), "recvbuf equal to sendbuf on rank 0.")
            }
        }

        swift_mpi_finalize()
    }
}
