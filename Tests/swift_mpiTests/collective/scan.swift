
import XCTest

@testable import libopenmpi_sys
@testable import swift_mpi

func triangle_number(_ num: Int32) -> Int32 {
    var tri_num: Int32 = 0
    for i in 1...num {
        tri_num += i
    }
    return tri_num
}

final class scan: XCTestCase {

    func test_mpi_scan() throws {

        GlobalTestObservationCenter.shared.registerAllObservers()

        let world =  SWIFT_MPI_COMM_WORLD!
        // let op = BuiltInOperation.sum()
        let op = SMPI_SUM()
        let size = swift_mpi_comm_size(world)
        let rank_plus_one = swift_mpi_comm_rank(world) + 1

        let count = 2
        let sendbuf = [Double](arrayLiteral: 1.0 * Double(rank_plus_one), 
                                                       2.0 * Double(rank_plus_one))
        var recvbuf = [Double](repeating: 0, count: count)

        if size > 1 {
            recvbuf = smpi_scan(sendbuf: sendbuf, recvbuf_desc: recvbuf, op: op, comm: world)
            XCTAssertEqual(sendbuf, 
                           [Double](arrayLiteral: 1.0 * Double(rank_plus_one), 
                                                  2.0 * Double(rank_plus_one)),
                           "sendbuf has not changed.")
            XCTAssertEqual(recvbuf, 
                           [Double](arrayLiteral: 1.0 * Double(triangle_number(rank_plus_one)),
                                                  2.0 * Double(triangle_number(rank_plus_one))),
                           "recvbuf equal to sendbuf on rank 0.")
        }
    }
}
