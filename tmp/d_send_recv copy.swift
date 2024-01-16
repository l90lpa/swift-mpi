
import XCTest

@testable import libopenmpi_sys
@testable import swift_mpi
import _Differentiation

final class send_recv: XCTestCase {
    // func test_mpi_send_recv() throws {
    //     // XCTest Documentation
    //     // https://developer.apple.com/documentation/xctest

    //     // Defining Test Cases and Test Methods
    //     // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods


    //     swift_mpi_init(CommandLine.argc, CommandLine.unsafeArgv)

    //     let comm =  SWIFT_MPI_COMM_WORLD!
    //     let size = swift_mpi_comm_size(comm)
    //     let rank = swift_mpi_comm_rank(comm)

    //     let count = 4
    //     let sendbuf = [Double](repeating: Double(rank + 1), count: count)
    //     var recvbuf = [Double](repeating: Double(rank + 1), count: count)

    //     if size > 1 {
    //         if rank == 0 {
    //             smpi_send(sendbuf, 1, 0, comm)
    //             recvbuf = smpi_recv(recvbuf, 1, 0, comm)
    //             XCTAssertEqual(sendbuf, [Double](repeating: 1, count: count), "sendbuf has not changed.")
    //             XCTAssertEqual(recvbuf, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 1.")
    //         }
    //         if rank == 1 {
    //             recvbuf = smpi_recv(recvbuf, 0, 0, comm)
    //             smpi_send(sendbuf, 0, 0, comm)
    //             XCTAssertEqual(sendbuf, [Double](repeating: 2, count: count), "sendbuf has not changed.")
    //             XCTAssertEqual(recvbuf, [Double](repeating: 1, count: count), "recvbuf equal to sendbuf on rank 0.")
    //         }
    //     }

    //     swift_mpi_finalize()
    // }    

    func test_jvp_mpi_send_recv() throws {

        swift_mpi_init(CommandLine.argc, CommandLine.unsafeArgv)

        let world =  SWIFT_MPI_COMM_WORLD!
        let size = swift_mpi_comm_size(world)
        let rank = swift_mpi_comm_rank(world)

        let count = 4
        let x = [Double](repeating: Double(rank + 1), count: count)

        @differentiable(reverse)
        func exchange(_ sendbuf: [Double]) -> [Double] {
            var recvbuf = [Double]()
            if size > 1 {
                if rank == 0 {
                    let zeroArray = smpi_send(sendbuf: sendbuf, dst: 1, tag: 0, comm: world)
                    recvbuf = smpi_recv(recvbuf_desc: zeroArray, src: 1, tag: 0, comm: world)
                }
                if rank == 1 {
                    recvbuf = smpi_recv(recvbuf_desc: sendbuf, src: 0, tag: 0, comm: world)
                    // Insert extract_left to ensure that send and recv cannot be reordered
                    var sendbuf = extract_left(sendbuf, recvbuf) 
                    let zeroArray = smpi_send(sendbuf: sendbuf, dst: 0, tag: 0, comm: world)
                    // Insert extract_left to ensure that send cannot be elided under autodiff
                    recvbuf = extract_left(recvbuf, zeroArray)
                }
            }
            return recvbuf
        }

        let (y, exchange_jvp) = valueWithDifferential(at: x, of: exchange)

        let dx = rank == 0 ? [Double].TangentVector([1,0,0,0]) : [Double].TangentVector([0,0,0,0])
        let dy = exchange_jvp(dx)
        print(dy)
        
        if size > 1 {
            if rank == 0 {
                XCTAssertEqual(x, [Double](repeating: 1, count: count), "sendbuf has not changed.")
                XCTAssertEqual(y, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 1.")
                XCTAssertEqual(dy, [Double].TangentVector([0,0,0,0]), "recvbuf equal to sendbuf on rank 1.")
            }
            if rank == 1 {
                XCTAssertEqual(x, [Double](repeating: 2, count: count), "sendbuf has not changed.")
                XCTAssertEqual(y, [Double](repeating: 1, count: count), "recvbuf equal to sendbuf on rank 0.")
                XCTAssertEqual(dy, [Double].TangentVector([1,0,0,0]), "recvbuf equal to sendbuf on rank 1.")
            }
        }

        // swift_mpi_finalize()
    }


    // func test_vjp_mpi_send_recv() throws {

    //     swift_mpi_init(CommandLine.argc, CommandLine.unsafeArgv)

    //     let world =  SWIFT_MPI_COMM_WORLD!
    //     let size = swift_mpi_comm_size(world)
    //     let rank = swift_mpi_comm_rank(world)

    //     let count = 4
    //     let x = [Double](repeating: Double(rank + 1), count: count)

    //     @differentiable(reverse)
    //     func exchange(_ sendbuf: [Double]) -> [Double] {
    //         var recvbuf = [Double]()
    //         if size > 1 {
    //             if rank == 0 {
    //                 let zeroArray = smpi_send(sendbuf: sendbuf, dst: 1, tag: 0, comm: world)
    //                 recvbuf = smpi_recv(recvbuf_desc: zeroArray, src: 1, tag: 0, comm: world)
    //             }
    //             if rank == 1 {
    //                 recvbuf = smpi_recv(recvbuf_desc: sendbuf, src: 0, tag: 0, comm: world)
    //                 // Insert extract_left to ensure that send and recv cannot be reordered
    //                 var sendbuf = extract_left(sendbuf, recvbuf) 
    //                 let zeroArray = smpi_send(sendbuf: sendbuf, dst: 0, tag: 0, comm: world)
    //                 // Insert extract_left to ensure that send cannot be elided under autodiff
    //                 recvbuf = extract_left(recvbuf, zeroArray)
    //             }
    //         }
    //         return recvbuf
    //     }

    //     let (y, exchange_vjp) = valueWithPullback(at: x, of: exchange)

    //     let Dy = rank == 0 ? [Double].TangentVector([1,0,0,0]) : [Double].TangentVector([0,0,0,0])
    //     let Dx = exchange_vjp(Dy)
    //     print(Dx)
        
    //     if size > 1 {
    //         if rank == 0 {
    //             XCTAssertEqual(x, [Double](repeating: 1, count: count), "sendbuf has not changed.")
    //             XCTAssertEqual(y, [Double](repeating: 2, count: count), "recvbuf equal to sendbuf on rank 1.")
    //             XCTAssertEqual(Dx, [Double].TangentVector([0,0,0,0]), "recvbuf equal to sendbuf on rank 1.")
    //         }
    //         if rank == 1 {
    //             XCTAssertEqual(x, [Double](repeating: 2, count: count), "sendbuf has not changed.")
    //             XCTAssertEqual(y, [Double](repeating: 1, count: count), "recvbuf equal to sendbuf on rank 0.")
    //             XCTAssertEqual(Dx, [Double].TangentVector([1,0,0,0]), "recvbuf equal to sendbuf on rank 1.")
    //         }
    //     }

    //     swift_mpi_finalize()
    // }

}
