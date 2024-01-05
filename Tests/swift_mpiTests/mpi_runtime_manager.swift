
import swift_mpi
import Foundation
import XCTest

class MPIRuntimeManager: NSObject, XCTestObservation {
    override init() {
        super.init()
        print("Starting MPI runtime ...")
        swift_mpi_init(CommandLine.argc, CommandLine.unsafeArgv)
    }

    func testBundleDidFinish(_ testBundle: Bundle) {
        print("Stopping MPI runtime ...")
        swift_mpi_finalize()
    }
}

class GlobalTestObservationCenter {
    static let shared = GlobalTestObservationCenter()

    private init(){}
    private var registered = false
    private let lock = NSLock()

    func registerAllObservers() {
        lock.lock()
        if !registered {
            XCTestObservationCenter.shared.addTestObserver(MPIRuntimeManager())
            registered = true
        }
        lock.unlock()
    }
}