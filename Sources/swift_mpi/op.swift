import libopenmpi
import libopenmpi_sys
import _Differentiation

public protocol Operation : C_MPI_Equivalent {
    typealias C_MPI_Type = MPI_Op
    static func mpi_equivalent() -> MPI_Op
}

public protocol DifferentiableValueOperation : Operation {
    static func allreduce_vjp<Element: ValueDatatype & Differentiable>(_ primal: Array<Element>, _ cotangent: Array<Element>.TangentVector, _ comm: MPI_Comm) -> (Array<Element>.TangentVector, Array<Element>.TangentVector)
}

public struct SMPI_SUM : DifferentiableValueOperation {
    public static func mpi_equivalent() -> MPI_Op {
        return SWIFT_MPI_SUM
    }

    public static func allreduce_vjp<Element: ValueDatatype & Differentiable>(_ primal: Array<Element>, _ cotangent: Array<Element>.TangentVector, _ comm: MPI_Comm) -> (Array<Element>.TangentVector, Array<Element>.TangentVector) {
        let Dy = cotangent.base as! [Element]
        let Dx = smpi_allreduce(sendbuf: Dy, recvbuf_desc: Dy, op: SMPI_SUM(), comm: comm)
        let zero = zerosLike(Dy)
        return (Array<Element>.TangentVector(Dx as! [Element.TangentVector]), Array<Element>.TangentVector(zero as! [Element.TangentVector]))
    }
}

public struct SMPI_MAX : DifferentiableValueOperation {
    public static func mpi_equivalent() -> MPI_Op {
        return SWIFT_MPI_MAX
    }

    public static func allreduce_vjp<Element: ValueDatatype & Differentiable>(_ primal: Array<Element>, _ cotangent: Array<Element>.TangentVector, _ comm: MPI_Comm) -> (Array<Element>.TangentVector, Array<Element>.TangentVector) {
        let x = primal
        let rank = swift_mpi_comm_rank(comm)
        var x_loc = [Element.ValueLocType](repeating: Element.ValueLocType.zero(), count: x.count)
        for i in 0...x_loc.count-1 {
            x_loc[i] = Element.to_value_loc(x[i], rank)
        }
        let y_loc = smpi_allreduce(sendbuf: x_loc, recvbuf_desc: x_loc, op: SMPI_MAXLOC(), comm: comm)
        
        let Dy = cotangent.base as! [Element]
        var Dx = smpi_allreduce(sendbuf: Dy, recvbuf_desc: Dy, op: SMPI_SUM(), comm: comm)
        for i in 0...Dy.count-1 {
            if y_loc[i].loc() != rank {
                Dx[i] = Element.zero()
            }
        }

        let zero = zerosLike(x)
        return (Array<Element>.TangentVector(Dx as! [Element.TangentVector]), Array<Element>.TangentVector(zero as! [Element.TangentVector]))
    }
}

public struct SMPI_MAXLOC : Operation {
    public static func mpi_equivalent() -> MPI_Op {
        return SWIFT_MPI_MAXLOC
    }
}