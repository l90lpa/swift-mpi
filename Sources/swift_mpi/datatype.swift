
import libopenmpi
import libopenmpi_sys
import _Differentiation


public protocol Datatype : C_MPI_Equivalent {
    typealias C_MPI_Type = MPI_Datatype
    static func mpi_equivalent() -> MPI_Datatype
    
    static func zero() -> Self
}

public protocol ValueDatatype : Datatype {
    associatedtype ValueLocType : ValueLocDatatype
    static func to_value_loc(_ value: Self, _ loc: Int32) -> ValueLocType
}

public protocol ValueLocDatatype : Datatype {
    associatedtype ValueType
    func value() -> ValueType 
    func loc() -> Int32 
}

extension Float : ValueDatatype {
    public static func mpi_equivalent() -> MPI_Datatype {
        return SWIFT_MPI_FLOAT
    }

    public static func zero() -> Float { return Float(0) }
    public typealias ValueLocType = FloatInt
    public static func to_value_loc(_ value: Self, _ loc: Int32) -> FloatInt {
        return FloatInt(value: value, loc: loc)
    }
}

extension Double : ValueDatatype {
    public static func mpi_equivalent() -> MPI_Datatype {
        return SWIFT_MPI_DOUBLE
    }

    public static func zero() -> Double { return Double(0) }
    public typealias ValueLocType = DoubleInt
    public static func to_value_loc(_ value: Self, _ loc: Int32) -> DoubleInt {
        return DoubleInt(value: value, loc: loc)
    }
}


extension FloatInt : ValueLocDatatype {
    public static func mpi_equivalent() -> MPI_Datatype {
        return SWIFT_MPI_FLOAT_INT
    }

    public static func zero() -> FloatInt { return FloatInt(value: 0, loc: 0) }
    public typealias ValueType = Float
    public func value() -> ValueType {
        return self.value
    }
    public func loc() -> Int32 {
        return self.loc
    }
}

extension FloatInt : Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.value == rhs.value && lhs.loc == rhs.loc
    }
}

extension DoubleInt : ValueLocDatatype {
    public static func mpi_equivalent() -> MPI_Datatype {
        return SWIFT_MPI_DOUBLE_INT
    }

    public static func zero() -> DoubleInt { return DoubleInt(value: 0, loc: 0) }
    public typealias ValueType = Double
    public func value() -> ValueType {
        return self.value
    }
    public func loc() -> Int32 {
        return self.loc
    }
}

extension DoubleInt : Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.value == rhs.value && lhs.loc == rhs.loc
    }
}