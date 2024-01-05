import _Differentiation

func extract_left<T>(_ left: T, _ right: T) -> T {
    return left
}

@derivative(of: extract_left)
func extract_left_vjp<T: Differentiable>(_ left: T, _ right: T) -> 
    (value: T, 
     pullback: (T.TangentVector) -> (T.TangentVector, T.TangentVector)) {
    return (value: left, pullback: {D in (D, .zero)})
}

@derivative(of: extract_left)
func extract_left_jvp<T: Differentiable>(_ left: T, _ right: T) -> 
    (value: T, 
     differential: (T.TangentVector, T.TangentVector) -> T.TangentVector) {
    return (value: left, differential: {(dleft: T.TangentVector, dright: T.TangentVector) in dleft})
}