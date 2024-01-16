import _Differentiation

// See: https://www.tensorflow.org/swift/tutorials/Swift_autodiff_sharp_edges#array_subscript_sets
extension Array where Element: Differentiable {
    @differentiable(where Element: Differentiable)
    mutating func updated(at index: Int, with newValue: Element) {
        self[index] = newValue
    }

    @derivative(of: updated)
    mutating func vjpUpdated(at index: Int, with newValue: Element)
      -> (value: Void, pullback: (inout TangentVector) -> (Element.TangentVector))
    {
        self.updated(at: index, with: newValue)
        return ((), { v in
            let dElement = v[index]
            v.base[index] = .zero
            return dElement
        })
    }
}

func zerosLike<T: Datatype>(_ array: [T]) -> [T] {
    return Array(repeating: T.zero(), count: array.count)
}