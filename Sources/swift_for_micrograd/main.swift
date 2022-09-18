

import Foundation

infix operator **


struct Value: Hashable {
    
    var data:      Float
    var grad:      Float
    var prev:      Set<Value>
    var op:        String
    var _backward: () -> ()
    
    // __repr__
    var description: String {
        return "Value(\(self.data), \(self.grad), \(self.op))"
    }

    // init
    init(_ data: Float, _ children: Set<Value>, _ op: String = "") {
        self.data = data
        self.grad = 0
        self.prev = children
        self.op = op
        self._backward = {}
    }
    
    init(_ data: Float) {
        self.data = data
        self.grad = 0
        self.prev = Set<Value>()
        self.op = ""
        self._backward = {}
    }

    // add
    func add(other: Value) -> Value {
        var cp1 = self
        var cp2 = other
        var result = Value(self.data + other.data, [cp1, cp2], "+")
        result._backward = {
            cp1.grad += result.grad
            cp2.grad += result.grad
        }
        return result
    }

    //overload add
    static func +(lhs: Value, rhs: Value) -> Value {
        return lhs.add(other: rhs)
    }

    // mul
    func mul(other: Value) -> Value {
        var cp1 = self
        var cp2 = other
        var result = Value(self.data * other.data, [cp1, cp2], "*")
        result._backward = {
            cp1.grad += cp2.grad * result.grad
            cp2.grad += cp1.grad * result.grad
        }
        return result
    }
    
    //overload mul
    static func *(lhs: Value, rhs: Value) -> Value {
        return lhs.mul(other: rhs)
    }

    // pow
    func power(other: Value) -> Value {
        var cp1 = self
        let cp2 = other
        var result = Value(pow(self.data, other.data), [self, other], "** + \(other.data)")
        result._backward = {
            cp1.grad += (cp2.data * pow(self.data, -1)) * result.grad
        }
        return result
    }

    //overload pow
    static func **(lhs: Value, rhs: Value) -> Value {
        return lhs.power(other: rhs)
    }

    // relu
    func relu() -> Value {
        var cp1 = self
        var result = Value(max(0, cp1.data), [cp1], "relu")
        result._backward = {
            cp1.grad += (result.data > 0 ? result.data : 0) * result.grad
        }
        return result
    }
    
    // backward
    func backward() {
        var cp1 = self
        var topo = [Value]()
        var visited = Set<Value>()

        var build_topo: (Value) -> () {
            return { v in
                if visited.contains(v) {
                    return
                }
                visited.insert(v)
                for p in v.prev {
                    build_topo(p)
                }
                topo.append(v)
            }
        }
        build_topo(cp1)
        cp1.grad = 1
        for v in topo.reversed() {
            v._backward()
        }
    }

    prefix static func -(value: Value) -> Value {
        var copy = value
        copy.data = copy.data * -1
        return copy
    }
    
    func radd(other: Value) -> Value {
        return self + other
    }
    
    func sub(other: Value) -> Value {
        return self + (-other)
    }
    
    func rsub(other: Value) -> Value {
        return other + (-self)
    }
    
    func rmul(other: inout Value) -> Value {
        return other * self
    }
    
    func truediv(other: Value) -> Value {
        return self * (-other)
    }
    
    func rtruediv(other: Value) -> Value {
        return other * (-self)
    }

    static func ==(lhs: Value, rhs: Value) -> Bool {
        return lhs.data == rhs.data && lhs.grad == rhs.grad && lhs.prev == rhs.prev && lhs.op == rhs.op
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(data)
    }
    
    static func vecmul(lhs: [Value], rhs: [Value]) -> Value {
        var ret: Float = 0.0
        for i in 0..<lhs.count {
            ret += Float(lhs[i].data * rhs[i].data)
        }
        return Value(ret)
    }
}


class Module {
    
    func parameters() -> [Value] {
        return [Value]()
    }
    
    func zero_grad() {
        for var p in self.parameters() {
            p.grad = 0
        }
    }
}


@dynamicCallable
class Neuron: Module  {

    var weights: [Value]
    var bias:    Value
    var nonlin:  Bool
    
    // __repr__
    var description: String {
        if self.nonlin {
            return "'ReLU' Neuron(\(self.weights.count))"
        }
        return "'Linear' Neuron(\(self.weights.count))"
    }
    
    init(nin: Int, _ nonlin: Bool = true) {
        self.weights = [Value]()
        for _ in 0..<nin {
            self.weights.append(Value(Float.random(in: -1.0 ..< 1.0)))
        }
        self.bias = Value(0)
        self.nonlin = nonlin
    }
    
    func dynamicallyCall(withArguments x: [[Value]]) -> Value {
        var ret = Value.vecmul(lhs: self.weights, rhs: x[0])
        ret.data += bias.data
        
        if self.nonlin == false {
            return ret
        }
        return ret.relu()
    }
    
    override func parameters() -> [Value] {
        return self.weights + [self.bias]
    }

}

@dynamicCallable
class Layer: Module {


    var neurons: [Neuron]
    
    
    init(nin: Int, nout: Int) {
        self.neurons = [Neuron]()
        for _ in 0..<nout {
            self.neurons.append(Neuron(nin: nin, true))
        }
    }
    
    func dynamicallyCall(withArguments x: [[Value]]) -> [Value] {
        var ret = [Value]()
        for n in self.neurons {
            ret.append(n(x[0]))
        }
        return ret
    }
    
    override func parameters() -> [Value] {
        var ret = [Value]()
        for n in self.neurons {
            for p in n.parameters() {
                ret.append(p)
            }
        }
        return ret
    }

}

@dynamicCallable
struct MLP {
    
    var sz:     [Int]
    var layers: [Layer]
    
    init(nin: Int, nouts: [Int]) {
        self.sz     = [nin] + nouts
        self.layers = [Layer]()
        for i in 0..<nouts.count {
            self.layers.append(Layer(nin: sz[i], nout: sz[i+1]))
        }
    }
    
    func dynamicallyCall(withArguments x: [[Value]]) -> [Value] {
        var ret = [Value]()
        for layer in self.layers {
            ret.append(contentsOf: layer(x[0]))
        }
        return ret
    }
}