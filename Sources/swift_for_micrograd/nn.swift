
import Foundation


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
final class Neuron: Module  {

    var weights: [Value]
    var bias:    Value
    var nonlin:  Bool
    
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
final class Layer: Module {

    var neurons: [Neuron]

    var description: String {
        var ret = "Layer of [" 
        for n in self.neurons {
            ret += n.description + ", "
        }
        ret += "]"
        return ret 
    }
    
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
final class MLP: Module {
    
    var sz:     [Int]
    var layers: [Layer]

    var description: String {
        var ret = "MLP of ["
        for layer in self.layers {
            ret += layer.description + ", "
        }
        ret += "]"
        return ret
    }
    
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

    override func parameters() -> [Value] {
        var ret = [Value]()
        for layer in self.layers {
            for p in layer.parameters() {
                ret.append(p)
            }
        }
        return ret
    }
}