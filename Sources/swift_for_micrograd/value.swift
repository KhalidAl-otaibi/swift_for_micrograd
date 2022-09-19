
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
        return "Value(\(self.data), \(self.grad))"
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
        var result = Value(pow(self.data, other.data), [cp1, cp2], "** + \(other.data)")
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
    
    func rmul(other: Value) -> Value {
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
