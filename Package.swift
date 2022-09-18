// swift-tools-version:5.5
import PackageDescription

let package = Package(
	name: "swift_for_micrograd",
	products: [
		.executable(name: "swift_for_micrograd", targets: ["swift_for_micrograd"]),
	],
	dependencies: [],
	targets: [
		.executableTarget(name: "swift_for_micrograd", dependencies: [])
	]
)
