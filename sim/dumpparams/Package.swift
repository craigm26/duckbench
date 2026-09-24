// swift-tools-version:5.9
import PackageDescription
let package = Package(
    name: "dumpparams",
    platforms: [.macOS(.v13)],
    dependencies: [.package(url: "https://github.com/craigm26/duckkit.git", from: "1.36.0")],
    targets: [.executableTarget(name: "dumpparams",
                                dependencies: [.product(name: "DuckKit", package: "duckkit")])]
)
