// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CameraPull",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "CameraPull",
            path: "Sources/CameraPull",
            linkerSettings: [.linkedFramework("ImageCaptureCore")]
        )
    ]
)
