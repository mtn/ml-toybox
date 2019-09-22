import Metal
import MetalKit

@available(OSX 10.11, *)
func run() {
    // Earlier version of iOS
    let mtlDevice = MTLCreateSystemDefaultDevice()

    guard let device = mtlDevice else {
        print("Failed to init metal")
        return
    }

    print("device is", device)
}

if #available(OSX 10.11, *) {
    run()
}
