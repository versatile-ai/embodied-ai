import Foundation
import AVFoundation
import AppKit

let args = CommandLine.arguments
let folder = URL(fileURLWithPath: args[1])
let output = URL(fileURLWithPath: args[2])
let fps = Int32(args[3])!
let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil).filter { $0.pathExtension == "png" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
guard let first = files.first, let ns = NSImage(contentsOf: first), let cg = ns.cgImage(forProposedRect: nil, context: nil, hints: nil) else { fatalError("No PNG frames") }
let width=cg.width, height=cg.height
if FileManager.default.fileExists(atPath: output.path) { try FileManager.default.removeItem(at: output) }
let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height])
input.expectsMediaDataInRealTime=false
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB, kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height, kCVPixelBufferCGImageCompatibilityKey as String: true, kCVPixelBufferCGBitmapContextCompatibilityKey as String: true])
writer.add(input)
guard writer.startWriting() else { fatalError("startWriting failed: \(String(describing: writer.error))") }
writer.startSession(atSourceTime: .zero)
let done=DispatchSemaphore(value:0)
var index=0
input.requestMediaDataWhenReady(on: DispatchQueue(label:"encode")) {
    while input.isReadyForMoreMediaData && index < files.count {
        autoreleasepool {
            let image=NSImage(contentsOf: files[index])!.cgImage(forProposedRect:nil,context:nil,hints:nil)!
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil,adaptor.pixelBufferPool!,&buffer)
            let pixel=buffer!
            CVPixelBufferLockBaseAddress(pixel,[])
            let context=CGContext(data:CVPixelBufferGetBaseAddress(pixel),width:width,height:height,bitsPerComponent:8,bytesPerRow:CVPixelBufferGetBytesPerRow(pixel),space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.noneSkipFirst.rawValue)!
            context.draw(image,in:CGRect(x:0,y:0,width:width,height:height))
            CVPixelBufferUnlockBaseAddress(pixel,[])
            guard adaptor.append(pixel,withPresentationTime:CMTime(value:Int64(index),timescale:fps)) else { fatalError("append failed") }
            index += 1
        }
    }
    if index == files.count {
        input.markAsFinished()
        writer.finishWriting { done.signal() }
    }
}
guard done.wait(timeout:.now()+50) == .success, writer.status == .completed else { fatalError("video export failed: \(String(describing:writer.error))") }
print("Encoded \(files.count) frames: \(output.path)")
