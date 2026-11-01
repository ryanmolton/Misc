import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct CameraPullApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var manager = CameraManager()

    var body: some Scene {
        WindowGroup("CameraPull") {
            ContentView()
                .environmentObject(manager)
                .frame(minWidth: 860, minHeight: 580)
        }
        .commands {
            CommandMenu("Camera") {
                Button("Select All") { manager.selectAll() }
                    .keyboardShortcut("a", modifiers: [.command, .shift])
                Button("Select Not Yet Downloaded") { manager.selectNew() }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
                Button("Deselect All") { manager.selectNone() }
                    .keyboardShortcut("d", modifiers: [.command, .shift])
                Divider()
                Button("Download Selected") { manager.downloadSelected() }
                    .keyboardShortcut(.return, modifiers: [.command])
                Divider()
                Button("Reconnect Camera") { manager.reconnect(manual: true) }
                    .keyboardShortcut("r", modifiers: [.command])
            }
        }
    }
}
