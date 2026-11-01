import AppKit
import ImageCaptureCore
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var m: CameraManager
    @AppStorage("thumbSize") private var thumbSize: Double = 150

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Group {
                if m.activeCamera == nil {
                    NoCameraView()
                } else if m.items.isEmpty {
                    ConnectingView()
                } else {
                    grid
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            footer
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "camera.fill").font(.title2).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                if m.cameras.count > 1 {
                    Picker("", selection: Binding(
                        get: { m.activeCamera.map(ObjectIdentifier.init) },
                        set: { id in
                            if let cam = m.cameras.first(where: { ObjectIdentifier($0) == id }) { m.activate(cam) }
                        })
                    ) {
                        ForEach(m.cameras, id: \.self) { cam in
                            Text(cam.name ?? "Camera").tag(Optional(ObjectIdentifier(cam)))
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 260)
                } else {
                    Text(m.activeCamera?.name ?? "No camera").font(.headline)
                }
                Text(m.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer()
            if !m.items.isEmpty {
                Image(systemName: "photo").foregroundStyle(.secondary)
                Slider(value: $thumbSize, in: 90...300).frame(width: 110)
            }
            Button {
                m.reconnect(manual: true)
            } label: {
                Label("Reconnect", systemImage: "arrow.clockwise")
            }
            .help("Close and reopen the connection to the camera (⌘R)")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: Grid

    private var grid: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 14, pinnedViews: [.sectionHeaders]) {
                ForEach(m.sections) { section in
                    Section {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: thumbSize, maximum: thumbSize + 40), spacing: 12)],
                                  alignment: .leading, spacing: 14) {
                            ForEach(section.items) { item in
                                ThumbCell(item: item, selected: m.selection.contains(item.id), size: thumbSize)
                                    .onAppear { m.requestThumbnail(item) }
                                    .onTapGesture {
                                        m.click(item, extend: NSEvent.modifierFlags.contains(.shift))
                                    }
                            }
                        }
                        .padding(.horizontal, 16)
                    } header: {
                        sectionHeader(section)
                    }
                }
            }
            .padding(.bottom, 16)
        }
    }

    private func sectionHeader(_ section: DaySection) -> some View {
        let ids = Set(section.items.map(\.id))
        let all = ids.isSubset(of: m.selection)
        return HStack {
            Text(section.id == Calendar.current.startOfDay(for: .distantPast)
                 ? "Unknown date"
                 : section.id.formatted(date: .complete, time: .omitted))
                .font(.headline)
            Text("\(section.items.count)").foregroundStyle(.secondary)
            Spacer()
            Button(all ? "Deselect day" : "Select day") { m.toggleSection(section) }
                .buttonStyle(.link)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(.bar)
    }

    // MARK: Footer

    private var footer: some View {
        VStack(spacing: 8) {
            if m.isDownloading {
                HStack {
                    ProgressView(value: Double(m.downloadDone), total: Double(max(m.downloadTotal, 1)))
                    Text("\(m.downloadDone) / \(m.downloadTotal)").monospacedDigit().foregroundStyle(.secondary)
                    Button("Cancel") { m.cancelDownload() }
                }
            }
            if !m.failures.isEmpty && !m.isDownloading {
                DisclosureGroup("\(m.failures.count) file(s) failed") {
                    ScrollView {
                        Text(m.failures.joined(separator: "\n"))
                            .font(.caption.monospaced())
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 80)
                }
                .foregroundStyle(.red)
            }
            HStack(spacing: 10) {
                Button("All") { m.selectAll() }.disabled(m.items.isEmpty)
                Button("New") { m.selectNew() }.disabled(m.items.isEmpty)
                    .help("Select everything not already in the destination folder")
                Button("None") { m.selectNone() }.disabled(m.selection.isEmpty)
                Text(selectionSummary).foregroundStyle(.secondary).monospacedDigit()

                Spacer()

                Image(systemName: "folder").foregroundStyle(.secondary)
                Button(m.destination.path.replacingOccurrences(of: NSHomeDirectory(), with: "~")) {
                    m.revealDestination()
                }
                .buttonStyle(.link)
                .lineLimit(1)
                .truncationMode(.head)
                .help("Show in Finder")
                Button("Change…") { m.chooseDestination() }.disabled(m.isDownloading)
                Toggle("Skip already downloaded", isOn: $m.skipExisting)
                    .toggleStyle(.checkbox)
                Button {
                    m.downloadSelected()
                } label: {
                    Label("Download", systemImage: "arrow.down.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(m.selection.isEmpty || m.isDownloading)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var selectionSummary: String {
        guard !m.items.isEmpty else { return "" }
        let size = ByteCountFormatter.string(fromByteCount: m.selectedBytes, countStyle: .file)
        return "\(m.selection.count) of \(m.items.count) selected · \(size)"
    }
}

// MARK: - Cells

struct ThumbCell: View {
    @ObservedObject var item: CameraFileItem
    let selected: Bool
    let size: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ZStack {
                RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12))
                if let thumb = item.thumbnail {
                    Image(nsImage: thumb).resizable().aspectRatio(contentMode: .fit)
                } else {
                    Image(systemName: item.isVideo ? "film" : "photo")
                        .font(.title).foregroundStyle(.tertiary)
                }
            }
            .frame(width: size, height: size * 0.72)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(alignment: .topTrailing) {
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .font(.title2)
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(Color.white, selected ? Color.accentColor : Color.black.opacity(0.35))
                    .shadow(radius: 1.5)
                    .padding(6)
            }
            .overlay(alignment: .bottomLeading) {
                Text(item.ext)
                    .font(.system(size: 10, weight: .bold))
                    .padding(.horizontal, 5).padding(.vertical, 2)
                    .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 3))
                    .foregroundStyle(.white)
                    .padding(6)
            }
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(selected ? Color.accentColor : .clear, lineWidth: 3)
            )

            HStack(spacing: 4) {
                Text(item.name).lineLimit(1).truncationMode(.middle)
                if item.onDisk {
                    Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                        .help("Already in the destination folder")
                }
                Spacer(minLength: 0)
            }
            .font(.caption)
            if let date = item.date {
                Text(date.formatted(date: .omitted, time: .shortened))
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .frame(width: size)
        .contentShape(Rectangle())
    }
}

// MARK: - Empty states

struct ConnectingView: View {
    @EnvironmentObject var m: CameraManager
    var body: some View {
        VStack(spacing: 14) {
            ProgressView()
            Text(m.status).foregroundStyle(.secondary).multilineTextAlignment(.center)
            if m.catalogComplete {
                Text("No photos or videos on the card.").font(.headline)
            }
        }
        .padding(40)
    }
}

struct NoCameraView: View {
    @EnvironmentObject var m: CameraManager
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                ProgressView().controlSize(.small)
                Text("Waiting for a camera…").font(.title2.weight(.semibold))
            }
            if m.accessDenied {
                Text("CameraPull isn't allowed to access cameras. Turn it on in System Settings → Privacy & Security → Camera, then relaunch.")
                    .foregroundStyle(.red)
            }
            Text("If your camera is plugged in and nothing shows up, check these:")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 10) {
                tip("power", "The camera is on and awake. Half-press the shutter if it went to sleep.")
                tip("wifi.slash", "Wi-Fi and Bluetooth are off on the camera. While Canon cameras are in a wireless connection, USB doesn't work. On the R8, turn on Airplane mode or turn off Wi-Fi in the Wireless settings menu.")
                tip("xmark.app", "EOS Utility is quit, along with its \"EOS Utility Launcher\" helper (check the menu bar and Activity Monitor). It can take exclusive control of the camera the moment you plug it in.")
                tip("cable.connector", "You're using a data USB-C cable, not a charge-only one, and it's plugged straight into the Mac, not through a hub.")
                tip("lock.shield", "On Apple Silicon Macs: if macOS asked whether to \"Allow accessory to connect,\" click Allow. You can change this in System Settings → Privacy & Security → Allow accessories to connect.")
            }
            HStack {
                Button("Scan Again") { m.rescan() }
                Button("Open Image Capture") {
                    NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Applications/Image Capture.app"))
                }
                .help("Apple's built-in app. If it can't see the camera either, the problem is the cable or a camera setting.")
            }
            .padding(.top, 4)
        }
        .frame(maxWidth: 620)
        .padding(40)
    }

    private func tip(_ icon: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon).frame(width: 20).foregroundStyle(Color.accentColor)
            Text(text).fixedSize(horizontal: false, vertical: true)
        }
    }
}
