import AppKit
import Foundation
import ImageCaptureCore

/// One photo/video on the camera's card.
final class CameraFileItem: ObservableObject, Identifiable {
    let file: ICCameraFile
    let id: ObjectIdentifier
    let name: String
    let size: Int64
    let date: Date?
    let ext: String

    @Published var thumbnail: NSImage?
    /// True when a file with the same name and size already exists in the destination folder.
    @Published var onDisk = false
    var thumbnailRequested = false

    init(file: ICCameraFile) {
        self.file = file
        id = ObjectIdentifier(file)
        name = file.name ?? "Untitled"
        size = Int64(file.fileSize)
        date = file.creationDate ?? file.modificationDate
        ext = (name as NSString).pathExtension.uppercased()
    }

    var isVideo: Bool { ["MP4", "MOV", "CRM", "AVI", "M4V"].contains(ext) }
}

struct DaySection: Identifiable {
    let id: Date
    var items: [CameraFileItem]
}

/// Finds cameras over USB using Apple's ImageCaptureCore (the same framework Image Capture
/// and Photos use) instead of Canon's SDK, and downloads the files you pick.
final class CameraManager: NSObject, ObservableObject {
    @Published private(set) var cameras: [ICCameraDevice] = []
    @Published private(set) var activeCamera: ICCameraDevice?
    @Published private(set) var status = "Looking for cameras…"
    @Published private(set) var items: [CameraFileItem] = []
    @Published private(set) var catalogComplete = false
    @Published var selection: Set<ObjectIdentifier> = []

    @Published private(set) var isDownloading = false
    @Published private(set) var downloadDone = 0
    @Published private(set) var downloadTotal = 0
    @Published private(set) var currentFileName: String?
    @Published private(set) var failures: [String] = []
    @Published private(set) var accessDenied = false

    @Published var skipExisting: Bool {
        didSet { UserDefaults.standard.set(skipExisting, forKey: "skipExisting") }
    }
    @Published private(set) var destination: URL

    private let browser = ICDeviceBrowser()
    private var itemsByID: [ObjectIdentifier: CameraFileItem] = [:]
    private var pending: [CameraFileItem] = []
    private var currentProgress: Progress?
    private var lastClicked: ObjectIdentifier?
    private var watchdog: DispatchWorkItem?
    private var openAttempts = 0

    override init() {
        let defaults = UserDefaults.standard
        destination = defaults.url(forKey: "destination")
            ?? FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Camera Imports", isDirectory: true)
        skipExisting = defaults.object(forKey: "skipExisting") as? Bool ?? true
        super.init()

        browser.delegate = self
        browser.browsedDeviceTypeMask = ICDeviceTypeMask(
            rawValue: ICDeviceTypeMask.camera.rawValue | ICDeviceLocationTypeMask.local.rawValue
        )
        browser.requestContentsAuthorization { [weak self] authStatus in
            DispatchQueue.main.async {
                self?.accessDenied = (authStatus == .denied || authStatus == .restricted)
            }
        }
        browser.start()
    }

    // MARK: - Derived

    var sections: [DaySection] {
        let cal = Calendar.current
        var out: [DaySection] = []
        for item in items {
            let day = cal.startOfDay(for: item.date ?? .distantPast)
            if let last = out.last, last.id == day {
                out[out.count - 1].items.append(item)
            } else {
                out.append(DaySection(id: day, items: [item]))
            }
        }
        return out
    }

    var selectedBytes: Int64 {
        items.reduce(0) { selection.contains($1.id) ? $0 + $1.size : $0 }
    }

    // MARK: - Camera control

    func activate(_ camera: ICCameraDevice) {
        guard activeCamera !== camera else { return }
        cancelDownload()
        if let old = activeCamera, old.hasOpenSession { old.requestCloseSession() }
        resetItems()
        activeCamera = camera
        openAttempts = 0
        open(camera)
    }

    private func open(_ camera: ICCameraDevice) {
        camera.delegate = self
        status = "Connecting to \(camera.name ?? "camera")…"
        camera.requestOpenSession()
        startWatchdog(for: camera)
    }

    /// Closes and reopens the session with the current camera. If there isn't one, rescans.
    func reconnect(manual: Bool) {
        guard let cam = activeCamera else { rescan(); return }
        if manual { openAttempts = 0 }
        openAttempts += 1
        cancelDownload()
        resetItems()
        status = "Reconnecting…"
        if cam.hasOpenSession { cam.requestCloseSession() }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self, self.activeCamera === cam else { return }
            self.open(cam)
        }
    }

    /// Restarts USB device discovery from scratch.
    func rescan() {
        status = "Looking for cameras…"
        browser.stop()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.browser.start()
        }
    }

    /// If the session hasn't opened after 15s, retry automatically a couple of times.
    /// (A sleeping or half-awake camera is the usual cause.)
    private func startWatchdog(for camera: ICCameraDevice) {
        watchdog?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.activeCamera === camera,
                  !camera.hasOpenSession, self.items.isEmpty else { return }
            if self.openAttempts < 3 {
                self.status = "Camera isn't responding. Retrying…"
                self.reconnect(manual: false)
            } else {
                self.status = "The camera is connected but not responding. Half-press the shutter to wake it, then click Reconnect."
            }
        }
        watchdog = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 15, execute: work)
    }

    private func resetItems() {
        items = []
        itemsByID = [:]
        selection = []
        lastClicked = nil
        catalogComplete = false
    }

    private func add(_ cameraItems: [ICCameraItem]) {
        var added: [CameraFileItem] = []
        for case let file as ICCameraFile in cameraItems where itemsByID[ObjectIdentifier(file)] == nil {
            let item = CameraFileItem(file: file)
            itemsByID[item.id] = item
            refreshOnDisk(item)
            added.append(item)
        }
        guard !added.isEmpty else { return }
        var all = items + added
        all.sort { ($0.date ?? .distantPast) > ($1.date ?? .distantPast) }
        items = all
    }

    private func remove(_ cameraItems: [ICCameraItem]) {
        let ids = Set(cameraItems.map { ObjectIdentifier($0) })
        guard !ids.isDisjoint(with: itemsByID.keys) else { return }
        items.removeAll { ids.contains($0.id) }
        ids.forEach { itemsByID[$0] = nil }
        selection.subtract(ids)
    }

    func requestThumbnail(_ item: CameraFileItem) {
        guard !item.thumbnailRequested else { return }
        item.thumbnailRequested = true
        item.file.requestThumbnail()
    }

    // MARK: - Selection

    func click(_ item: CameraFileItem, extend: Bool) {
        if extend, let anchor = lastClicked,
           let a = items.firstIndex(where: { $0.id == anchor }),
           let b = items.firstIndex(where: { $0.id == item.id }) {
            for i in min(a, b)...max(a, b) { selection.insert(items[i].id) }
        } else if selection.contains(item.id) {
            selection.remove(item.id)
        } else {
            selection.insert(item.id)
        }
        lastClicked = item.id
    }

    func selectAll() { selection = Set(items.map(\.id)) }
    func selectNone() { selection = [] }
    func selectNew() { selection = Set(items.filter { !$0.onDisk }.map(\.id)) }

    func toggleSection(_ section: DaySection) {
        let ids = Set(section.items.map(\.id))
        if ids.isSubset(of: selection) { selection.subtract(ids) } else { selection.formUnion(ids) }
    }

    // MARK: - Destination

    func chooseDestination() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.directoryURL = destination
        guard panel.runModal() == .OK, let url = panel.url else { return }
        destination = url
        UserDefaults.standard.set(url, forKey: "destination")
        items.forEach(refreshOnDisk)
    }

    func revealDestination() {
        try? FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        NSWorkspace.shared.open(destination)
    }

    private func refreshOnDisk(_ item: CameraFileItem) {
        let url = destination.appendingPathComponent(item.name)
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attrs?[.size] as? NSNumber)?.int64Value
        item.onDisk = (size == item.size)
    }

    /// Canon file numbers wrap at 9999, so a different file can have the same name as one
    /// you already downloaded. Never overwrite it: save as IMG_1234-1.CR3 instead.
    private func targetName(for item: CameraFileItem) -> String {
        let fm = FileManager.default
        if !fm.fileExists(atPath: destination.appendingPathComponent(item.name).path) { return item.name }
        let base = (item.name as NSString).deletingPathExtension
        let ext = (item.name as NSString).pathExtension
        var n = 1
        while true {
            let candidate = ext.isEmpty ? "\(base)-\(n)" : "\(base)-\(n).\(ext)"
            if !fm.fileExists(atPath: destination.appendingPathComponent(candidate).path) { return candidate }
            n += 1
        }
    }

    // MARK: - Downloading

    func downloadSelected() {
        guard !isDownloading else { return }
        let queue = items.filter { selection.contains($0.id) }
        guard !queue.isEmpty else { return }
        do {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        } catch {
            status = "Can't create the destination folder: \(error.localizedDescription)"
            return
        }
        pending = queue
        downloadTotal = queue.count
        downloadDone = 0
        failures = []
        isDownloading = true
        downloadNext()
    }

    func cancelDownload() {
        guard isDownloading else { return }
        pending = []
        currentProgress?.cancel()
        currentProgress = nil
        isDownloading = false
        currentFileName = nil
        status = "Download cancelled. \(downloadDone) of \(downloadTotal) done."
    }

    /// Files are downloaded one at a time; cameras are far more reliable that way than with
    /// many parallel requests.
    private func downloadNext() {
        guard isDownloading else { return }

        while let next = pending.first, skipExisting, next.onDisk {
            pending.removeFirst()
            downloadDone += 1
            selection.remove(next.id)
        }
        guard let item = pending.first else { finishDownloads(); return }
        pending.removeFirst()

        let target = targetName(for: item)
        var options: [ICDownloadOption: Any] = [.downloadsDirectoryURL: destination]
        if target != item.name { options[.saveAsFilename] = target }

        currentFileName = item.name
        status = "Downloading \(item.name)…"
        let dest = destination
        currentProgress = item.file.requestDownload(options: options) { [weak self] filename, error in
            DispatchQueue.main.async {
                guard let self else { return }
                if let error {
                    self.failures.append("\(item.name): \(error.localizedDescription)")
                } else {
                    let url = dest.appendingPathComponent(filename ?? target)
                    if let date = item.date {
                        try? FileManager.default.setAttributes(
                            [.creationDate: date, .modificationDate: date], ofItemAtPath: url.path)
                    }
                    if target == item.name { item.onDisk = true }
                    self.selection.remove(item.id)
                }
                self.downloadDone += 1
                self.downloadNext()
            }
        }
    }

    private func finishDownloads() {
        isDownloading = false
        currentFileName = nil
        currentProgress = nil
        items.forEach(refreshOnDisk)
        let ok = downloadTotal - failures.count
        status = failures.isEmpty
            ? "Done: \(ok) file\(ok == 1 ? "" : "s") in \(destination.lastPathComponent)."
            : "Finished with \(failures.count) error\(failures.count == 1 ? "" : "s"). \(ok) file\(ok == 1 ? "" : "s") downloaded."
    }
}

// MARK: - ICDeviceBrowserDelegate

extension CameraManager: ICDeviceBrowserDelegate {
    func deviceBrowser(_ browser: ICDeviceBrowser, didAdd device: ICDevice, moreComing: Bool) {
        guard let camera = device as? ICCameraDevice else { return }
        DispatchQueue.main.async {
            if !self.cameras.contains(where: { $0 === camera }) { self.cameras.append(camera) }
            if self.activeCamera == nil { self.activate(camera) }
        }
    }

    func deviceBrowser(_ browser: ICDeviceBrowser, didRemove device: ICDevice, moreGoing: Bool) {
        DispatchQueue.main.async { self.handleRemoval(of: device) }
    }

    func deviceBrowserDidEnumerateLocalDevices(_ browser: ICDeviceBrowser) {
        DispatchQueue.main.async {
            if self.cameras.isEmpty { self.status = "No camera found. Plug it in and turn it on." }
        }
    }

    private func handleRemoval(of device: ICDevice) {
        cameras.removeAll { $0 === device }
        guard activeCamera === device else { return }
        watchdog?.cancel()
        cancelDownload()
        resetItems()
        activeCamera = nil
        status = "Camera disconnected."
        if let next = cameras.first { activate(next) }
    }
}

// MARK: - ICCameraDeviceDelegate

extension CameraManager: ICCameraDeviceDelegate {
    func didRemove(_ device: ICDevice) {
        DispatchQueue.main.async { self.handleRemoval(of: device) }
    }

    func device(_ device: ICDevice, didOpenSessionWithError error: Error?) {
        DispatchQueue.main.async {
            guard self.activeCamera === device else { return }
            if let error {
                self.status = "Couldn't open the camera: \(error.localizedDescription)"
            } else {
                self.status = "Reading memory card…"
            }
        }
    }

    func device(_ device: ICDevice, didCloseSessionWithError error: Error?) {}

    func device(_ device: ICDevice, didEncounterError error: Error?) {
        guard let error else { return }
        DispatchQueue.main.async {
            if self.activeCamera === device { self.status = "Camera error: \(error.localizedDescription)" }
        }
    }

    func deviceDidBecomeReady(withCompleteContentCatalog device: ICCameraDevice) {
        DispatchQueue.main.async {
            guard self.activeCamera === device else { return }
            self.watchdog?.cancel()
            self.add(device.mediaFiles ?? [])
            self.catalogComplete = true
            let n = self.items.count
            self.status = "\(n) file\(n == 1 ? "" : "s") on \(device.name ?? "camera")."
        }
    }

    func cameraDevice(_ camera: ICCameraDevice, didAdd items: [ICCameraItem]) {
        DispatchQueue.main.async {
            guard self.activeCamera === camera else { return }
            self.add(items)
        }
    }

    func cameraDevice(_ camera: ICCameraDevice, didRemove items: [ICCameraItem]) {
        DispatchQueue.main.async { self.remove(items) }
    }

    func cameraDevice(_ camera: ICCameraDevice, didReceiveThumbnail thumbnail: CGImage?,
                      for item: ICCameraItem, error: Error?) {
        guard let thumbnail else { return }
        let image = NSImage(cgImage: thumbnail, size: .zero)
        DispatchQueue.main.async {
            self.itemsByID[ObjectIdentifier(item)]?.thumbnail = image
        }
    }

    func cameraDevice(_ camera: ICCameraDevice, didReceiveMetadata metadata: [AnyHashable: Any]?,
                      for item: ICCameraItem, error: Error?) {}
    func cameraDevice(_ camera: ICCameraDevice, didRenameItems items: [ICCameraItem]) {}
    func cameraDeviceDidChangeCapability(_ camera: ICCameraDevice) {}
    func cameraDevice(_ camera: ICCameraDevice, didReceivePTPEvent eventData: Data) {}
    func cameraDeviceDidRemoveAccessRestriction(_ device: ICDevice) {}
    func cameraDeviceDidEnableAccessRestriction(_ device: ICDevice) {}
}
