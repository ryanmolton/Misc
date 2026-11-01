# CameraPull

A small Mac app that does one thing: plug in your camera (built for a Canon EOS R8, works with
any PTP camera or iPhone), see thumbnails of everything on the card, pick what you want, and
download it to a folder.

It talks to the camera through Apple's **ImageCaptureCore** framework (the same one Image
Capture and Photos use), not Canon's SDK, so EOS Utility isn't involved at all.

## Features

- Detects the camera as soon as it's plugged in, and keeps working through unplug and replug
- Thumbnail grid grouped by day, newest first, with a "Select day" button for each day
- Click to select, Shift-click to select a range, plus **All / New / None** buttons
  (**New** = everything that isn't already in your destination folder)
- Files already downloaded get a green badge and are skipped by default
- Never overwrites a file: Canon file numbers wrap at 9999, so if `IMG_1234.CR3` already exists
  and is a different file, the new one is saved as `IMG_1234-1.CR3`
- Downloaded files keep the date they were shot as their file date
- Downloads one file at a time, which cameras handle more reliably, with a progress bar and Cancel
- If the camera doesn't respond, it retries the connection automatically. **Reconnect** (⌘R)
  does the same thing by hand.

## Build

You need macOS 13 or later and Apple's command-line developer tools. If you don't have them,
this installs them (no full Xcode needed):

```bash
xcode-select --install
```

Then:

```bash
cd CameraPull
./build-app.sh
open build/CameraPull.app
```

Drag `build/CameraPull.app` into `/Applications` to keep it. The first time you open it, macOS
asks for permission to access the camera. Click **Allow**.

## If the camera still isn't detected

These are the usual causes, and they affect every app, not just EOS Utility:

1. **Wi-Fi or Bluetooth is on in the camera.** Canon cameras turn off USB while they're in a
   wireless connection. Turn on Airplane mode on the R8, or turn off Wi-Fi in its wireless settings.
2. **EOS Utility, or its "EOS Utility Launcher" helper, is running.** It can take exclusive control
   of the camera as soon as it's plugged in. Quit it, and remove it from Login Items if it starts
   on its own.
3. **The cable only charges.** Many USB-C cables can't carry data. Use the cable that came with the
   camera, and plug it straight into the Mac rather than through a hub.
4. **Apple Silicon is blocking the accessory.** Check System Settings → Privacy & Security →
   *Allow accessories to connect*.
5. **The camera went to sleep.** Half-press the shutter, then click Reconnect.
