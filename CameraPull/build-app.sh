#!/bin/bash
# Builds CameraPull.app into ./build. Requires Xcode or the Xcode Command Line Tools
# (install with: xcode-select --install).
set -euo pipefail
cd "$(dirname "$0")"

swift build -c release
BIN="$(swift build -c release --show-bin-path)/CameraPull"

APP="build/CameraPull.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/CameraPull"
cp Info.plist "$APP/Contents/Info.plist"

# Ad-hoc sign so macOS will run it and remember permissions.
codesign --force --deep --sign - "$APP"

echo
echo "Built $APP"
echo "Drag it to /Applications, or run: open \"$APP\""
