# Monday Folder Creator (Chrome extension)

A side panel that lists the projects visible on a Monday.com board. For each
one it shows whether a project folder already exists, and lets you:

- **Create** a folder by copying your template folder and naming it from the
  row, e.g. `{Item ID}-{Project}` → `12111781128-Enterprise-IsatPhone 3-Video-Asset-2026`
- **Copy path** of an existing folder, to paste into File Explorer or Finder

It reads only what's on screen, so it needs no Monday API access.

## Install (each user)

1. Copy this `monday-folder-creator` folder somewhere permanent on the computer.
2. In Chrome go to `chrome://extensions`, turn on **Developer mode** (top right),
   click **Load unpacked** and choose the folder.
3. Pin the extension, then click its icon. The side panel opens.
4. Click **Open settings** and:
   - choose the **Template folder**
   - choose the **Project folders location**
   - optionally type that location's full path (for "Copy path")
   - check the naming pattern, then click **Save**

## Daily use

Open the board (e.g. *FY27 Creative › Main table*) and click the extension
icon. Every row on screen is listed. **Folder exists** means a folder was found
whose name starts with that row's Item ID, so it's still found if someone
renamed the end of it. Otherwise click **Create**.

The list follows the page as you scroll, filter or switch groups. Monday only
draws the rows on screen, so scroll down to see more. The Item ID and project
name come from each row's own page data, so the Item ID column doesn't need
to be visible. Other columns used in the folder name need to be on screen
(not scrolled off to the side).

## Notes

- **Folder permission:** Chrome sometimes asks again for permission to use the
  folders, e.g. after a restart. Click **Allow folder access** in the panel.
  Choosing "Allow on every visit" stops the prompt.
- **Template placeholders:** file or folder names inside the template can use
  the same `{Column}` placeholders, e.g. `{Item ID} Brief.docx`.
- **Name clean-up:** characters Windows/macOS don't allow (`/ \ : * ? " < > |`)
  are replaced, and names are cut at 150 characters.
- **Opening the folder:** Chrome won't let an extension open Finder or
  File Explorer. Instead, "Copy path" puts the path on the clipboard
  (Finder: ⇧⌘G, Explorer: the address bar). An "Open folder" button would
  need a small helper program installed on each computer.
- **If projects stop being detected:** the extension finds columns by their
  header text. If a column is renamed, update the names in Settings.
