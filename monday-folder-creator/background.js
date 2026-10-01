// Clicking the toolbar icon opens the side panel next to the Monday page.
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((err) => console.error(err));
