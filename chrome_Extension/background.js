// Chrome拡張機能のバックグラウンドスクリプト
chrome.tabs.onActivated.addListener(async (activeInfo) => {
    try {
        const tab = await chrome.tabs.get(activeInfo.tabId);
        console.log("Current Tab Info:", tab); // タブ情報をログに出力
        console.log("Sending URL:", tab.url);

        await fetch("http://127.0.0.1:5000/browser-data", { // URLを確認
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: tab.url, title: tab.title })
        });
    } catch (error) {
        console.error("Error fetching tab or sending data:", error);
    }
});
