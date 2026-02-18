let songList = document.querySelector("#songs #list");
let statsContent = document.querySelector("#stats-content");
let searchInput = document.querySelector("#search");
let sortSelect = document.querySelector("#sort");
let headerTitle = document.querySelector("#header-title");
let downloadResultsButton = document.querySelector("#download-results");

let deletedSongsStorageKey = "deletedSongs";
let lastSelectedSongStorageKey = "lastSelectedSong";
let deletedSongNames = new Set(JSON.parse(localStorage.getItem(deletedSongsStorageKey) || "[]"));

let songConclusions = {};
let songConclusionColors = {};

function calculateConclusion(songName) {
    let totals = {};
    let models = results[songName];

    Object.values(models).forEach((genreList) =>
        genreList.forEach((g) => (totals[g.genre] = (totals[g.genre] || 0) + g.prob)),
    );

    let modelCount = Object.keys(models).length;

    return Object.entries(totals)
        .map(([genre, prob]) => ({ genre, prob: prob / modelCount }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 3);
}

function getConclusionColor(conclusion) {
    let topProb = conclusion[0]?.prob || 0;
    let secondProb = conclusion[1]?.prob || 0;
    let probDiff = topProb - secondProb;

    let isGreen = (topProb >= 35 && probDiff >= 10) || probDiff >= 20;
    let isRed = topProb < 20 || probDiff < 5;

    if (isGreen) return "green";
    if (isRed) return "red";
    return "neutral";
}

Object.keys(results).forEach((songName) => {
    let conclusion = calculateConclusion(songName);
    songConclusions[songName] = conclusion;
    songConclusionColors[songName] = getConclusionColor(conclusion);
});

function sortSongNames(songNames) {
    let sortValue = sortSelect.value;

    if (sortValue === "name") {
        return [...songNames].sort((a, b) => a.localeCompare(b));
    }

    let colorOrder = { green: 0, neutral: 1, red: 2 };

    return [...songNames].sort((a, b) => {
        let colorDiff = colorOrder[songConclusionColors[a]] - colorOrder[songConclusionColors[b]];
        if (colorDiff !== 0) return colorDiff;
        return a.localeCompare(b);
    });
}

function renderSongList(songNames) {
    let sortedSongNames = sortSongNames(songNames);

    songList.innerHTML = sortedSongNames
        .map(
            (songName) => `
    <div class="song-entry" data-song-name="${songName}">
        <span class="song-label">${songName}</span>
        <button type="button" class="song-remove" data-remove-song-name="${songName}" aria-label="Remove">×</button>
    </div>
`,
        )
        .join("");

    songEntries = document.querySelectorAll(".song-entry");
    songRemoveButtons = document.querySelectorAll(".song-remove");

    songEntries.forEach(
        (entry) =>
            (entry.onclick = () => {
                songEntries.forEach((e) => e.classList.remove("active"));
                entry.classList.add("active");
                showSongStats(entry.dataset.songName);
            }),
    );

    songRemoveButtons.forEach(
        (button) =>
            (button.onclick = (event) => {
                event.stopPropagation();
                removeSong(button.dataset.removeSongName);
            }),
    );
}

let songNames = Object.keys(results).filter((songName) => !deletedSongNames.has(songName));

let songEntries;
let songRemoveButtons;

renderSongList(songNames);

function persistDeletedSongs() {
    localStorage.setItem(deletedSongsStorageKey, JSON.stringify(Array.from(deletedSongNames)));
}

function persistLastSelectedSong(songName) {
    localStorage.setItem(lastSelectedSongStorageKey, songName);
}

function removeSong(songName) {
    deletedSongNames.add(songName);
    persistDeletedSongs();

    let songEntry = songList.querySelector(`.song-entry[data-song-name="${CSS.escape(songName)}"]`);
    if (songEntry) songEntry.remove();

    songNames = songNames.filter((name) => name !== songName);

    renderSongList(songNames);

    let storedLastSelectedSong = localStorage.getItem(lastSelectedSongStorageKey);
    if (storedLastSelectedSong === songName) localStorage.removeItem(lastSelectedSongStorageKey);

    let activeEntry = songList.querySelector(".song-entry.active");
    if (!activeEntry && songEntries.length) songEntries[0].click();

    if (!songEntries.length) {
        headerTitle.textContent = "";
        statsContent.innerHTML = "";
    }
}

function joinAlternatives(alts) {
    if (!alts || alts.length === 0) return "";
    if (alts.length === 1) return alts[0];
    return alts.slice(0, -1).join(", ") + " or " + alts[alts.length - 1];
}

function findCommentEntry(group, genreName, fallbackType) {
    if (!group) return null;

    if (group[genreName]) return { ...group[genreName], type: fallbackType };

    for (const type in group) if (group[type][genreName]) return { ...group[type][genreName], type };

    return null;
}

function getCommentEntry(modelName, genreName) {
    return (
        findCommentEntry(comments.all, genreName, "all") || findCommentEntry(comments[modelName], genreName, "model")
    );
}

function renderComment(entry) {
    if (!entry) return "";

    let alts = entry.alts?.length
        ? `<span class="comment-alts">Could also be ${joinAlternatives(entry.alts)
              .split(/,\s*|\s+or\s+/)
              .map((a) => `<span class="comment-alt">${a}</span>`)
              .join(", ")
              .replace(/, ([^,]*)$/, " or $1")}.</span>`
        : "";

    let textInline = entry.comment && !entry.alts ? `<span class="comment-text-inline">${entry.comment}</span>` : "";

    let textBlock = entry.comment && entry.alts ? `<div class="comment-text">${entry.comment}</div>` : "";

    return `
        <div class="genre-comment">
            <span class="comment-type">${entry.type}</span>
            ${alts}${textInline}${textBlock}
        </div>
    `;
}

function renderModelBlock(modelName, genreList, isConclusion) {
    let topProb = genreList[0]?.prob || 0;
    let secondProb = genreList[1]?.prob || 0;
    let probDiff = topProb - secondProb;

    let isGreen = (topProb >= 35 && probDiff >= 10) || probDiff >= 20;
    let isRed = topProb < 20 || probDiff < 5;

    let className =
        "model-block" +
        (isConclusion ? " model-conclusion" : "") +
        (isGreen ? " model-good" : isRed ? " model-bad" : "");

    let prettyModelName = modelName.replaceAll("edm", "EDM");
    prettyModelName = prettyModelName
        .split('_')
        .map(word => word[0].toUpperCase() + word.slice(1))
        .join(' ');
    
    return `
        <div class="${className}">
            <div class="model-content">

                ${!isConclusion ? `<div class="model-name">${prettyModelName}</div>` : ""}

                ${genreList
                    .map((g, i) => {
                        let commentEntry = getCommentEntry(isConclusion ? "all" : modelName, g.genre);

                        return `
                        <div class="genre-row${commentEntry ? " genre-has-comment" : ""}">
                            <div class="genre-label">
                                <span>${i + 1}. ${g.genre}</span>
                                <span>${g.prob.toFixed(2)}%</span>
                            </div>

                            <div class="progress">
                                <div class="progress-bar" style="width:${g.prob}%"></div>
                            </div>

                            ${renderComment(commentEntry)}
                        </div>
                    `;
                    })
                    .join("")}

            </div>
        </div>
    `;
}

function showSongStats(songName) {
    headerTitle.textContent = songName;
    persistLastSelectedSong(songName);

    statsContent.innerHTML = `
        <div class="conclusion-section">
            ${renderModelBlock("Conclusion", songConclusions[songName], true)}
        </div>

        <div class="models-grid">
            ${Object.entries(results[songName])
                .map(([model, genres]) => renderModelBlock(model, genres, false))
                .join("")}
        </div>
    `;
}

let storedLastSelectedSong = localStorage.getItem(lastSelectedSongStorageKey);
let storedEntry = storedLastSelectedSong
    ? songList.querySelector(`.song-entry[data-song-name="${CSS.escape(storedLastSelectedSong)}"]`)
    : null;

if (storedEntry) storedEntry.click();
else if (songEntries.length) songEntries[0].click();

searchInput.oninput = () => {
    let searchValue = searchInput.value.toLowerCase();

    songEntries.forEach(
        (entry) => (entry.style.display = entry.dataset.songName.toLowerCase().includes(searchValue) ? "flex" : "none"),
    );
};

sortSelect.onchange = () => {
    let activeSongName = songList.querySelector(".song-entry.active")?.dataset.songName;
    renderSongList(songNames);

    if (activeSongName) {
        let newActiveEntry = songList.querySelector(`.song-entry[data-song-name="${CSS.escape(activeSongName)}"]`);
        if (newActiveEntry) newActiveEntry.click();
    }
};

function buildFilteredResultsObject() {
    let filteredResults = {};
    let deletedSongNameArray = Array.from(deletedSongNames);

    Object.keys(results).forEach((songName) => {
        if (deletedSongNameArray.includes(songName)) return;
        filteredResults[songName] = results[songName];
    });

    return filteredResults;
}

function downloadFilteredResultsJs() {
    let filteredResults = buildFilteredResultsObject();
    let fileContents = "let results = " + JSON.stringify(filteredResults, null, 4) + ";\n";

    let blob = new Blob([fileContents], { type: "application/javascript;charset=utf-8" });
    let blobUrl = URL.createObjectURL(blob);

    let downloadLink = document.createElement("a");
    downloadLink.href = blobUrl;
    downloadLink.download = "results.js";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    downloadLink.remove();

    URL.revokeObjectURL(blobUrl);
}

downloadResultsButton.onclick = downloadFilteredResultsJs;
