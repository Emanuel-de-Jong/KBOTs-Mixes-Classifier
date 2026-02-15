let songList = document.querySelector("#songs #list");
let statsContent = document.querySelector("#stats-content");
let searchInput = document.querySelector("#search");
let headerTitle = document.querySelector("#header-title");

let songNames = Object.keys(results);

songList.innerHTML = songNames
    .map(
        (songName) => `
    <div class="song-entry" data-song-name="${songName}">
        <span>${songName}</span>
    </div>
`,
    )
    .join("");

let songEntries = document.querySelectorAll(".song-entry");

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

    let className =
        "model-block" +
        (isConclusion ? " model-conclusion" : "") +
        (topProb >= 75 ? " model-good" : topProb < 50 ? " model-bad" : "");

    return `
        <div class="${className}">
            <div class="model-content">

                ${!isConclusion ? `<div class="model-name">${modelName}</div>` : ""}

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

    statsContent.innerHTML = `
        <div class="conclusion-section">
            ${renderModelBlock("Conclusion", calculateConclusion(songName), true)}
        </div>

        <div class="models-grid">
            ${Object.entries(results[songName])
                .map(([model, genres]) => renderModelBlock(model, genres, false))
                .join("")}
        </div>
    `;
}

songEntries.forEach(
    (entry) =>
        (entry.onclick = () => {
            songEntries.forEach((e) => e.classList.remove("active"));
            entry.classList.add("active");
            showSongStats(entry.dataset.songName);
        }),
);

if (songEntries.length) songEntries[0].click();

searchInput.oninput = () => {
    let searchValue = searchInput.value.toLowerCase();

    songEntries.forEach(
        (entry) =>
            (entry.style.display = entry.dataset.songName.toLowerCase().includes(searchValue) ? "block" : "none"),
    );
};
