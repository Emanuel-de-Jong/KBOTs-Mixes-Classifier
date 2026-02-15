let songList = document.querySelector("#songs #list");
let statsContent = document.querySelector("#stats-content");
let statsPlaceholder = document.querySelector("#stats-placeholder");
let searchInput = document.querySelector("#search");

songListHtml = "";
for (const songName in results) {
    songListHtml += `
        <div class="song-entry" data-song-name="${songName}">
            <span>${songName}</span>
        </div>
    `;
}

songList.innerHTML = songListHtml;

let songEntries = document.querySelectorAll(".song-entry");

function joinAlternatives(alts) {
    if (!alts || alts.length === 0) return "";
    if (alts.length === 1) return alts[0];
    if (alts.length === 2) return `${alts[0]} or ${alts[1]}`;

    let allButLast = alts.slice(0, -1).join(", ");
    let lastAlt = alts[alts.length - 1];
    return `${allButLast} or ${lastAlt}`;
}

function findCommentEntryInGroup(commentGroup, genreName, typeFallback) {
    if (!commentGroup || typeof commentGroup !== "object") return null;

    let directEntry = commentGroup[genreName];
    if (directEntry && (directEntry.comment || directEntry.alts)) {
        return {
            comment: directEntry.comment,
            alts: directEntry.alts,
            type: typeFallback,
        };
    }

    for (const subcategoryName in commentGroup) {
        let subcategoryGroup = commentGroup[subcategoryName];

        if (!subcategoryGroup || typeof subcategoryGroup !== "object") continue;

        let subcategoryEntry = subcategoryGroup[genreName];
        if (subcategoryEntry && (subcategoryEntry.comment || subcategoryEntry.alts)) {
            return {
                comment: subcategoryEntry.comment,
                alts: subcategoryEntry.alts,
                type: subcategoryName,
            };
        }
    }

    return null;
}

function getCommentEntry(modelName, genreName) {
    let allGroup = comments && comments.all ? comments.all : null;
    let allEntry = findCommentEntryInGroup(allGroup, genreName, "all");

    if (allEntry) {
        return {
            comment: allEntry.comment,
            alts: allEntry.alts,
            type: allEntry.type,
            source: "all",
        };
    }

    let modelGroup = comments && comments[modelName] ? comments[modelName] : null;
    let modelEntry = findCommentEntryInGroup(modelGroup, genreName, "model");

    if (modelEntry) {
        return {
            comment: modelEntry.comment,
            alts: modelEntry.alts,
            type: "model",
            source: "model",
        };
    }

    return null;
}

function showSongStats(songName) {
    statsPlaceholder.style.display = "none";

    statsHtml = `
        <div class="song-title">${songName}</div>
        <div class="models-grid">
    `;

    for (const modelName in results[songName]) {
        let genreList = results[songName][modelName];

        let modelBlockClass = "model-block";
        if (genreList && genreList.length > 0) {
            let topProbability = genreList[0].prob;

            if (topProbability >= 75) {
                modelBlockClass += " model-good";
            } else if (topProbability < 50) {
                modelBlockClass += " model-bad";
            }
        }

        statsHtml += `
            <div class="${modelBlockClass}">
                <div class="model-name">${modelName}</div>
        `;

        for (let genreIndex = 0; genreIndex < genreList.length; genreIndex++) {
            let genreName = genreList[genreIndex].genre;
            let probability = genreList[genreIndex].prob;

            let commentEntry = getCommentEntry(modelName, genreName);
            let hasComment = commentEntry && (commentEntry.comment || (commentEntry.alts && commentEntry.alts.length > 0));

            let commentHtml = "";
            if (hasComment) {
                let typeLabel = commentEntry.type;

                commentHtml += `
                    <div class="genre-comment">
                        <span class="comment-type">${typeLabel}</span>
                `;

                if (commentEntry.comment) {
                    commentHtml += `
                        <span class="comment-text">${commentEntry.comment}</span>
                    `;
                }

                if (commentEntry.alts && commentEntry.alts.length > 0) {
                    let alternativesText = joinAlternatives(commentEntry.alts);

                    let formattedAlternatives = alternativesText
                        .split(/,\s*|\s+or\s+/)
                        .filter(function (altText) {
                            return altText && altText.trim().length > 0;
                        })
                        .map(function (altText) {
                            return `<span class="comment-alt">${altText.trim()}</span>`;
                        });

                    if (commentEntry.alts.length === 1) {
                        commentHtml += `
                            <span class="comment-alts">Could also be ${formattedAlternatives[0]}.</span>
                        `;
                    } else if (commentEntry.alts.length === 2) {
                        commentHtml += `
                            <span class="comment-alts">Could also be ${formattedAlternatives[0]} or ${formattedAlternatives[1]}.</span>
                        `;
                    } else {
                        let allButLast = formattedAlternatives.slice(0, -1).join(", ");
                        let lastAlt = formattedAlternatives[formattedAlternatives.length - 1];

                        commentHtml += `
                            <span class="comment-alts">Could also be ${allButLast} or ${lastAlt}.</span>
                        `;
                    }
                }

                commentHtml += `
                    </div>
                `;
            }

            statsHtml += `
                <div class="genre-row${hasComment ? " genre-has-comment" : ""}">
                    <div class="genre-label">
                        <span>${genreIndex + 1}. ${genreName}</span>
                        <span>${probability.toFixed(2)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" style="width: ${probability}%"></div>
                    </div>
                    ${commentHtml}
                </div>
            `;
        }

        statsHtml += `
            </div>
        `;
    }

    statsHtml += `
        </div>
    `;

    statsContent.innerHTML = statsHtml;
}

songEntries.forEach(function (entryElement) {
    entryElement.addEventListener("click", function () {
        songEntries.forEach(function (removeElement) {
            removeElement.classList.remove("active");
        });

        entryElement.classList.add("active");

        let songName = entryElement.getAttribute("data-song-name");

        showSongStats(songName);
    });
});

if (songEntries.length > 0) {
    let firstEntryElement = songEntries[0];
    firstEntryElement.classList.add("active");

    let firstSongName = firstEntryElement.getAttribute("data-song-name");

    showSongStats(firstSongName);
}

searchInput.addEventListener("input", function () {
    let searchValue = searchInput.value.toLowerCase();

    songEntries.forEach(function (entryElement) {
        let songName = entryElement.getAttribute("data-song-name").toLowerCase();

        if (songName.includes(searchValue)) {
            entryElement.style.display = "block";
        } else {
            entryElement.style.display = "none";
        }
    });
});
