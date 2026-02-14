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

function showSongStats(songName) {
    statsPlaceholder.style.display = "none";

    statsHtml = `
        <div class="song-title">${songName}</div>
        <div class="models-grid">
    `;

    for (const modelName in results[songName]) {
        statsHtml += `
            <div class="model-block">
                <div class="model-name">${modelName}</div>
        `;

        let genreList = results[songName][modelName];

        for (let genreIndex = 0; genreIndex < genreList.length; genreIndex++) {
            let genreName = genreList[genreIndex].genre;
            let probability = genreList[genreIndex].prob;

            statsHtml += `
                <div class="genre-row">
                    <div class="genre-label">
                        <span>${genreIndex + 1}. ${genreName}</span>
                        <span>${probability.toFixed(2)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" style="width: ${probability}%"></div>
                    </div>
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
