let songList = document.querySelector("#songs #list");

songListHtml = "";
for (const songName in results) {
    songListHtml += `
        <div>
            <span>${songName}</span>
        `;
    for (const modelName in results[songName]) {

    }

    songListHtml += "</div>";
}

songList.innerHTML = songListHtml;
