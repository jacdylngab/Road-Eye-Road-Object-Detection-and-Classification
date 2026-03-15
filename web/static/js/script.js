function uploadImage() {
    const file = document.getElementById("fileInput").files[0];

    const formData = new FormData();
    formData.append("image", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        document.getElementById("result").src = url;
    });
}

/*
.files[0] — gets the first (and only) selected file
FormData — a special object for sending files over HTTP (think of it like a container)
formData.append("image", file) — puts the file in that container with the key "image"
There's no Content-Type header — when sending FormData, the browser sets it automatically
response.blob() — reads the response as raw binary (since the server is sending back an image, not JSON)
URL.createObjectURL(blob) — converts the binary into a URL the <img> tag can use
*/
