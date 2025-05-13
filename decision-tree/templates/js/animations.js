const workspace = document.getElementById("workspace");
const gridContainer = document.getElementById("gridContainer");
const zoomSlider = document.getElementById("zoomSlider");
const zoomValue = document.getElementById("zoomValue");
let scale = 1;
let isDragging = false;
let startX, startY;
let offsetX = 0,
    offsetY = 0;

function applyTransform() {
    gridContainer.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
}

workspace.addEventListener("mousedown", (event) => {
    isDragging = true;
    startX = event.clientX - offsetX;
    startY = event.clientY - offsetY;
    workspace.classList.add("cursor-grabbing");
});

document.addEventListener("mousemove", (event) => {
    if (isDragging) {
        offsetX = event.clientX - startX;
        offsetY = event.clientY - startY;
        applyTransform();
    }
});

document.addEventListener("mouseup", () => {
    isDragging = false;
    workspace.classList.remove("cursor-grabbing");
});

zoomSlider.addEventListener("input", (event) => {
    scale = zoomSlider.value / 100;
    zoomValue.textContent = `${zoomSlider.value}%`
    applyTransform();
});

workspace.addEventListener("wheel", (event) => {
    event.preventDefault();
    const zoomFactor = 0.1;
    const mouseX = 165;
    const mouseY = 8;
    const rect = workspace.getBoundingClientRect();

    const offsetXBeforeZoom = (mouseX - rect.left - offsetX) / scale;
    const offsetYBeforeZoom = (mouseY - rect.top - offsetY) / scale;

    const delta = Math.sign(event.deltaY) * zoomFactor;
    scale = Math.min(Math.max(0.2, scale - delta), 3);

    offsetX = mouseX - offsetXBeforeZoom * scale - rect.left;
    offsetY = mouseY - offsetYBeforeZoom * scale - rect.top;
    zoomSlider.value = scale * 100;
    zoomValue.textContent = `${zoomSlider.value}%`;
    applyTransform();
});
